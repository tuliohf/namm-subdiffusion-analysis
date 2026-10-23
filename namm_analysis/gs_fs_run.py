#!/usr/bin/env python3
"""
gs_fs_run.py

Compute, from a LAMMPS custom dump of unwrapped coordinates (id xu yu zu):
  - P_par(r,t)   = 2*pi*r*Gs,par(r,t)   in-plane radial displacement probability  (Eq. P_par)
  - Gs,perp(z,t)                        out-of-plane (signed) displacement density
  - Fs,par(q,t)  = <J0(q*dr_par)>       in-plane self-intermediate scattering fn (Eq. fs)
  - Fs,perp(q,t) = <cos(q*dz)>          out-of-plane self-ISF
  - NGP_par(t), NGP_perp(t)             directional non-Gaussian parameters (Eq. ngp_dir)
  - MSD_par(t), MSD_perp(t)             for cross-checking against compute msd output

Multi-origin averaging: for each log-spaced target lag, every valid frame pair
(t0, t0+t) actually present in the (non-uniformly sampled, 5-phase) trajectory
is used, subject to a per-lag cap on the number of origins for speed.

Provenance: recovered verbatim from the analysis session of August 2026
(Na-MMT subdiffusion paper, J. Phys. Chem. C). Changes made afterwards: this
header paragraph; the `--weighted` option (off by default, so the default
output is unchanged); and np.fromstring replaced by an equivalent split-based
parse, since np.fromstring is deprecated in NumPy 2. Note: MSD_par here is the full in-plane value <dx^2+dy^2>;
the paper's MSD_par is the per-component value (1/2 of it).

Usage:
  python3 gs_fs_run.py -i dump.Ow_unwrap_16_1.lammpstrj -o out --n-lags 25 \
      --n-bins 60 --max-pairs 3000 --q 2.8
"""
import argparse, json, os, sys, time
import numpy as np
from scipy.special import j0

def parse_lammpstrj(fpath):
    """Fast parser assuming fixed atom count per frame, columns: id xu yu zu."""
    with open(fpath, "r") as f:
        lines = f.readlines()

    i = 0
    n = len(lines)
    timesteps = []
    frames = []  # list of (natoms,3) arrays, id-sorted
    natoms = None
    while i < n:
        if not lines[i].startswith("ITEM: TIMESTEP"):
            i += 1
            continue
        ts = int(lines[i+1])
        na = int(lines[i+3])
        if natoms is None:
            natoms = na
        elif na != natoms:
            raise ValueError(f"atom count changed ({natoms}->{na}) at ts={ts}; "
                              f"fixed-atom-count assumption violated")
        # header: TIMESTEP, ts, NUMBER OF ATOMS, na, BOX BOUNDS, 3 box lines, ATOMS line = 9 lines
        data_start = i + 9
        data_end = data_start + na
        block = lines[data_start:data_end]
        arr = np.array(" ".join(block).split(), dtype=np.float64).reshape(na, 4)
        arr = arr[arr[:, 0].argsort()]  # sort by id for consistent atom ordering
        timesteps.append(ts)
        frames.append(arr[:, 1:4])
        i = data_end
    timesteps = np.array(timesteps, dtype=np.int64)
    coords = np.stack(frames, axis=0)  # (nframes, natoms, 3)
    order = np.argsort(timesteps)
    return timesteps[order], coords[order]


def pick_lag_targets(timesteps, n_lags):
    dt_min = np.min(np.diff(timesteps))
    span = timesteps[-1] - timesteps[0]
    lo = max(dt_min, 1)
    return np.unique(np.round(np.logspace(np.log10(lo), np.log10(span), n_lags)).astype(np.int64))


def gather_pairs_for_lag(timesteps, target_lag, max_pairs, rng):
    """Find (i,j) index pairs with timesteps[j]-timesteps[i] closest to target_lag,
    picking one best j per i (nearest match), then subsample origins if too many."""
    n = len(timesteps)
    # for each i, find j via searchsorted on timesteps for value timesteps[i]+target_lag
    target_vals = timesteps + target_lag
    j_candidates = np.searchsorted(timesteps, target_vals)
    valid_i = []
    valid_j = []
    for i in range(n):
        j = j_candidates[i]
        best_j = -1
        best_err = None
        for jj in (j - 1, j):
            if 0 <= jj < n:
                err = abs((timesteps[jj] - timesteps[i]) - target_lag)
                if best_err is None or err < best_err:
                    best_err = err
                    best_j = jj
        if best_j > i and best_err <= 0.15 * target_lag:
            valid_i.append(i)
            valid_j.append(best_j)
    if len(valid_i) == 0:
        return None, None, 0
    valid_i = np.array(valid_i)
    valid_j = np.array(valid_j)
    actual_lag = np.median(timesteps[valid_j] - timesteps[valid_i])
    n_pairs = len(valid_i)
    if max_pairs > 0 and n_pairs > max_pairs:
        sel = rng.choice(n_pairs, size=max_pairs, replace=False)
        sel.sort()
        valid_i = valid_i[sel]
        valid_j = valid_j[sel]
    return valid_i, valid_j, actual_lag


def compute_all(timesteps, coords, n_lags, n_bins, max_pairs, q_list, dt_fs, seed=0,
                weighted=False):
    rng = np.random.default_rng(seed)
    lag_targets = pick_lag_targets(timesteps, n_lags)

    fs_rows, gs_par_rows, gs_perp_rows, ngp_rows = [], [], [], []

    for lag in lag_targets:
        if weighted:
            from sampling import gather_pairs_for_lag_weighted
            vi, vj, actual_lag = gather_pairs_for_lag_weighted(timesteps, lag, max_pairs, rng)
        else:
            vi, vj, actual_lag = gather_pairs_for_lag(timesteps, lag, max_pairs, rng)
        if vi is None:
            continue
        n_origins = len(vi)
        dr = coords[vj] - coords[vi]          # (n_origins, natoms, 3)
        dx, dy, dz = dr[..., 0], dr[..., 1], dr[..., 2]
        dr_par = np.sqrt(dx**2 + dy**2).ravel()
        dz_flat = dz.ravel()

        t_ns = actual_lag * dt_fs * 1e-6  # steps * fs -> ns

        # ---- MSD & directional NGP ----
        msd_par = np.mean(dr_par**2)
        msd_perp = np.mean(dz_flat**2)
        m4_par = np.mean(dr_par**4)
        m4_perp = np.mean(dz_flat**4)
        ngp_par = m4_par / (2.0 * msd_par**2) - 1.0 if msd_par > 0 else np.nan
        ngp_perp = m4_perp / (3.0 * msd_perp**2) - 1.0 if msd_perp > 0 else np.nan
        reliable = int(n_origins >= 10)
        ngp_rows.append(dict(t_ns=t_ns, lag_steps=int(actual_lag), n_origins=n_origins,
                              msd_par=msd_par, msd_perp=msd_perp,
                              ngp_par=ngp_par, ngp_perp=ngp_perp, reliable=reliable))

        # ---- Fs(q,t) ----
        for q in q_list:
            fs_par = float(np.mean(j0(q * dr_par)))
            fs_perp = float(np.mean(np.cos(q * dz_flat)))
            fs_rows.append(dict(t_ns=t_ns, lag_steps=int(actual_lag), q=q,
                                 n_origins=n_origins, fs_par=fs_par, fs_perp=fs_perp))

        # ---- Gs / P_par, Gs_perp histograms ----
        rmax = np.percentile(dr_par, 99.5) if dr_par.size else 1.0
        rmax = max(rmax, 1e-6)
        bins_r = np.linspace(0, rmax, n_bins + 1)
        hist_r, edges_r = np.histogram(dr_par, bins=bins_r, density=True)
        centers_r = 0.5 * (edges_r[:-1] + edges_r[1:])
        for rc, val in zip(centers_r, hist_r):
            gs_par_rows.append(dict(t_ns=t_ns, lag_steps=int(actual_lag),
                                     r=rc, P_par=val, n_origins=n_origins))

        zmax = np.percentile(np.abs(dz_flat), 99.5) if dz_flat.size else 1.0
        zmax = max(zmax, 1e-6)
        bins_z = np.linspace(-zmax, zmax, n_bins + 1)
        hist_z, edges_z = np.histogram(dz_flat, bins=bins_z, density=True)
        centers_z = 0.5 * (edges_z[:-1] + edges_z[1:])
        for zc, val in zip(centers_z, hist_z):
            gs_perp_rows.append(dict(t_ns=t_ns, lag_steps=int(actual_lag),
                                      z=zc, Gs_perp=val, n_origins=n_origins))

    return fs_rows, gs_par_rows, gs_perp_rows, ngp_rows


def write_csv(rows, path, fieldnames):
    import csv
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def process_file(fpath, out_root, n_lags, n_bins, max_pairs, q_list, dt_fs, weighted=False):
    base = os.path.splitext(os.path.basename(fpath))[0]
    out_dir = os.path.join(out_root, base)
    os.makedirs(out_dir, exist_ok=True)

    t0 = time.time()
    timesteps, coords = parse_lammpstrj(fpath)
    natoms = coords.shape[1]
    nframes = coords.shape[0]

    fs_rows, gs_par_rows, gs_perp_rows, ngp_rows = compute_all(
        timesteps, coords, n_lags, n_bins, max_pairs, q_list, dt_fs, weighted=weighted)

    write_csv(fs_rows, os.path.join(out_dir, "fs.csv"),
              ["t_ns", "lag_steps", "q", "n_origins", "fs_par", "fs_perp"])
    write_csv(gs_par_rows, os.path.join(out_dir, "gs_par.csv"),
              ["t_ns", "lag_steps", "r", "P_par", "n_origins"])
    write_csv(gs_perp_rows, os.path.join(out_dir, "gs_perp.csv"),
              ["t_ns", "lag_steps", "z", "Gs_perp", "n_origins"])
    write_csv(ngp_rows, os.path.join(out_dir, "ngp.csv"),
              ["t_ns", "lag_steps", "n_origins", "msd_par", "msd_perp",
               "ngp_par", "ngp_perp", "reliable"])

    summary = dict(file=fpath, natoms=natoms, nframes=nframes,
                    t_span_steps=int(timesteps[-1] - timesteps[0]),
                    n_lags_requested=n_lags, n_bins=n_bins, max_pairs=max_pairs,
                    q_list=q_list, dt_fs=dt_fs, weighted=weighted,
                    elapsed_s=time.time() - t0)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description="Compute Gs/Fs/NGP from LAMMPS unwrapped dumps")
    p.add_argument("-i", "--input", nargs="+", required=True)
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--n-lags", type=int, default=25)
    p.add_argument("--n-bins", type=int, default=60)
    p.add_argument("--max-pairs", type=int, default=3000)
    p.add_argument("--q", type=float, nargs="+", default=[2.8])
    p.add_argument("--dt-fs", type=float, default=2.0)
    p.add_argument("--weighted", action="store_true",
                   help="time-density-weighted origin sampling (non-uniform dump schedules)")
    args = p.parse_args(argv)

    os.makedirs(args.output, exist_ok=True)
    all_summaries = []
    for fpath in args.input:
        print(f"[gs_fs_run] processing {fpath} ...", flush=True)
        s = process_file(fpath, args.output, args.n_lags, args.n_bins,
                          args.max_pairs, args.q, args.dt_fs, weighted=args.weighted)
        print(f"[gs_fs_run]   done: natoms={s['natoms']} nframes={s['nframes']} "
              f"elapsed={s['elapsed_s']:.1f}s", flush=True)
        all_summaries.append(s)
    with open(os.path.join(args.output, "batch_summary.json"), "w") as f:
        json.dump(all_summaries, f, indent=2)


if __name__ == "__main__":
    main()

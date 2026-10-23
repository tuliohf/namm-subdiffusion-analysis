#!/usr/bin/env python3
"""
tamsd.py -- time-averaged MSD of individual particles and ergodicity diagnostics.

    delta_i^2(Delta) = 1/(T-Delta) int_0^{T-Delta} (1/2) |r_par,i(t+Delta) - r_par,i(t)|^2 dt

(per-component in-plane value, consistent with MSD_par in the paper), evaluated
on the uniformly sampled part of the dump (phase 5, every 40 ps).

Diagnostics (SI "Time-averaged MSD of individual trajectories and ergodicity"):
  (i)   <delta^2>(Delta) vs ensemble MSD_par from the same trajectory
        (single origin t=0, as in LAMMPS compute msd), and their ratio
  (ii)  stationarity: <delta^2>(Delta) evaluated in n_blocks consecutive blocks
  (iii) EB(Delta) = <(delta^2)^2>/<delta^2>^2 - 1, compared with 2 Delta/(3T),
        the Brownian value for the per-component in-plane TAMSD, and
        EB as a function of measurement time T at fixed Delta
  (iv)  optional split by gallery (the two interlayers are inequivalent)

Jump counts per ion, the fourth diagnostic, come from jumps.py.

Provenance: rebuilt in October 2026 from the SI text of the revision session
(the session scripts were not printed in full and are not recoverable verbatim).

Example
  python3 tamsd.py -i dump.Iw_unwrap_16_*.lammpstrj -o tamsd_Na_16 --by-gallery
"""
import argparse, json, os
import numpy as np

from io_lammps import read_dump, uniform_tail


def tamsd_fft(x):
    """Per-particle TAMSD for all lags k = 0..N-1 (FFT algorithm).
    x: (N, P, d). Returns (N, P) = mean over origins of |x(n+k)-x(n)|^2."""
    N = x.shape[0]
    D = np.sum(x ** 2, axis=-1)                                  # (N, P)
    D = np.r_[D, np.zeros((1, D.shape[1]))]
    Q = 2 * D.sum(axis=0)
    S1 = np.empty((N, x.shape[1]))
    for k in range(N):
        Q = Q - D[k - 1] - D[N - k]
        S1[k] = Q / (N - k)
    F = np.fft.rfft(x, n=2 * N, axis=0)
    ac = np.fft.irfft(F * F.conj(), axis=0)[:N].sum(axis=-1)     # sum over dims
    S2 = ac / (N - np.arange(N))[:, None]
    return S1 - 2 * S2


def lag_indices(N, n=30):
    return np.unique(np.round(np.logspace(0, np.log10(max(N - 2, 1)), n)).astype(int))


def analyze(xy, dt_ns, n_blocks=6, n_lags=30):
    """xy: (N, P, 2) unwrapped in-plane coordinates on a uniform grid."""
    N, P, _ = xy.shape
    T = (N - 1) * dt_ns
    ks = lag_indices(N, n_lags)
    full = 0.5 * tamsd_fft(xy)[ks]                               # (nk, P), per component
    mean = full.mean(axis=1)
    eb = (full ** 2).mean(axis=1) / mean ** 2 - 1
    ens = 0.5 * np.mean(np.sum((xy[ks] - xy[0]) ** 2, axis=-1), axis=1)
    out = dict(
        Delta_ns=ks * dt_ns, T_ns=T, n_particles=P,
        tamsd_mean=mean, ensemble_msd=ens, ratio=mean / ens,
        EB=eb, EB_brownian=2 * ks * dt_ns / (3 * T),
        tamsd_individual=full,
    )
    # stationarity: blocks
    nb = N // n_blocks
    if nb > 3:
        kb = ks[ks < nb - 1]
        blocks = [0.5 * tamsd_fft(xy[b * nb:(b + 1) * nb])[kb].mean(axis=1) for b in range(n_blocks)]
        out["block_Delta_ns"] = kb * dt_ns
        out["block_tamsd"] = np.array(blocks)                    # (n_blocks, nkb)
    # EB vs measurement time at a fixed lag (the shortest lag >= 1 ns if available)
    k0 = int(ks[np.searchsorted(ks * dt_ns, 1.0)]) if ks[-1] * dt_ns >= 1.0 else int(ks[len(ks) // 2])
    Ts, EBT = [], []
    for frac in (0.125, 0.25, 0.5, 1.0):
        n = int(frac * N)
        if n > 3 * k0:
            d = 0.5 * np.mean(np.sum((xy[k0:n] - xy[:n - k0]) ** 2, axis=-1), axis=0)
            Ts.append((n - 1) * dt_ns)
            EBT.append(np.mean(d ** 2) / np.mean(d) ** 2 - 1)
    out.update(EB_T_lag_ns=k0 * dt_ns, EB_T_T_ns=np.array(Ts), EB_T=np.array(EBT))
    return out


def gallery_labels(z, box):
    from residence import gallery_centers, assign_gallery
    lo, Lz = box[2, 0], box[2, 1] - box[2, 0]
    c = gallery_centers(z, lo, Lz)
    lab = assign_gallery(z, c, lo, Lz)
    # one label per particle (no inter-gallery exchange): majority vote
    return np.round(lab.mean(axis=0)).astype(int)


def save(res, path):
    flat = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in res.items()
            if k != "tamsd_individual"}
    with open(path + ".json", "w") as f:
        json.dump(flat, f, indent=1)
    np.savetxt(path + "_individual.csv",
               np.c_[res["Delta_ns"], res["tamsd_individual"]], delimiter=",",
               header="Delta_ns," + ",".join(f"p{i}" for i in range(res["n_particles"])),
               comments="")


def main(argv=None):
    p = argparse.ArgumentParser(description="Time-averaged MSD and ergodicity diagnostics")
    p.add_argument("-i", "--input", nargs="+", required=True,
                   help="one dump per trajectory (replicas are pooled as extra particles)")
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--dt-fs", type=float, default=2.0)
    p.add_argument("--step", type=int, default=None, help="uniform dump interval in steps (default: most frequent)")
    p.add_argument("--n-blocks", type=int, default=6)
    p.add_argument("--by-gallery", action="store_true")
    a = p.parse_args(argv)
    os.makedirs(a.output, exist_ok=True)

    xy_all, gal_all, dt_ns, Nmin = [], [], None, None
    for f in a.input:
        ts, c, _, box = read_dump(f)
        idx, step = uniform_tail(ts, a.step)
        dt_ns = step * a.dt_fs * 1e-6
        xy_all.append(c[idx, :, :2])
        gal_all.append(gallery_labels(c[idx, :, 2], box))
        Nmin = len(idx) if Nmin is None else min(Nmin, len(idx))
    xy = np.concatenate([x[:Nmin] for x in xy_all], axis=1)      # common measurement time
    gal = np.concatenate(gal_all)

    res = analyze(xy, dt_ns, a.n_blocks)
    save(res, os.path.join(a.output, "all"))
    if a.by_gallery:
        for g in (0, 1):
            save(analyze(xy[:, gal == g], dt_ns, a.n_blocks), os.path.join(a.output, f"gallery{g}"))
    m = (res["Delta_ns"] >= 0.04) & (res["Delta_ns"] <= 40)
    print(f"T = {res['T_ns']:.1f} ns, particles = {res['n_particles']}, "
          f"<TAMSD>/MSD averaged over 0.04-40 ns = {np.nanmean(res['ratio'][m]):.3f}")


if __name__ == "__main__":
    main()

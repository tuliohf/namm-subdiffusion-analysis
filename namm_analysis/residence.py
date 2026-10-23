#!/usr/bin/env python3
"""
residence.py -- continuous residence (survival) times in Na-MMT interlayers.

Observables (paper, Methods / SI "Residence-time analysis"):
  tau_site      Na+ persistence in the same lateral adsorption basin. Basins are
                the periodic 2D Voronoi cells of the substitution sites facing the
                Na+ gallery (nearest-site assignment = Voronoi tessellation).
  tau_inner     Na+ continuously inner-sphere   (r_min(Na-O_fw) < 3.3 A)
  tau_outer     Na+ continuously outer-sphere   (3.3 <= r_min < 5.5 A)
  tau_mobile    Na+ continuously mobile         (r_min >= 5.5 A)
  tau_Na-shell  a water O continuously within 3.15 A of the SAME Na+
  tau_surf      water continuously within 3.5 A of any framework O
                (geometric criterion, no angular H-bond condition)

Survival function of state s, conditioned on starting in s:
    S_s(t) = < 1[state continuously = s on [t0, t0+t]] >  /  < 1[state(t0) = s] >
pooled over particles and time origins (and later over replicas), with
time-density-weighted origins, phase-boundary exclusion, phase-5 grid snapping
and a cumulative-minimum filter (see sampling.py). Continuity is checked on the
stored frames, so sojourns shorter than the local dump interval are not resolved.

    tau = integral_0^T S_s(t) dt   (trapezoid on the lag grid, S(0)=1)
If S_s(T) > 0.05 no tau is reported (non-decaying plateau / lower bound);
no extrapolation beyond the simulated window.

Gallery assignment uses the z-positions of the mobile species (two-cluster
split of the wrapped z of Na+ and water), NOT the substitution atoms, which sit
mid-layer and are symmetric between galleries.

Provenance: rebuilt in October 2026 from the methods text, the SI and the code
fragments of the August 2026 session (original files compute_coordination_full.py,
compute_water_residence.py, residence_setup.py, not recoverable verbatim). The
cutoffs and corrections are those of the paper; agreement with the published
values should be checked on one state before relying on this file.

Inputs
  --na     Na+ dump (id xu yu zu)
  --water  water-oxygen dump (id xu yu zu)
  --framework  x y z of the (frozen) framework oxygens, plain table
  --sites      x y z of the substitution atoms (oct. Mg, tet. Al), plain table

Example
  python3 residence.py --na dump.Iw_unwrap_16_1.lammpstrj \
      --water dump.Ow_unwrap_16_1.lammpstrj --framework Ofw_16.xyz \
      --sites subs_16.xyz -o res_16_1
"""
import argparse, json, os
import numpy as np
from scipy.spatial import cKDTree

from io_lammps import read_dump, common_frames, read_xyz_table, wrap
from sampling import gather_pairs_split, snap_to_grid, enforce_monotone

R_INNER, R_OUTER = 3.3, 5.5     # A, minima of the bimodal r_min(Na-O_fw) distribution
R_SHELL = 3.15                  # A, first minimum of g(Na-Ow)
R_SURF = 3.5                    # A, conventional geometric water-surface criterion
PLATEAU_THRESHOLD = 0.05


# ----------------------------------------------------------------------------
# geometry helpers
# ----------------------------------------------------------------------------
def gallery_centers(z_mobile, lo, Lz):
    """Two gallery centres from the wrapped z of the mobile species (circular
    two-cluster split: largest empty gap in z defines the clay layer)."""
    z = np.sort(np.mod(z_mobile.ravel() - lo, Lz))
    gaps = np.diff(np.r_[z, z[0] + Lz])
    k = np.argmax(gaps)
    zcut = z[k] + gaps[k] / 2                       # inside the first clay layer
    zs = np.sort(np.mod(z - zcut, Lz))              # one unbroken axis through both galleries
    k2 = np.argmax(np.diff(zs))                     # second clay layer
    c = np.array([zs[:k2 + 1].mean(), zs[k2 + 1:].mean()])
    return np.sort(np.mod(c + zcut, Lz)) + lo


def assign_gallery(z, centers, lo, Lz):
    d = np.abs(((z[..., None] - centers) + Lz / 2) % Lz - Lz / 2)
    return np.argmin(d, axis=-1)


def kdtree_xyz(points, box):
    lo, L = box[:, 0], box[:, 1] - box[:, 0]
    return cKDTree(wrap(points, lo, L) - lo, boxsize=L), lo, L


# ----------------------------------------------------------------------------
# state trajectories
# ----------------------------------------------------------------------------
def states_site(na, sites, box, d001):
    """Index of the nearest substitution site (2D periodic) among the sites
    facing the Na+ gallery. Returns (nf, nNa) int array."""
    lo, L = box[:, 0], box[:, 1] - box[:, 0]
    cz = gallery_centers(na[..., 2], lo[2], L[2])
    g_na = assign_gallery(na[..., 2], cz, lo[2], L[2])           # (nf, nNa)
    # a site faces gallery g if it lies within d001/2 (+0.5 A) of its centre:
    # octahedral sites (mid-layer) face both galleries, tetrahedral ones only one
    dz_site = np.abs(((sites[:, 2][:, None] - cz) + L[2] / 2) % L[2] - L[2] / 2)
    faces = dz_site <= d001 / 2 + 0.5                             # (nsite, 2)
    out = np.full(g_na.shape, -1, dtype=np.int64)
    for g in (0, 1):
        idx = np.where(faces[:, g])[0]
        tree = cKDTree(wrap(sites[idx, :2], lo[:2], L[:2]) - lo[:2], boxsize=L[:2])
        m = g_na == g
        _, k = tree.query(wrap(na[..., :2][m], lo[:2], L[:2]) - lo[:2])
        out[m] = idx[k] + g * len(sites)       # distinct label per gallery
    return out


def rmin_framework(pos, framework, box):
    tree, lo, L = kdtree_xyz(framework, box)
    d, _ = tree.query(wrap(pos.reshape(-1, 3), lo, L) - lo)
    return d.reshape(pos.shape[:-1])


def states_coordination(na, framework, box):
    r = rmin_framework(na, framework, box)
    return np.where(r < R_INNER, 0, np.where(r < R_OUTER, 1, 2)), r


def states_shell(water, na, box):
    """Boolean (nf, npairs) in-shell indicator for every (water, Na) pair that is
    ever within R_SHELL; returns indicator and pair list."""
    lo, L = box[:, 0], box[:, 1] - box[:, 0]
    nf = water.shape[0]
    per_frame, pairs = [], set()
    for f in range(nf):
        tw = cKDTree(wrap(water[f], lo, L) - lo, boxsize=L)
        hits = tw.query_ball_point(wrap(na[f], lo, L) - lo, R_SHELL)
        s = {(w, i) for i, ws in enumerate(hits) for w in ws}
        per_frame.append(s)
        pairs |= s
    pairs = sorted(pairs)
    index = {p: k for k, p in enumerate(pairs)}
    ind = np.zeros((nf, len(pairs)), dtype=np.int8)
    for f, s in enumerate(per_frame):
        if s:
            ind[f, [index[p] for p in s]] = 1
    return ind, np.array(pairs)


# ----------------------------------------------------------------------------
# survival
# ----------------------------------------------------------------------------
def run_end(states):
    """run_end[k, p] = first frame index > k where particle p changes state."""
    nf = states.shape[0]
    out = np.empty(states.shape, dtype=np.int64)
    out[-1] = nf
    for k in range(nf - 2, -1, -1):
        out[k] = np.where(states[k + 1] == states[k], out[k + 1], k + 1)
    return out


def survival(ts, states, target_state, lags_steps, dt_fs=2.0, max_pairs=3000,
             seed=0, any_state=False):
    """S(t) on the lag grid. `any_state=True`: persistence in whatever state the
    particle starts in (used for tau_site, where every Na+ is always in a basin)."""
    rng = np.random.default_rng(seed)
    rend = run_end(states)
    t_out, S_out, n_out = [], [], []
    for lag in lags_steps:
        lag = snap_to_grid(lag)
        lag_ns = lag * dt_fs * 1e-6
        vi, vj, actual = gather_pairs_split(ts, lag, lag_ns, max_pairs, rng, dt_fs=dt_fs)
        if vi is None:
            continue
        st0 = states[vi]
        start = np.ones_like(st0, bool) if any_state else (st0 == target_state)
        alive = start & (rend[vi] > vj[:, None])
        n0 = start.sum()
        if n0 == 0:
            continue
        t_out.append(actual * dt_fs * 1e-6)
        S_out.append(alive.sum() / n0)
        n_out.append(int(n0))
    t, S = np.array(t_out), np.array(S_out)
    o = np.argsort(t)
    return t[o], enforce_monotone(S[o]), np.array(n_out)[o]


def integrate_tau(t, S):
    """tau = int_0^T S dt with S(0)=1; NaN if S(T) > PLATEAU_THRESHOLD."""
    if len(S) == 0:
        return np.nan, "no data"
    if S[-1] > PLATEAU_THRESHOLD:
        flag = "plateau" if S[-1] > 0.5 else "lower bound"
        return np.nan, flag
    tt, SS = np.r_[0.0, t], np.r_[1.0, S]
    return float(np.sum(0.5 * (SS[1:] + SS[:-1]) * np.diff(tt))), "ok"


def lag_grid(ts, n=40):
    span = ts[-1] - ts[0]
    lo = max(np.min(np.diff(ts)), 1)
    return np.unique(np.round(np.logspace(np.log10(lo), np.log10(0.5 * span), n)).astype(np.int64))


# ----------------------------------------------------------------------------
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--na", required=True)
    p.add_argument("--water")
    p.add_argument("--framework", required=True)
    p.add_argument("--sites")
    p.add_argument("--d001", type=float, help="interlayer spacing (A); default Lz/2")
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--n-lags", type=int, default=40)
    p.add_argument("--max-pairs", type=int, default=3000)
    p.add_argument("--dt-fs", type=float, default=2.0)
    a = p.parse_args(argv)
    os.makedirs(a.output, exist_ok=True)

    ts_na, na, _, box = read_dump(a.na)
    framework = read_xyz_table(a.framework)
    d001 = a.d001 or (box[2, 1] - box[2, 0]) / 2
    lags = lag_grid(ts_na, a.n_lags)
    curves, taus = {}, {}

    if a.sites:
        st = states_site(na, read_xyz_table(a.sites), box, d001)
        curves["site"] = survival(ts_na, st, None, lags, a.dt_fs, a.max_pairs, any_state=True)
    st, rmin = states_coordination(na, framework, box)
    np.savetxt(os.path.join(a.output, "rmin_Na_Ofw.txt"), rmin.ravel(), fmt="%.3f",
               header="pooled r_min(Na-O_framework), A: histogram to check the 3.3/5.5 A minima")
    for k, name in enumerate(["inner", "outer", "mobile"]):
        curves[name] = survival(ts_na, st, k, lags, a.dt_fs, a.max_pairs)

    if a.water:
        ts_w, water, _, _ = read_dump(a.water)
        iw, ina, ts_c = common_frames(ts_w, ts_na)
        ind, _ = states_shell(water[iw], na[ina], box)
        curves["Na-shell"] = survival(ts_c, ind, 1, lag_grid(ts_c, a.n_lags), a.dt_fs, a.max_pairs)
        surf = (rmin_framework(water, framework, box) < R_SURF).astype(np.int8)
        curves["surf"] = survival(ts_w, surf, 1, lag_grid(ts_w, a.n_lags), a.dt_fs, a.max_pairs)

    for name, (t, S, n0) in curves.items():
        np.savetxt(os.path.join(a.output, f"S_{name}.csv"), np.c_[t, S, n0], delimiter=",",
                   header="t_ns,S,n_start", comments="")
        tau, flag = integrate_tau(t, S)
        taus[name] = dict(tau_ns=tau, flag=flag, S_end=float(S[-1]) if len(S) else None)
    with open(os.path.join(a.output, "tau.json"), "w") as f:
        json.dump(dict(d001=d001, cutoffs=dict(inner=R_INNER, outer=R_OUTER, shell=R_SHELL,
                                                surf=R_SURF), tau=taus), f, indent=2)
    print(json.dumps(taus, indent=2))


if __name__ == "__main__":
    main()

"""
sampling.py -- time-origin sampling for non-uniformly dumped LAMMPS trajectories.

The production dumps use a five-phase schedule (dense early windows, then one
frame every 20,000 steps = 40 ps). Naive multi-origin averaging over such a
trajectory over-represents the dense early window (~0.7 % of a 300 ns run) and
produces artifacts. Three corrections are applied (SI, residence-time section):

  1. time-density-weighted origin sampling   -> gather_pairs_for_lag_weighted
  2. exclusion of lags near phase boundaries -> gather_pairs_split
  3. non-increasing survival curves          -> enforce_monotone (cumulative minimum)

Provenance: `gather_pairs_for_lag_weighted` and `gather_pairs_split` are
recovered from the August 2026 session (file residence_setup.py); the
`dt_fs` argument is accepted for call compatibility and is unused.
`enforce_monotone` is a re-implementation of the cumulative-minimum filter
described in the SI.
"""
import numpy as np

# Phase-transition times of the 5-phase dump schedule (ns), 2 fs timestep
PHASE_BOUNDARIES_NS = [0.002, 0.022, 0.222, 2.222]
BOUNDARY_EXCLUSION_FRAC = 0.35  # skip target lags within this fractional log-distance of a boundary


def gather_pairs_for_lag_weighted(timesteps, target_lag, max_pairs, rng, tol=0.15, dt_fs=None):
    """Like gather_pairs_for_lag, but samples origins with probability proportional
    to their local time-density weight (average time-gap to neighboring frames),
    correcting for the 5-phase dump schedule's massive over-representation of the
    densely-sampled early window (phases 1-4, first ~2.2ns) relative to its true
    (~0.7%) share of the full trajectory duration."""
    n = len(timesteps)
    target_vals = timesteps + target_lag
    j_candidates = np.searchsorted(timesteps, target_vals)
    valid_i, valid_j = [], []
    for i in range(n):
        j = j_candidates[i]
        best_j, best_err = -1, None
        for jj in (j-1, j):
            if 0 <= jj < n:
                err = abs((timesteps[jj]-timesteps[i]) - target_lag)
                if best_err is None or err < best_err:
                    best_err, best_j = err, jj
        if best_j > i and best_err <= tol*target_lag:
            valid_i.append(i); valid_j.append(best_j)
    if len(valid_i) == 0:
        return None, None, 0
    valid_i = np.array(valid_i); valid_j = np.array(valid_j)
    actual_lag = np.median(timesteps[valid_j]-timesteps[valid_i])

    # local time-density weight per origin: gap to neighboring frames (steps), so
    # sparse (phase-5) origins get high weight, dense-window origins get low weight
    gaps = np.empty(n)
    gaps[1:-1] = (timesteps[2:]-timesteps[:-2])/2.0
    gaps[0] = timesteps[1]-timesteps[0]
    gaps[-1] = timesteps[-1]-timesteps[-2]
    w = gaps[valid_i]
    w = w/w.sum()

    n_pairs = len(valid_i)
    if max_pairs > 0 and n_pairs > max_pairs:
        sel = rng.choice(n_pairs, size=max_pairs, replace=True, p=w)
        sel = np.unique(sel)
        valid_i = valid_i[sel]; valid_j = valid_j[sel]
    return valid_i, valid_j, actual_lag


def gather_pairs_split(ts, target_lag_steps, target_lag_ns, max_pairs, rng, dt_fs=2.0):
    """Wraps gather_pairs_for_lag_weighted (origin-density-corrected sampling), and
    additionally skips target lags that fall too close to one of the 5-phase dump
    schedule's internal transition times, since the origin pool available right at a
    boundary is small and non-representative regardless of weighting (an inherent
    discreteness effect of the fixed dump schedule, not fixable by resampling)."""
    for b in PHASE_BOUNDARIES_NS:
        if abs(np.log10(target_lag_ns) - np.log10(b)) < BOUNDARY_EXCLUSION_FRAC * abs(np.log10(2.0)):
            return None, None, 0
    return gather_pairs_for_lag_weighted(ts, target_lag_steps, max_pairs, rng, dt_fs=dt_fs)


def snap_to_grid(target_lag_steps, grid_steps=20000):
    """Phase-5 grid snapping: lags beyond the dense window are rounded to an
    integer multiple of the phase-5 dump interval, so that the matched lag is
    exactly realisable rather than off-grid."""
    if target_lag_steps < grid_steps:
        return int(target_lag_steps)
    return int(round(target_lag_steps / grid_steps) * grid_steps)


def enforce_monotone(S):
    """Cumulative-minimum (isotonic) filter: a survival probability is
    non-increasing in t. NaNs are left in place."""
    S = np.asarray(S, dtype=float).copy()
    ok = np.isfinite(S)
    S[ok] = np.minimum.accumulate(S[ok])
    return S

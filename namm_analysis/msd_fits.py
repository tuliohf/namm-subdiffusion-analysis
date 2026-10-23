#!/usr/bin/env python3
"""
msd_fits.py -- two-crossover fit of the in-plane MSD (ballistic -> generalized
diffusive -> Fickian), Fickian diffusion coefficient D_F, per-replica spread and
convergence with trajectory length.

    MSD_par(t) = 2 D_alpha t_b^alpha (t/t_b)^2 / (1 + t/t_b)^(2-alpha) * (1 + t/t_f)^(1-alpha)
    D_F        = D_alpha t_f^(alpha-1)            (t_b cancels)

MSD_par is the per-component in-plane MSD, (MSD_x + MSD_y)/2 [A^2], t in ns.

Procedure (SI, "Fitting considerations"):
  1. t_b from a single-crossover fit (ballistic -> generalized diffusive);
  2. two-crossover fit with t_b fixed, D_alpha, alpha, t_f free. Fitting all four
     parameters converges to a degenerate solution (alpha at its lower bound,
     t_f < 0.02 ns), hence the fixed t_b;
  3. fits on log10 MSD vs log10 t, resampled uniformly in log t (4000 points);
     bounds alpha in [0.02, 1.6], log10 t_f in [-3, 4];
  4. t_f at the upper bound (>1e4 ns): no Fickian crossover resolved (reported as
     > 1e4 ns); D_F not reported when t_f >= 100 ns (poorly constrained,
     far beyond the ~300 ns window).

Input: LAMMPS fix print files of compute msd / compute msd/nongauss, columns
    step | water x y z | Na x y z | water total r4 NGP3D | Na total r4 NGP3D
(2 fs timestep; duplicate rows at the dump-phase transitions are dropped).
Several files = replicas of the same state, averaged before the fit
(--per-replica also fits each one separately for the spread of D_F).

Provenance: `single_crossover`, `two_crossover`, `fit_single_for_tb`, `fit_two`
and `compute_DF` are recovered verbatim from the August 2026 analysis session
(fit_all27_and_DF.py; the D_F threshold is the 100 ns value adopted in that
session). The file reader, replica averaging, convergence loop and command line
were written in October 2026 around them.

Example
  python3 msd_fits.py -i MSD_16_A.txt MSD_16_B.txt MSD_16_C.txt --species water \
      --per-replica --convergence -o fit_16_water
"""
import argparse, json, os
import numpy as np
from scipy.optimize import curve_fit

DT_FS = 2.0
COLS = {"water": (1, 2), "na": (4, 5)}      # x and y columns of each species
TF_UNRESOLVED_NS = 100.0


# ---------------------------------------------------------------------------
# recovered verbatim (August 2026)
# ---------------------------------------------------------------------------
def single_crossover(t, D_alpha, alpha, t_c):
    x = t/t_c
    return 2*D_alpha*t_c**alpha * x**2 / (1+x)**(2-alpha)

def two_crossover(t, D_alpha, alpha, t_f, t_b):
    x = t/t_b
    return 2*D_alpha*t_b**alpha * x**2/(1+x)**(2-alpha) * (1+t/t_f)**(1-alpha)

def fit_single_for_tb(t_full, y_full):
    valid = (~np.isnan(y_full)) & (y_full>0) & (t_full>0)
    tt, yy = t_full[valid], y_full[valid]
    if len(tt) < 100: return 3e-5
    logt, logy = np.log10(tt), np.log10(yy)
    logt_u = np.linspace(logt.min(), logt.max(), 4000)
    logy_u = np.interp(logt_u, logt, logy)
    t_u = 10**logt_u
    ntail = min(500, len(logy_u)-1)
    slope_long = (logy_u[-1]-logy_u[-ntail])/(logt_u[-1]-logt_u[-ntail]+1e-12)
    alpha0 = np.clip(slope_long, 0.05, 1.5)
    D0 = (10**logy_u[-1])/(2*(10**logt_u[-1])**alpha0)
    def logmodel(logt_, D_alpha, alpha, logtc):
        t_c = 10**logtc
        return np.log10(np.clip(single_crossover(10**logt_, D_alpha, alpha, t_c),1e-14,None))
    try:
        popt,_ = curve_fit(logmodel, logt_u, logy_u, p0=[D0,alpha0,np.log10(3e-5)],
                            bounds=([1e-8,0.02,-5],[1e6,1.6,2]), maxfev=50000)
        return 10**popt[2]
    except Exception:
        return 3e-5

def fit_two(t_full, y_full, t_b_fixed):
    valid = (~np.isnan(y_full)) & (y_full>0) & (t_full>0)
    tt, yy = t_full[valid], y_full[valid]
    if len(tt) < 100: return dict(success=False)
    logt, logy = np.log10(tt), np.log10(yy)
    logt_u = np.linspace(logt.min(), logt.max(), 4000)
    logy_u = np.interp(logt_u, logt, logy)
    ntail = min(500, len(logy_u)-1)
    slope_long = (logy_u[-1]-logy_u[-ntail])/(logt_u[-1]-logt_u[-ntail]+1e-12)
    alpha0 = np.clip(slope_long, 0.05, 1.5)
    D0 = (10**logy_u[-1])/(2*(10**logt_u[-1])**alpha0)
    def logmodel(logt_, D_alpha, alpha, log_tf):
        return np.log10(np.clip(two_crossover(10**logt_, D_alpha, alpha, 10**log_tf, t_b_fixed),1e-14,None))
    p0 = [D0, alpha0, np.log10(50)]
    try:
        popt, pcov = curve_fit(logmodel, logt_u, logy_u, p0=p0,
                                bounds=([1e-8,0.02,-3],[1e6,1.6,4]), maxfev=100000)
        D_alpha, alpha, log_tf = popt
        t_f = 10**log_tf
        pred = logmodel(logt_u, *popt)
        r2 = 1 - np.sum((logy_u-pred)**2)/np.sum((logy_u-np.mean(logy_u))**2)
        return dict(D_alpha=D_alpha, alpha=alpha, t_f=t_f, t_b=t_b_fixed, r2=r2, success=True)
    except Exception as e:
        return dict(success=False, error=str(e))

def compute_DF(res, tf_unresolved_threshold=TF_UNRESOLVED_NS):
    """D_F = D_alpha * t_f^(alpha-1). No t_b. NaN if t_f unresolved/pinned to bound."""
    if not res.get('success'):
        return np.nan
    if res['t_f'] >= tf_unresolved_threshold:
        return np.nan
    return res['D_alpha'] * res['t_f']**(res['alpha']-1)


# ---------------------------------------------------------------------------
# added October 2026
# ---------------------------------------------------------------------------
def read_msd(fpath, species):
    """Return t [ns], MSD_par [A^2] (per component) from a fix print file."""
    a = np.loadtxt(fpath, comments="#", ndmin=2)
    _, first = np.unique(a[:, 0], return_index=True)       # drop duplicate timesteps
    a = a[np.sort(first)]
    a = a[np.argsort(a[:, 0], kind="stable")]
    cx, cy = COLS[species]
    t = (a[:, 0] - a[0, 0]) * DT_FS * 1e-6
    return t, 0.5 * (a[:, cx] + a[:, cy])


def average_replicas(curves):
    """Mean MSD_par on the grid of the shortest replica (log-t interpolation)."""
    t_ref = min((t for t, _ in curves), key=lambda t: t[-1])
    t_ref = t_ref[t_ref > 0]
    ys = [10 ** np.interp(np.log10(t_ref), np.log10(t[t > 0]), np.log10(np.clip(y[t > 0], 1e-14, None)))
          for t, y in curves]
    return t_ref, np.mean(ys, axis=0)


def fit_state(t, y):
    t_b = fit_single_for_tb(t, y)
    res = fit_two(t, y, t_b)
    res["D_F"] = compute_DF(res)
    res["t_f_at_bound"] = bool(res.get("success") and res["t_f"] > 0.99e4)
    return res


def convergence(t, y, t_b, n=18):
    """Refit (t_b fixed) on trajectories truncated at log-spaced lengths."""
    rows = []
    for tmax in np.geomspace(1.0, t[-1], n):
        m = t <= tmax
        r = fit_two(t[m], y[m], t_b)
        if r.get("success"):
            rows.append([tmax, r["D_alpha"], r["alpha"], r["t_f"], compute_DF(r)])
    return np.array(rows)


def clean(d):
    return {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in d.items()}


def main(argv=None):
    p = argparse.ArgumentParser(description="Two-crossover fit of MSD_par")
    p.add_argument("-i", "--input", nargs="+", required=True, help="replicas of one state")
    p.add_argument("--species", choices=["water", "na"], required=True)
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--per-replica", action="store_true")
    p.add_argument("--convergence", action="store_true")
    a = p.parse_args(argv)
    os.makedirs(a.output, exist_ok=True)

    curves = [read_msd(f, a.species) for f in a.input]
    t, y = average_replicas(curves)
    res = fit_state(t, y)
    out = dict(species=a.species, files=a.input, replica_average=clean(res))
    if res.get("success"):
        tf = ">1e4 (unresolved)" if res["t_f_at_bound"] else f"{res['t_f']:.4g}"
        print(f"alpha={res['alpha']:.3f}  t_b={res['t_b']*1e3:.4f} ps  t_f={tf} ns  "
              f"D_alpha={res['D_alpha']:.4g}  D_F={res['D_F']:.4g} A^2/ns  R2={res['r2']:.4f}")
        model = two_crossover(t, res["D_alpha"], res["alpha"], res["t_f"], res["t_b"])
        np.savetxt(os.path.join(a.output, "msd_par_fit.csv"), np.c_[t, y, model], delimiter=",",
                   header="t_ns,msd_par_mean,msd_par_fit", comments="")

        if a.per_replica:
            reps = []
            for f, (tr, yr) in zip(a.input, curves):
                r = fit_two(tr, yr, res["t_b"])          # t_b of the replica average
                r["D_F"] = compute_DF(r)
                reps.append(dict(file=f, **clean(r)))
            DF = np.array([r["D_F"] for r in reps if r.get("success")], float)
            DF = DF[np.isfinite(DF)]
            out["per_replica"] = reps
            if len(DF) > 1:
                out["D_F_mean"] = float(DF.mean())
                out["D_F_sem"] = float(DF.std(ddof=1) / np.sqrt(len(DF)))
        if a.convergence:
            c = convergence(t, y, res["t_b"])
            np.savetxt(os.path.join(a.output, "convergence.csv"), c, delimiter=",",
                       header="t_max_ns,D_alpha,alpha,t_f_ns,D_F", comments="")
    with open(os.path.join(a.output, "fit.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)


if __name__ == "__main__":
    main()

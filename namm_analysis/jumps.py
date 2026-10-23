#!/usr/bin/env python3
"""
jumps.py -- jump detection, waiting-time statistics and displacement tails.

Jump detection (SI "Waiting-time and displacement distributions"): a particle
stays in the same sojourn while its in-plane position remains within R of the
running mean position (cage centre) accumulated since the start of the sojourn;
the first frame exceeding R is a jump and starts a new sojourn.
R = 2.6 A for Na+ (half the distance between adjacent ditrigonal cavities),
R = 2.0 A for water. Uniformly sampled frames only (phase 5, 40 ps).

Outputs
  Psi(tau)          Kaplan-Meier survival of the waiting times; first sojourn
                    of each trajectory discarded (unknown start), last one
                    right-censored
  beta_loc(tau)     -dln Psi / dln tau
  beta_w            mean beta_loc over t_f/30 <= tau <= t_f   (needs --tf)
  tau_star          first tau with beta_loc >= 1
  <tau>             mean of the complete waiting times
  <cos theta>       successive jump vectors (cage centre to cage centre)
  D_RW              <l^2>/(4 <tau>)  (uncorrelated random walk, 2D)
  jumps per particle per 100 ns
  van Hove tails    G_s(dx, t)/sigma(t) along x (and y, pooled), excess kurtosis,
                    and an apparent power-law exponent fitted to the tail

Provenance: rebuilt in October 2026 from the SI text of the revision session
(the session scripts were not printed in full and are not recoverable verbatim).

Example
  python3 jumps.py -i dump.Iw_unwrap_16_*.lammpstrj -o jumps_Na_16 --R 2.6 --tf 1.7
"""
import argparse, json, os
import numpy as np

from io_lammps import read_dump, uniform_tail


def detect(xy, R):
    """xy: (N, 2) one particle. Returns list of (start_index, centre) sojourns."""
    soj, start, s = [], 0, xy[0].copy()
    n = 1
    for k in range(1, len(xy)):
        c = s / n
        if np.hypot(*(xy[k] - c)) > R:
            soj.append((start, c))
            start, s, n = k, xy[k].copy(), 1
        else:
            s += xy[k]; n += 1
    soj.append((start, s / n))
    return soj


def kaplan_meier(t, observed):
    """Psi(t) at the distinct event times. observed=False -> right-censored."""
    t, observed = np.asarray(t, float), np.asarray(observed, bool)
    times = np.unique(t[observed])
    S, surv = 1.0, []
    for u in times:
        at_risk = np.sum(t >= u)
        d = np.sum((t == u) & observed)
        S *= 1 - d / at_risk
        surv.append(S)
    return times, np.array(surv)


def local_slope(tau, Psi, smooth=5):
    m = (Psi > 0) & (tau > 0)
    lt, lp = np.log(tau[m]), np.log(Psi[m])
    grid = np.linspace(lt[0], lt[-1], max(len(lt) // 2, 10))
    lpg = np.interp(grid, lt, lp)
    if smooth > 1:
        lpg = np.convolve(np.r_[[lpg[0]] * (smooth // 2), lpg, [lpg[-1]] * (smooth // 2)],
                          np.ones(smooth) / smooth, mode="valid")
    return np.exp(grid), -np.gradient(lpg, grid)


def analyze(trajs, dt_ns, R, tf=None):
    waits, observed, l2, cosines, njumps, T_each = [], [], [], [], [], []
    for xy_t in trajs:                                   # one array (N, P, 2) per trajectory
        N, P, _ = xy_t.shape
        T = (N - 1) * dt_ns
        for p in range(P):
            soj = detect(xy_t[:, p], R)
            njumps.append((len(soj) - 1) / T * 100.0)
            T_each.append(T)
            starts = [s for s, _ in soj]
            for k in range(1, len(soj)):                 # first sojourn: start unknown
                end = starts[k + 1] if k + 1 < len(soj) else N - 1
                waits.append((end - starts[k]) * dt_ns)
                observed.append(k + 1 < len(soj))
            jv = np.diff(np.array([c for _, c in soj]), axis=0)
            l2.extend(np.sum(jv ** 2, axis=1))
            if len(jv) > 1:
                a, b = jv[:-1], jv[1:]
                cosines.extend(np.sum(a * b, 1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)))
    waits, observed = np.array(waits), np.array(observed)
    out = dict(R=R, dt_ns=dt_ns, n_sojourns=int(len(waits)), n_complete=int(observed.sum()),
               jumps_per_100ns_median=float(np.median(njumps)),
               n_without_jump=int(np.sum(np.array(njumps) == 0)),
               jumps_per_100ns=np.array(njumps))
    if observed.sum() == 0:
        return out
    tau, Psi = kaplan_meier(waits, observed)
    mean_tau = float(waits[observed].mean())
    ts, beta = local_slope(tau, Psi)
    above = np.where(beta >= 1)[0]
    out.update(tau=tau, Psi=Psi, beta_tau=ts, beta_loc=beta, mean_tau_ns=mean_tau,
               tau_star_ns=float(ts[above[0]]) if len(above) else np.nan,
               mean_cos=float(np.mean(cosines)) if cosines else np.nan,
               D_RW=float(np.mean(l2) / (4 * mean_tau)) if l2 else np.nan)
    if tf:
        m = (ts >= tf / 30) & (ts <= tf)
        out["beta_w"] = float(np.mean(beta[m])) if m.any() else np.nan
    return out


def displacement_tails(trajs, dt_ns, lags_ns, n_bins=80, tail_from=3.0):
    """G_s(dx,t) of one Cartesian in-plane component (x and y pooled), rescaled
    by sigma(t); excess kurtosis; apparent exponent 1+mu of |dx|^-(1+mu) fitted
    on |dx|/sigma > tail_from."""
    res = []
    for lag in lags_ns:
        k = int(round(lag / dt_ns))
        parts = [(x[k:] - x[:-k]).reshape(-1) for x in trajs if 0 < k < x.shape[0]]
        if not parts or sum(p.size for p in parts) < 100:
            continue
        d = np.concatenate(parts)
        sig = d.std()
        u = d / sig
        exk = float(np.mean(u ** 4) - 3)
        h, e = np.histogram(u, bins=n_bins, range=(-12, 12), density=True)
        c = 0.5 * (e[1:] + e[:-1])
        tail = (np.abs(c) > tail_from) & (h > 0)
        slope = np.nan
        if tail.sum() > 4:
            slope = -np.polyfit(np.log(np.abs(c[tail])), np.log(h[tail]), 1)[0]
        res.append(dict(lag_ns=k * dt_ns, sigma=float(sig), excess_kurtosis=exk,
                        apparent_1_plus_mu=float(slope), u=c.tolist(), Gs=h.tolist()))
    return res


def main(argv=None):
    p = argparse.ArgumentParser(description="Jump detection, waiting times, van Hove tails")
    p.add_argument("-i", "--input", nargs="+", required=True)
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--R", type=float, required=True, help="2.6 for Na+, 2.0 for water (A)")
    p.add_argument("--tf", type=float, help="Fickian crossover time t_f (ns) for beta_w")
    p.add_argument("--dt-fs", type=float, default=2.0)
    p.add_argument("--step", type=int, default=None)
    p.add_argument("--tail-lags", type=float, nargs="+", default=[0.04, 0.2, 1.0, 5.0, 20.0])
    a = p.parse_args(argv)
    os.makedirs(a.output, exist_ok=True)

    trajs, dt_ns = [], None
    for f in a.input:
        ts, c, _, _ = read_dump(f)
        idx, step = uniform_tail(ts, a.step)
        dt_ns = step * a.dt_fs * 1e-6
        trajs.append(c[idx, :, :2])

    res = analyze(trajs, dt_ns, a.R, a.tf)
    if "tau" in res:
        np.savetxt(os.path.join(a.output, "Psi.csv"), np.c_[res["tau"], res["Psi"]],
                   delimiter=",", header="tau_ns,Psi", comments="")
        np.savetxt(os.path.join(a.output, "beta_loc.csv"), np.c_[res["beta_tau"], res["beta_loc"]],
                   delimiter=",", header="tau_ns,beta_loc", comments="")
    np.savetxt(os.path.join(a.output, "jumps_per_100ns.txt"), res["jumps_per_100ns"], fmt="%.3f")
    summary = {k: v for k, v in res.items() if not isinstance(v, np.ndarray)}
    with open(os.path.join(a.output, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(a.output, "vanhove_tails.json"), "w") as f:
        json.dump(displacement_tails(trajs, dt_ns, a.tail_lags), f)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

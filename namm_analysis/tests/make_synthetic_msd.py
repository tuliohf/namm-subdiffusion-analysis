"""Synthetic compute-msd files (three replicas) following the two-crossover model,
in the column layout of the production runs. Usage:
    python3 make_synthetic_msd.py <outdir>
Water-like 1W parameters: D_alpha = 10, alpha = 0.58, t_f = 0.297 ns, t_b = 0.021 ps
(true D_F = 16.65 A^2/ns).
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from msd_fits import two_crossover

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
rng = np.random.default_rng(0)
steps = np.r_[np.arange(0, 1000, 1), np.arange(1000, 11000, 10), np.arange(11000, 111000, 100),
              np.arange(111000, 1111001, 1000), np.arange(1111000, 150_000_001, 20000)]
t = steps * 2e-6
for r in "ABC":
    m = two_crossover(np.clip(t, 1e-9, None), 10.0, 0.58, 0.297, 2.1e-5) * np.exp(rng.normal(0, 0.02, len(t)))
    m[0] = 0
    a = np.zeros((len(t), 15))
    a[:, 0] = steps
    a[:, 1] = a[:, 2] = m           # water x, y (per component)
    a[:, 4] = a[:, 5] = 0.3 * m     # Na x, y
    np.savetxt(os.path.join(out, f"MSD_test_{r}.txt"), a, fmt="%.6g",
               header="step wx wy wz nax nay naz w_tot w_r4 w_ngp na_tot na_r4 na_ngp pad pad")

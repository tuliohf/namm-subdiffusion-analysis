"""Synthetic test dumps: Brownian particles in two slit galleries, written with a
five-phase dump schedule like the production runs. Usage:
    python3 make_synthetic.py <outdir>
"""
import os, sys
import numpy as np

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
rng = np.random.default_rng(1)
Lx = Ly = 20.0
d001 = 12.5
Lz = 2 * d001
# 5-phase schedule (steps, dt = 2 fs): boundaries at 0.002, 0.022, 0.222, 2.222 ns
ts = np.r_[np.arange(0, 1000, 10), np.arange(1000, 11000, 100), np.arange(11000, 111000, 1000),
           np.arange(111000, 1111000, 10000), np.arange(1111000, 10_000_001, 20000)]
D = 1e-4  # A^2 per step per component (= 0.05 A^2/ps)


def walk(n, D, zc):
    pos = np.empty((len(ts), n, 3))
    x = np.c_[rng.uniform(0, Lx, n), rng.uniform(0, Ly, n), zc + rng.normal(0, 0.3, n)]
    last = 0
    for k, t in enumerate(ts):
        dt = t - last
        x[:, :2] += rng.normal(0, np.sqrt(2 * D * dt), (n, 2))
        x[:, 2] = zc + rng.normal(0, 0.3, n)
        pos[k] = x
        last = t
    return pos


def write(path, pos):
    with open(path, "w") as f:
        for k, t in enumerate(ts):
            f.write(f"ITEM: TIMESTEP\n{t}\nITEM: NUMBER OF ATOMS\n{pos.shape[1]}\n")
            f.write(f"ITEM: BOX BOUNDS pp pp pp\n0 {Lx}\n0 {Ly}\n0 {Lz}\n")
            f.write("ITEM: ATOMS id xu yu zu\n")
            for i, p in enumerate(pos[k]):
                f.write(f"{i+1} {p[0]:.5g} {p[1]:.5g} {p[2]:.5g}\n")


zc = np.repeat([d001 / 2, 1.5 * d001], [6, 6])
na = np.concatenate([walk(1, D, z) for z in zc], axis=1)
w = np.concatenate([walk(20, 5 * D, z) for z in (d001 / 2, 1.5 * d001)], axis=1)
write(os.path.join(out, "dump.Iw_unwrap_test_1.lammpstrj"), na)
write(os.path.join(out, "dump.Ow_unwrap_test_1.lammpstrj"), w)
# framework O: square grids 1.4 A below/above each gallery centre +- 3 A
g = np.arange(0, Lx, 2.5)
X, Y = np.meshgrid(g, g)
planes = [d001 / 2 - 3.0, d001 / 2 + 3.0, 1.5 * d001 - 3.0, 1.5 * d001 + 3.0]
fw = np.vstack([np.c_[X.ravel(), Y.ravel(), np.full(X.size, z)] for z in planes])
np.savetxt(os.path.join(out, "Ofw.xyz"), fw, fmt="%.3f")
# substitution sites: octahedral, mid-layer (z = 0 and z = d001)
sites = np.c_[rng.uniform(0, Lx, 6), rng.uniform(0, Ly, 6), np.repeat([0.0, d001], 3)]
np.savetxt(os.path.join(out, "subs.xyz"), sites, fmt="%.3f")
print("frames", len(ts))

"""
io_lammps.py -- readers shared by residence.py, tamsd.py and jumps.py.

Dump format assumed (as in the production runs):
    dump ... custom N file id xu yu zu
i.e. unwrapped coordinates, constant atom count, one dump per species
(dump.Iw_unwrap_<c0>_<rep>.lammpstrj for Na+, dump.Ow_unwrap_<c0>_<rep>.lammpstrj
for water oxygens).
"""
import numpy as np


def read_dump(fpath):
    """Return (timesteps[nf], coords[nf, natoms, 3], ids[natoms], box[3x2]).

    Atoms are sorted by id in every frame; frames are sorted by timestep and
    duplicate timesteps (written at phase transitions) are dropped."""
    with open(fpath) as f:
        lines = f.readlines()
    i, n = 0, len(lines)
    ts_list, frames, box, ids = [], [], None, None
    while i < n:
        if not lines[i].startswith("ITEM: TIMESTEP"):
            i += 1
            continue
        ts = int(lines[i + 1])
        na = int(lines[i + 3])
        if box is None:
            box = np.array([[float(v) for v in lines[i + 5 + k].split()[:2]] for k in range(3)])
        cols = lines[i + 8].split()[2:]
        arr = np.array(" ".join(lines[i + 9:i + 9 + na]).split(), dtype=float).reshape(na, len(cols))
        c = {name: k for k, name in enumerate(cols)}
        arr = arr[np.argsort(arr[:, c["id"]])]
        if ids is None:
            ids = arr[:, c["id"]].astype(int)
        elif len(arr) != len(ids):
            raise ValueError(f"atom count changed at timestep {ts}")
        ts_list.append(ts)
        frames.append(arr[:, [c["xu"], c["yu"], c["zu"]]])
        i += 9 + na
    ts = np.array(ts_list, dtype=np.int64)
    coords = np.stack(frames)
    order = np.argsort(ts, kind="stable")
    ts, coords = ts[order], coords[order]
    keep = np.r_[True, np.diff(ts) > 0]
    return ts[keep], coords[keep], ids, box


def common_frames(ts_a, ts_b):
    """Indices selecting the timesteps present in both dumps."""
    common = np.intersect1d(ts_a, ts_b)
    return np.searchsorted(ts_a, common), np.searchsorted(ts_b, common), common


def uniform_tail(ts, step=None):
    """Indices of the uniformly sampled part of a multi-phase dump (phase 5).

    If `step` is None, the most frequent frame spacing is used. Returns the
    longest run of frames spaced exactly by `step`."""
    d = np.diff(ts)
    if step is None:
        vals, counts = np.unique(d, return_counts=True)
        step = vals[np.argmax(counts)]
    good = d == step
    best_len, best_start, run_start = 0, 0, None
    for k, g in enumerate(np.r_[good, False]):
        if g and run_start is None:
            run_start = k
        elif not g and run_start is not None:
            if k - run_start > best_len:
                best_len, best_start = k - run_start, run_start
            run_start = None
    return np.arange(best_start, best_start + best_len + 1), int(step)


def read_xyz_table(fpath):
    """Plain whitespace table with x y z in the last three columns (comments with #).
    Used for the frozen framework oxygens and the substitution sites."""
    a = np.loadtxt(fpath, comments="#", ndmin=2)
    return a[:, -3:]


def wrap(x, lo, L):
    return lo + np.mod(x - lo, L)

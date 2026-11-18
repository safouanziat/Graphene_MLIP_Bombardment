"""Extra DFT candidate frames from the stored MD trajectories (no new MD): frames at given delays after the closest approach.

    python -W ignore extra_frames.py | tee extra_frames.txt

Sputtered / isolated C atoms are REMOVED (they are free magnetic atoms in DFT and break the SCF), the N is parked if > --park A from the sheet.
Frames that break the recipe distance rules (C-N >= 1.1, C-C >= 1.0 A), have the N further than --max-nc from every C, or already exist in
the earlier DFT sets are skipped.
Output: <dst>/pending.db, kind = N_<site>_E<e>_r<run>_t<fs>fs_x

Frame index -> time is calibrated against two stored frames (an early one where the N moves fast, and a late one); the script stops if
they disagree.
"""
import argparse
import hashlib
import os
from pathlib import Path

import recipe as R   # first: gpaw before ase.parallel

import numpy as np
from ase.db import connect
from ase.geometry import find_mic
from ase.io import read
from ase.neighborlist import neighbor_list

p = argparse.ArgumentParser()
p.add_argument("--src", default="data/nimpact_final_n7/")
p.add_argument("--dst", default="data/nimpact_more/")
p.add_argument("--top-energies", type=int, nargs="*", default=[20, 30, 50])
p.add_argument("--bridge-energies", type=int, nargs="*", default=[30, 50])
p.add_argument("--offsets", type=int, nargs="*", default=[10, 30])
p.add_argument("--max", type=int, default=30)
p.add_argument("--n", type=int, default=4)
p.add_argument("--a-dft", type=float, default=2.4704)
p.add_argument("--park", type=float, default=7.0)
p.add_argument("--iso", type=float, default=2.3)
p.add_argument("--max-nc", type=float, default=3.0)
p.add_argument("--dump-fs", type=float, default=5.0)
a = p.parse_args()

src = Path(a.src)
trajs = sorted((src / "traj").glob("N_*.extxyz"))


def convert(fr):
    at = fr.copy()
    at.calc = None
    sc = a.a_dft / (np.linalg.norm(at.cell[0]) / a.n)
    cell = np.array(at.cell)
    at.positions[:, :2] *= sc
    cell[0] *= sc
    cell[1] *= sc
    at.set_cell(cell, scale_atoms=False)
    return at


def approach(fr):
    n = len(fr) - 1
    return fr.get_distances(n, range(n), mic=True).min()


def shape_dev(ref, at):
    """rms difference of the N and the first 20 sheet atoms after removing any overall translation."""
    ids = list(range(20)) + [len(at) - 1]
    jds = list(range(20)) + [len(ref) - 1]
    v = at.positions[ids] - ref.positions[jds]
    v, _ = find_mic(v, ref.cell, pbc=True)
    return float(np.sqrt(((v - v.mean(axis=0)) ** 2).sum(axis=1).mean()))


# 1. calibrate frame index -> time with two stored frames
tr = read(src / "traj" / "N_top_E20_r0.extxyz", index=":")
tr_c = [convert(fr) for fr in tr]
refs = [("data/nimpact_final_n7/pending.db", "N_top_E20_r0_t25fs", 25),
        ("data/nimpact_final_n7_rm/labelled.db", "N_top_E20_r0_t125fs_rm", 125)]
shifts = []
for db, kind, t in refs:
    ref = next(r for r in connect(db).select() if r.kind == kind).toatoms()
    dev = [shape_dev(ref, at) for at in tr_c]
    best = int(np.argmin(dev))
    shifts.append(best - int(round(t / a.dump_fs)))
    print(f"calibration {kind}: best trajectory index {best}, deviation {dev[best]:.3f} A (expected index {int(round(t / a.dump_fs))})")
if shifts[0] != shifts[1] or max(0.0, 0.0) > 0:
    raise SystemExit(f"calibration failed: the two reference frames give different shifts {shifts}")
shift = shifts[0]
print(f"time = {a.dump_fs:g} fs x (index - {shift})")

# 2. what we already have
have = set()
for db in ["data/nimpact_final_n7/pending.db", "data/nimpact_final_n7_rm/pending.db", "data/nimpact_final_n7_iso/pending.db"]:
    if os.path.exists(db):
        for r in connect(db).select():
            have.add("_".join(r.kind.split("_")[1:5]))

# 3. candidates
cands = []
for f in trajs:
    _, site, e, run = f.stem.split("_")
    E = int(e[1:])
    if (site == "top" and E not in a.top_energies) or (site == "bridge" and E not in a.bridge_energies) or site == "hollow":
        continue
    frames = read(f, index=":")
    dm = [approach(fr) for fr in frames]
    ic = int(np.argmin(dm))
    for off in a.offsets:
        i = ic + int(round(off / a.dump_fs))
        if i >= len(frames):
            continue
        t = int(round(a.dump_fs * (i - shift)))
        key = f"{site}_E{E}_{run}_t{t}fs"
        if key in have:
            continue
        at = convert(frames[i])
        nN = len(at) - 1
        zpl = float(np.median(at.positions[:nN, 2]))
        zn = at.positions[-1, 2] - zpl
        if abs(zn) > a.park:
            at.positions[-1, 2] = zpl + np.sign(zn) * a.park
        drop = []
        for k in range(nN):
            d = at.get_distances(k, [j for j in range(len(at)) if j != k], mic=True)
            if d.min() > a.iso or abs(at.positions[k, 2] - zpl) > a.park:
                drop.append(k)
        at = at[[k for k in range(len(at)) if k not in drop]]
        nc = len(at) - 1
        at.wrap()
        at = R.normalise(at)
        at.set_initial_magnetic_moments([0.3] * nc + [3.0])
        d_n = at.get_distances(nc, range(nc), mic=True).min()
        if d_n > a.max_nc:
            print(f"  skip {key}: N too far from the sheet (N-C {d_n:.2f})")
            continue
        ii, jj, dd = neighbor_list("ijd", at, 2.0)
        m = (ii < nc) & (jj < nc)
        d_cc = dd[m].min() if m.any() else 9.0
        margin = min(d_n - 1.1, d_cc - 1.0)
        if margin < 0:
            print(f"  skip {key}: distance rule broken (C-N {d_n:.2f}, C-C {d_cc:.2f})")
            continue
        cands.append((f"N_{key}_x", at, dict(family="nframe", margin=round(float(margin), 3), t_fs=t, nc_min=round(float(d_n), 3),
                                              n_height=round(float(zn), 2)), len(drop)))

print(f"{len(cands)} candidate frames")
if len(cands) > a.max:
    idx = sorted(set(np.linspace(0, len(cands) - 1, a.max).round().astype(int)))
    cands = [cands[i] for i in idx]
Path(a.dst).mkdir(parents=True, exist_ok=True)
out = connect(a.dst + "pending.db", append=False)
for kind, at, kv, nd in cands:
    kv = dict(kv)
    kv["kind"] = kind
    kv["hash"] = hashlib.sha1(np.round(at.positions, 4).tobytes()).hexdigest()[:16]
    out.write(at, **kv)
    print(f"  {kind:34s} {len(at):3d} atoms (removed {nd})  N-C {kv['nc_min']:.2f}  N height {kv['n_height']:6.2f}  margin {kv['margin']:.2f}")
print(len(cands), "frames written to", a.dst + "pending.db")

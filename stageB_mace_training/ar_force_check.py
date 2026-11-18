"""Sign check of the force on the projectile: is a model's repulsive wall too soft or too hard compared with DFT?

For every labelled impact snapshot the force on the Ar atom is projected on the unit vector from the nearest carbon atom to
the Ar (positive = pushed away from the carbon = repulsive).  Per method it prints this radial component, the ratio
model/DFT, and the part of the error perpendicular to that axis.

    python ar_force_check.py --models zsomat=$HOME/.cache/mace/maceomat0mediummodel | tee ar_force_check.txt

Needs data/impacts/selected.extxyz and data/impacts/labelled.db (as impacts.py report).  ratio < 1 : model too soft.
"""
import argparse
import os

import numpy as np
from ase.db import connect
from ase.io import read

import recipe as R

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--dir", default="data/impacts")
p.add_argument("--models", nargs="*", default=[])
p.add_argument("--dtype", default="float64")
p.add_argument("--threads", type=int, default=8)
p.add_argument("--min-force", type=float, default=1.0, help="frames with DFT |F_Ar| below this are excluded from the summary ratio (eV/A)")
args = p.parse_args()

lab = {r.hash: r for r in connect(os.path.join(args.dir, "labelled.db"), serial=True).select()}
geoms = [g for g in read(os.path.join(args.dir, "selected.extxyz"), ":") if g.info["hash"] in lab]
print(f"{len(geoms)} labelled snapshots")

methods = dict(s.split("=", 1) for s in args.models)
forces = {"DFT": [np.array(lab[g.info["hash"]].forces) for g in geoms]}
if any("airebo_forces" in g.arrays for g in geoms):
    forces["airebo+zbl"] = [g.arrays["airebo_forces"] for g in geoms]
for name, spec in methods.items():
    if R.TOY:
        from ase.calculators.emt import EMT
        calc = EMT()
    else:
        import torch
        torch.set_num_threads(args.threads)
        from mace.calculators import MACECalculator
        calc = MACECalculator(model_paths=os.path.expanduser(spec), device="cpu", default_dtype=args.dtype)
    out = []
    for g in geoms:
        a = g.copy()
        a.calc = calc
        out.append(np.array(a.get_forces()))
    forces[name] = out

names = [k for k in forces if k != "DFT"]
print(f"\nradial force on Ar (eV/A, + = repulsive, along nearest-C -> Ar); ratio = model / DFT")
print(f"{'snapshot':22s} {'ArC':>5s} {'DFT':>7s} " + " ".join(f"{k[:10]:>10s} {'ratio':>6s} {'perp':>5s}" for k in names))
ratios = {k: [] for k in names}
for i, g in enumerate(geoms):
    n = len(g) - 1
    d = g.get_distances(n, range(n), mic=True)
    j = int(np.argmin(d))
    v = g.get_distance(j, n, mic=True, vector=True)
    u = v / np.linalg.norm(v)
    fd = forces["DFT"][i][n]
    rd = float(fd @ u)
    row = f"{g.info['target'] + '_E' + str(int(g.info['e_kin'])) + '_t' + str(int(g.info['t_fs'])):22s} {d[j]:5.2f} {rd:7.2f} "
    for k in names:
        fm = forces[k][i][n]
        rm = float(fm @ u)
        perp = float(np.linalg.norm((fm - fd) - ((fm - fd) @ u) * u))
        ratio = rm / rd if abs(rd) > 1e-9 else float("nan")
        if np.linalg.norm(fd) >= args.min_force:
            ratios[k].append(ratio)
        row += f" {rm:10.2f} {ratio:6.2f} {perp:5.2f}"
    print(row)
print(f"\nmedian model/DFT radial-force ratio on frames with DFT |F_Ar| >= {args.min_force} eV/A (1 = right, <1 = too soft, >1 = too hard):")
for k in names:
    r = np.array(ratios[k])
    print(f"  {k:12s} median {np.median(r):.2f}   min {r.min():.2f}   max {r.max():.2f}   (n = {len(r)})")

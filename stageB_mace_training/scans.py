"""Ar-C training scans: an Ar atom at fixed distances above the (DFT-relaxed) 4x4 graphene sheet, plus the isolated Ar atom.

Why: the fine-tuned models must contain Ar (mace_run_train ... --foundation_model_elements True) and need Ar-C labels.  The dynamic
impact snapshots (impacts.py) stay the TEST set; these static scans are the TRAINING data for the Ar-C interaction.

    python scans.py                       # writes data/arscan/pending.db (+ geometries.extxyz); 37 rows
    python dft_flow.py --pending data/arscan/pending.db --train data/arscan/labelled.db --workdir data/arscan

Rows (family '<family>-atom' / '<family>-scan', default family 'ar'; for nitrogen: --proj N --family n --proj-magmom 3.0):
  1 isolated Ar atom in the sheet cell (E0 of Ar in this DFT recipe)
  27 = 3 sites (top, bridge, hollow) x 9 closest Ar-C distances (1.65 ... 4.5 A), flat sheet
  9  = 3 sites x 3 distances (1.8, 2.1, 2.6 A) with a rattled sheet (sigma 0.04 A) and the Ar shifted sideways by <= 0.25 A
No pair closer than the recipe's minimum (Ar-C 1.6 A) is written.
"""
import argparse
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.build import graphene
from ase.db import connect
from ase.io import read, write

import recipe as R

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--src", default="../stageA_dft_check/dft_cheap/relaxed")
p.add_argument("--dir", default="data/arscan")
p.add_argument("--n", type=int, default=4)
p.add_argument("--proj", default="Ar")
p.add_argument("--family", default="ar", help="row families are <family>-atom and <family>-scan (use n for nitrogen)")
p.add_argument("--proj-magmom", type=float, default=0.0,
               help="initial magnetic moment of the projectile (muB), e.g. 3.0 for N; the sheet atoms keep the recipe moment. 0 = recipe default for all atoms")
p.add_argument("--dist", nargs="+", type=float, default=[1.65, 1.8, 1.95, 2.1, 2.3, 2.6, 3.0, 3.5, 4.5],
               help="closest projectile-C distances of the flat scans (A)")
p.add_argument("--rattle-dist", nargs="+", type=float, default=[1.8, 2.1, 2.6])
p.add_argument("--rattle", type=float, default=0.04, help="gaussian sigma of the sheet displacement (A)")
p.add_argument("--shift", type=float, default=0.25, help="max sideways shift of the projectile in the rattled copies (A)")
p.add_argument("--seed", type=int, default=7)
args = p.parse_args()
D = Path(args.dir)
D.mkdir(parents=True, exist_ok=True)

f = Path(args.src) / "gpaw_pristine_n4.extxyz"
a = 2.46 if (R.TOY or not f.exists()) else float(np.linalg.norm(read(f).cell[0])) / args.n
sheet = graphene(a=a, size=(args.n, args.n, 1), vacuum=10.0)
sheet.pbc = True
print(f"lattice constant {a:.4f} A, {len(sheet)} C atoms")
acc = a / np.sqrt(3)                                        # C-C bond length

centre = np.array([sheet.cell[0, 0] / 2 + sheet.cell[1, 0] / 2, sheet.cell[1, 1] / 2])
i0 = int(np.argmin(np.linalg.norm(sheet.positions[:, :2] - centre, axis=1)))
d0 = sheet.get_distances(i0, range(len(sheet)), mic=True)
d0[i0] = 9e9
nb = [int(k) for k in np.argsort(d0)[:2]]
sites = {                                                   # xy of the projectile and the lateral offset s to the nearest C atom
    "top": (sheet.positions[i0, :2], 0.0),
    "bridge": (0.5 * (sheet.positions[i0, :2] + sheet.positions[nb[0], :2]), acc / 2),
    "hollow": (sheet.positions[i0, :2] + sum(sheet.get_distance(i0, k, mic=True, vector=True) for k in nb)[:2], acc),
}

rng = np.random.default_rng(args.seed)
rows = []


def add(atoms, family, kind):
    at = atoms.copy()
    if args.proj_magmom:
        at.set_initial_magnetic_moments([R.RECIPE["init_magmom"]] * (len(at) - 1) + [args.proj_magmom])
    at = R.normalise(at)
    ok, margin = R.worst_pair(at)
    if not ok:
        print(f"  skipped {kind}: closest pair violates the recipe minimum (margin {margin:.2f})")
        return
    at.info.update(kind=kind, family=family, hash=R.structure_hash(at))
    rows.append((at, family, kind, margin))


# 1. isolated projectile atom, in the same cell (same k-point mesh and box as the sheet rows)
iso = Atoms(args.proj, positions=[[0.0, 0.0, sheet.positions[:, 2].mean()]], cell=sheet.cell, pbc=True)
add(iso, f"{args.family}-atom", f"{args.proj}_isolated")
# 2. flat scans
for site, (xy, s) in sites.items():
    for d in args.dist:
        h = float(np.sqrt(max(d * d - s * s, 0.0)))
        at = sheet.copy()
        at += Atoms(args.proj, positions=[[xy[0], xy[1], at.positions[:, 2].mean() + h]])
        add(at, f"{args.family}-scan", f"{site}_d{d:.2f}")
# 3. rattled copies
for site, (xy, s) in sites.items():
    for d in args.rattle_dist:
        h = float(np.sqrt(max(d * d - s * s, 0.0)))
        at = sheet.copy()
        at.positions += rng.normal(0.0, args.rattle, at.positions.shape)
        sh = rng.uniform(-args.shift, args.shift, 2)
        at += Atoms(args.proj, positions=[[xy[0] + sh[0], xy[1] + sh[1], sheet.positions[:, 2].mean() + h]])
        add(at, f"{args.family}-scan", f"{site}_d{d:.2f}_rattled")

pdb = connect(D / "pending.db")
have = {r.hash for r in pdb.select()}
new = 0
for at, family, kind, margin in rows:
    if at.info["hash"] in have:
        continue
    pdb.write(at, hash=at.info["hash"], family=family, kind=kind, margin=round(margin, 3))
    new += 1
write(D / "geometries.extxyz", [r[0] for r in rows])
print(f"{len(rows)} structures, {new} new rows in {D}/pending.db; geometries in {D}/geometries.extxyz")
for at, family, kind, margin in rows[:3] + rows[-2:]:
    n = len(at) - 1
    dm = at.get_distances(n, range(n), mic=True).min() if kind != f"{args.proj}_isolated" else float("nan")
    print(f"  {kind:26s} {len(at):3d} atoms  closest {args.proj}-C {dm:.2f} A  margin {margin:.2f}")
print(f"next:\n  python dft_flow.py --pending {D}/pending.db --train {D}/labelled.db --workdir {D}")

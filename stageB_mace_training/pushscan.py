"""Bond-breaking test without Ar: push ONE carbon atom ('push1') or the TWO atoms of one C-C bond ('push2', the bridge-impact pair)
rigidly down through a DFT-relaxed 4x4 sheet, everything else frozen.  Same labels (recipe) for DFT, then pushcompare.py compares the
energy along the path for DFT / AIREBO / MACE.  Why: at a bridge hit the potentials disagree on whether the two atoms are pushed out
(MACE: divacancy 3/3) or the sheet springs back (AIREBO: 0/5); Ar is inside 1.6 A there, so DFT cannot label the impact frames.

    python pushscan.py                       # data/pushscan/pending.db, 2 x 11 rows (s = 0 ... 2.5 A)
    export OMP_NUM_THREADS=1
    nohup nice -n 10 mpirun -np 5 python dft_flow.py --pending data/pushscan/pending.db --train data/pushscan/labelled.db --workdir data/pushscan > pushscan_dft.log 2>&1 &
    python -W ignore pushcompare.py --airebo ../stageA_dft_check/CH.airebo --models data/ft/omat-0_all_ar/omat-0_all_ar.model
The pushed atoms stay within 2.5 A of the sheet plane: further out they are (almost) isolated carbon atoms, which do not converge in SCF.
"""
import argparse
from pathlib import Path

import numpy as np
from ase.build import graphene
from ase.db import connect
from ase.io import read, write

import recipe as R

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--src", default="../stageA_dft_check/dft_cheap/relaxed")
p.add_argument("--dir", default="data/pushscan")
p.add_argument("--n", type=int, default=4)
p.add_argument("--s", nargs="+", type=float, default=[0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5],
               help="downward displacement of the pushed atom(s), A")
args = p.parse_args()
D = Path(args.dir)
D.mkdir(parents=True, exist_ok=True)

f = Path(args.src) / "gpaw_pristine_n4.extxyz"
a = 2.46 if (R.TOY or not f.exists()) else float(np.linalg.norm(read(f).cell[0])) / args.n
sheet = graphene(a=a, size=(args.n, args.n, 1), vacuum=10.0)
sheet.pbc = True
cen = (sheet.cell[0, :2] + sheet.cell[1, :2]) / 2
i0 = int(np.argmin(np.linalg.norm(sheet.positions[:, :2] - cen, axis=1)))
d0 = sheet.get_distances(i0, range(len(sheet)), mic=True)
d0[i0] = 9e9
j0 = int(np.argmin(d0))
print(f"lattice constant {a:.4f} A, {len(sheet)} C atoms; pushed atoms {i0} (and {j0}), bond {d0[j0]:.3f} A")

rows = []
for family, idx in (("push1", [i0]), ("push2", [i0, j0])):
    for s in args.s:
        at = sheet.copy()
        at.positions[idx, 2] -= s
        at = R.normalise(at)
        ok, margin = R.worst_pair(at)
        if not ok:
            print(f"  skipped {family} s={s}: pair violates the recipe minimum (margin {margin:.2f})")
            continue
        at.info.update(kind=f"{family}_s{s:.2f}", family=family, hash=R.structure_hash(at))
        rows.append((at, family, at.info["kind"], margin))

pdb = connect(D / "pending.db")
have = {r.hash for r in pdb.select()}
new = 0
for at, family, kind, margin in rows:
    if at.info["hash"] in have:
        continue
    pdb.write(at, hash=at.info["hash"], family=family, kind=kind, margin=round(margin, 3), s=float(kind.split("_s")[1]))
    new += 1
write(D / "geometries.extxyz", [r[0] for r in rows])
print(f"{len(rows)} structures, {new} new rows in {D}/pending.db")
print(f"next:\n  nohup nice -n 10 mpirun -np 5 python dft_flow.py --pending {D}/pending.db --train {D}/labelled.db --workdir {D} > pushscan_dft.log 2>&1 &")

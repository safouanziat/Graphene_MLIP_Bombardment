"""Nitrogen-doped graphene structures for the DFT flow (Stage D, block D3 and the N2 reference).  Writes data/ndefects/pending.db.

Starting points are the Stage A DFT-relaxed 4x4 sheets (../stageA_dft_check/dft_cheap/relaxed/gpaw_<fam>_n4.extxyz).  Nitrogen is put in by
editing them (no relaxation here: single points at these geometries plus rattled / strained copies are labelled, as for the carbon families):

  n-graphitic   pristine, one C -> N (substitution in the lattice)                                32 atoms
  n-adatom-br   the relaxed C adatom (bridge site) -> N                                             33 atoms
  n-adatom-top  pristine + N above one atom, N-C 1.45 A                                             33 atoms
  n-v1-1n / 2n / 3n   V1 with 1, 2 or all 3 of the atoms nearest to the vacancy replaced by N       31 atoms
  n-v2-2n / 4n  V2 with the 2 most distant / all 4 atoms nearest to the vacancy centre replaced by N  30 atoms
  n-ref         N2 molecule (bond 1.10 A) in the sheet cell: reference energy for formation energies

The vacancy is located by comparing with the pristine sheet; the atoms 'nearest to the vacancy' are an approximation of the pyridinic sites
(the relaxed V1 / V2 geometries are reconstructed, so this is not a coordination count).  Pyrrolic (N in a five-ring) and N interstitials need a
reconstruction that is not built here; they should come from a model relaxation / active learning later.
Each family = the edited structure + --per-family rattled (0.03 / 0.06 / 0.10 / 0.15 A) and strained (+-2 %) copies, like configs.py.
Initial moments are the recipe default (0.3 muB per atom).  Rows closer than the recipe minimum (C-N 1.1, N-N 1.0 A) are rejected.

    python ndefects.py                       # data/ndefects/pending.db
    nohup nice -n 10 mpirun -np 5 python dft_flow.py --pending data/ndefects/pending.db --train data/ndefects/labelled.db --workdir data/ndefects > ndefects_dft.log 2>&1 &
"""
import argparse
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.db import connect
from ase.io import read, write

import recipe as R

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--src", default="../stageA_dft_check/dft_cheap/relaxed")
p.add_argument("--dir", default="data/ndefects")
p.add_argument("--per-family", type=int, default=4)
p.add_argument("--rattle", nargs="+", type=float, default=[0.03, 0.06, 0.10, 0.15])
p.add_argument("--strain", type=float, default=0.02)
p.add_argument("--seed", type=int, default=3)
args = p.parse_args()
D = Path(args.dir)
D.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(args.seed)


def load(fam):
    f = Path(args.src) / f"gpaw_{fam}_n4.extxyz"
    if not f.exists():
        raise SystemExit(f"missing {f}")
    at = read(f)
    at.calc = None
    at.set_initial_magnetic_moments(np.zeros(len(at)))
    return at


def to_n(at, idx):
    at = at.copy()
    s = at.get_chemical_symbols()
    for i in idx:
        s[i] = "N"
    at.set_chemical_symbols(s)
    return at


def vacancy_centre(pristine, defect):
    """mean position of the pristine sites that have no atom of the defect structure within 1.0 A (periodic in x, y)"""
    miss = []
    for q in pristine.positions:
        t = defect.copy()
        t += Atoms("X", positions=[q])
        dd = t.get_distances(len(t) - 1, range(len(t) - 1), mic=True)
        if dd.min() > 1.0:
            miss.append(q)
    if not miss:
        raise SystemExit("no vacancy found when comparing with the pristine sheet")
    return np.array(miss), np.mean(miss, axis=0)


def nearest_to(defect, centre, k):
    t = defect.copy()
    t += Atoms("X", positions=[centre])
    d = t.get_distances(len(t) - 1, range(len(t) - 1), mic=True)
    return [int(i) for i in np.argsort(d)[:k]]


pristine = load("pristine")
families = {}                                                      # name -> edited structure

# graphitic: the atom nearest the cell centre
cen = (pristine.cell[0] + pristine.cell[1]) / 2
i0 = int(np.argmin(np.linalg.norm(pristine.positions[:, :2] - cen[:2], axis=1)))
families["n-graphitic"] = to_n(pristine, [i0])

# adatom, bridge: the relaxed C adatom (the atom furthest from the sheet plane) becomes N
ad = load("adatom")
z = ad.positions[:, 2] - np.median(ad.positions[:, 2])
families["n-adatom-br"] = to_n(ad, [int(np.argmax(np.abs(z)))])

# adatom, top: N 1.45 A straight above the atom nearest the cell centre
top = pristine.copy()
top += Atoms("N", positions=[[*pristine.positions[i0, :2], pristine.positions[i0, 2] + 1.45]])
families["n-adatom-top"] = top

# vacancies
v1 = load("V1")
_, c1 = vacancy_centre(pristine, v1)
n1 = nearest_to(v1, c1, 3)
families["n-v1-1n"] = to_n(v1, n1[:1])
families["n-v1-2n"] = to_n(v1, n1[:2])
families["n-v1-3n"] = to_n(v1, n1)

v2 = load("V2")
_, c2 = vacancy_centre(pristine, v2)
n2 = nearest_to(v2, c2, 4)
best = max(((a, b) for a in n2 for b in n2 if a < b), key=lambda ab: v2.get_distance(ab[0], ab[1], mic=True))
families["n-v2-2n"] = to_n(v2, list(best))
families["n-v2-4n"] = to_n(v2, n2)

# N2 molecule in the sheet cell
ref = Atoms("N2", positions=[[0, 0, 0], [1.10, 0, 0]], cell=pristine.cell, pbc=True)
ref.positions += [cen[0] - 0.55, cen[1], pristine.positions[:, 2].mean()]

pdb = connect(D / "pending.db")
have = {r.hash for r in pdb.select()}
out, new, rej = [], 0, 0


def add(at, family, kind):
    global new, rej
    at = R.normalise(at)
    h = R.structure_hash(at)
    ok, margin = R.worst_pair(at)
    if not ok:
        rej += 1
        print(f"  rejected {family} {kind}: closest pair margin {margin:.2f} A")
        return
    out.append(at)
    if h in have:
        return
    pdb.write(at, hash=h, family=family, kind=kind, margin=round(margin, 3))
    have.add(h)
    new += 1


for fam, base in families.items():
    add(base, fam, "edited")
    for i in range(args.per_family):
        at = base.copy()
        amp = args.rattle[i % len(args.rattle)]
        s = 1 + rng.uniform(-args.strain, args.strain)
        cell = np.array(at.cell)
        cell[:2] *= s
        at.set_cell(cell, scale_atoms=True)
        at.rattle(stdev=amp, seed=int(rng.integers(1 << 30)))
        add(at, fam, f"rattle{amp:.2f}_s{s:.3f}")
add(ref, "n-ref", "N2_molecule")

write(D / "geometries.extxyz", out)
from collections import Counter
cnt = Counter(r.family for r in pdb.select())
print(f"{len(out)} structures, {new} new rows, {rej} rejected, in {D}/pending.db")
for k, v in cnt.items():
    sym = next(r.toatoms().get_chemical_symbols() for r in pdb.select(family=k))
    print(f"  {k:14s} {v:2d} rows  {len(sym):2d} atoms  N atoms: {sym.count('N')}")
print(f"next:\n  export OMP_NUM_THREADS=1\n  nohup nice -n 10 mpirun -np 5 python dft_flow.py --pending {D}/pending.db --train {D}/labelled.db "
      f"--workdir {D} > ndefects_dft.log 2>&1 &")

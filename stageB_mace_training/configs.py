"""
Generate the UNLABELLED structures that the DFT flow will label.  Writes data/pending.db (ASE db, key 'hash').

Families (all 4x4, 32 +- 2 atoms, geometries from Stage A):
  - Stage-A relaxed structures (pristine, V1, V2, SW, adatom, frenkel) with random displacements of several
    amplitudes and small biaxial strains                                      -> near-equilibrium and thermal-like
  - impact snapshots (--impacts file.extxyz): AIREBO+ZBL snapshots already cropped to a small cell -> far from
    equilibrium (cropping is a separate step, see README)
Every structure is normalised to the recipe geometry, rejected if two atoms are closer than recipe min_dist, and
skipped if its hash is already in pending.db.   Re-running is safe (idempotent).

    python configs.py --src ../stageA_dft_check/dft_cheap/relaxed --per-family 12
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from ase.db import connect
from ase.io import read

import recipe as R

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--src", default="../stageA_dft_check/dft_cheap/relaxed")
p.add_argument("--db", default="data/pending.db")
p.add_argument("--families", nargs="+", default=["pristine", "V1", "V2", "SW", "adatom", "frenkel"])
p.add_argument("--per-family", type=int, default=12, help="perturbed copies per family (plus the relaxed one)")
p.add_argument("--rattle", nargs="+", type=float, default=[0.03, 0.06, 0.10, 0.15])
p.add_argument("--strain", type=float, default=0.02, help="max |biaxial strain|")
p.add_argument("--impacts", help="extxyz with small-cell impact snapshots")
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()

Path(args.db).parent.mkdir(parents=True, exist_ok=True)
db = connect(args.db)
have = {r.hash for r in db.select()}
rng = np.random.default_rng(args.seed)
n_new = n_dup = n_rej = 0


def add(at, family, kind):
    global n_new, n_dup, n_rej
    at = R.normalise(at)
    h = R.structure_hash(at)
    if h in have:
        n_dup += 1
        return
    ok, margin = R.worst_pair(at)
    if not ok:
        n_rej += 1
        return
    db.write(at, hash=h, family=family, kind=kind, margin=round(margin, 3))
    have.add(h)
    n_new += 1


for fam in args.families:
    f = Path(args.src) / f"gpaw_{fam}_n4.extxyz"
    if not f.exists():
        print(f"skip {fam}: {f} not found")
        continue
    base = read(f)
    base.calc = None
    add(base, fam, "relaxed")
    for i in range(args.per_family):
        at = base.copy()
        amp = args.rattle[i % len(args.rattle)]
        s = 1 + rng.uniform(-args.strain, args.strain)
        cell = np.array(at.cell)
        cell[:2] *= s
        at.set_cell(cell, scale_atoms=True)
        at.rattle(stdev=amp, seed=int(rng.integers(1 << 30)))
        add(at, fam, f"rattle{amp:.2f}_s{s:.3f}")

if args.impacts:
    for i, at in enumerate(read(args.impacts, ":")):
        add(at, "impact", f"impact{i}")

print(f"{args.db}: +{n_new} new, {n_dup} already present, {n_rej} rejected (pair closer than recipe min_dist); "
      f"total {db.count()}")

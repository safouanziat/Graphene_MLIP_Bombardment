"""
Export train.db to MACE-ready extxyz (REF_energy, REF_forces) + statistics.json + export_summary.json.

  - rows whose recipe id differs from recipe.RECIPE_ID are refused (listed, not exported)
  - every family is ONE relaxed base structure plus rattled/strained copies of it, so a random split leaks: validation
    structures are near-copies of training ones.  Three split modes (all deterministic, by structure hash):
      default          stratified k-fold inside each family (--nfolds 5, --fold 0..4).  Validation = non-relaxed rows with
                       weight 1.0 only; relaxed bases and strongly rattled rows always stay in train.  Optimistic, use it
                       for sanity checks and for comparing models on the same folds.
      --holdout-family X   leave-one-family-out: all rows of family X are validation.  Honest test of generalisation
                       to an unseen defect type (use it for the headline numbers).
    The decisive test remains the separate acceptance run (defect energies vs converged DFT).
  - per-structure weight (info key 'config_weight', understood by MACE): strongly rattled rows lie far above the
    relaxed energies (rattle0.15 ~ +1.2 eV/atom) and would dominate the loss, so they are down-weighted
    (--weight-rattle, default 0.3 for amplitude >= --rattle-threshold 0.15)
  - statistics.json: element table, energy/atom mean+std, force RMS, average neighbour count (cutoff --r-max),
    counts per family, recipe id  -> a training run can be reproduced from the files alone
  - export_summary.json: sanity check per split (finite energies/forces, targets present, weights used)
"""
import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

import recipe as R

from ase.db import connect
from ase.io import write
from ase.neighborlist import neighbor_list

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--train", default="data/train.db")
p.add_argument("--out", default="data")
p.add_argument("--nfolds", type=int, default=5)
p.add_argument("--fold", type=int, default=0, help="fold index 0..nfolds-1 used as validation")
p.add_argument("--holdout-family", default="", help="leave-one-family-out: this family is the validation set")
p.add_argument("--extra-train", nargs="*", default=[], help="further labelled databases whose multi-atom rows ALL go into train "
               "(e.g. data/arscan/labelled.db); single-atom rows are isolated-atom references and are not exported")
p.add_argument("--valid-copies", type=float, default=0.0,
               help="with --fold -1: fraction of the rows copied into valid.extxyz while ALL rows stay in train (0 = off)")
p.add_argument("--extra-fmax-weight", type=float, default=0.0,
               help="extra-train rows whose largest force exceeds this (eV/A) get --weight-rattle instead of 1.0; 0 = off (old behaviour)")
p.add_argument("--extra-holdout-family", nargs="*", default=[],
               help="families of the extra databases that go to VALIDATION instead of train (all their rows)")
p.add_argument("--r-max", type=float, default=5.0, help="cutoff used for the average-neighbour statistic")
p.add_argument("--weight-rattle", type=float, default=0.3, help="config_weight of strongly rattled rows")
p.add_argument("--rattle-threshold", type=float, default=0.15, help="rattle amplitude from which rows are down-weighted")
args = p.parse_args()


def weight_of(kind):
    m = re.match(r"rattle([0-9.]+)", kind or "")
    return args.weight_rattle if m and float(m.group(1)) >= args.rattle_threshold - 1e-9 else 1.0


rows = [r for r in connect(args.train, serial=True).select()]
refused = sum(r.get("recipe") != R.RECIPE_ID for r in rows)
rows = [r for r in rows if r.get("recipe") == R.RECIPE_ID]
# balanced folds: inside each family the eligible rows are ordered by a hash of their id and dealt out round-robin
fold = {}
for fam in sorted({r.family for r in rows}):
    elig = [r for r in rows if r.family == fam and r.get("kind") != "relaxed" and weight_of(r.get("kind")) == 1.0]
    elig.sort(key=lambda r: hashlib.sha1(r.hash.encode()).hexdigest())
    fold.update({r.hash: i % args.nfolds for i, r in enumerate(elig)})

tr, va = [], []
for r in rows:
    at = r.toatoms()
    at.info["REF_energy"] = float(r.energy)
    at.info["config_type"] = r.family
    at.info["config_weight"] = weight_of(r.get("kind"))
    at.info["hash"] = r.hash
    at.arrays["REF_forces"] = r.forces
    at.calc = None
    if args.holdout_family:
        is_val = r.family == args.holdout_family
    else:
        is_val = fold.get(r.hash) == args.fold
    (va if is_val else tr).append(at)

n_extra = 0
n_extra_val = 0
for dbpath in args.extra_train:
    for r in connect(dbpath, serial=True).select():
        if r.get("recipe") != R.RECIPE_ID:
            refused += 1
            continue
        if r.natoms < 2:
            continue
        at = r.toatoms()
        at.info["REF_energy"] = float(r.energy)
        at.info["config_type"] = r.family
        at.info["config_weight"] = 1.0
        if args.extra_fmax_weight > 0 and float(np.abs(r.forces).max()) > args.extra_fmax_weight:
            at.info["config_weight"] = args.weight_rattle      # very distorted extra rows (huge forces) must not dominate the loss
        at.info["hash"] = r.hash
        at.arrays["REF_forces"] = r.forces
        at.calc = None
        if r.family in args.extra_holdout_family:             # honest test on a family that lives in an extra database
            at.info["config_weight"] = 1.0
            va.append(at)
            n_extra_val += 1
            continue
        tr.append(at)
        n_extra += 1
if args.extra_train:
    print(f"{n_extra} rows from {', '.join(args.extra_train)} added to train, {n_extra_val} to validation")

if args.valid_copies > 0 and not va:
    # final model: EVERY row stays in train; the validation file only holds COPIES of a deterministic subset (checkpoint
    # selection then follows the fit, not generalisation).  Without this, mace_run_train --valid_fraction removes random
    # rows from training, and singleton references (N2 molecule, the pristine row) can be lost.
    k = max(1, int(round(args.valid_copies * len(tr))))
    pick = sorted(range(len(tr)), key=lambda i: hashlib.sha1(tr[i].info["hash"].encode()).hexdigest())[:k]
    va = [tr[i].copy() for i in pick]
    for a, i in zip(va, pick):
        a.info.update(tr[i].info)
        a.arrays["REF_forces"] = tr[i].arrays["REF_forces"].copy()
        a.info["config_weight"] = 1.0
    print(f"{len(va)} validation rows are copies of training rows (all rows train)")

out = Path(args.out)
out.mkdir(exist_ok=True)
write(out / "train.extxyz", tr)
write(out / "valid.extxyz", va)


def check(split):
    bad, w = 0, Counter()
    for a in split:
        f = a.arrays["REF_forces"]
        if not (np.isfinite(a.info["REF_energy"]) and np.isfinite(f).all() and f.shape == (len(a), 3)):
            bad += 1
        w[a.info["config_weight"]] += 1
    return dict(n=len(split), invalid=bad, weights={str(k): v for k, v in sorted(w.items())},
                families=dict(Counter(a.info["config_type"] for a in split)))


def stats(split):
    if not split:
        return {}
    z = sorted({int(x) for a in split for x in a.numbers})
    epa = np.array([a.info["REF_energy"] / len(a) for a in split])
    frms = float(np.sqrt(np.mean(np.concatenate([a.arrays["REF_forces"].ravel() for a in split]) ** 2)))
    nn = np.mean([len(neighbor_list("i", a, args.r_max)) / len(a) for a in split])
    # least-squares per-element reference energies (the usual starting point for MACE E0s='average')
    cnt = np.array([[int((a.numbers == x).sum()) for x in z] for a in split], float)
    e = np.array([a.info["REF_energy"] for a in split])
    e0, *_ = np.linalg.lstsq(cnt, e, rcond=None)
    return dict(recipe=R.RECIPE_ID, atomic_numbers=z, r_max=args.r_max,
                energy_per_atom_mean=float(epa.mean()), energy_per_atom_std=float(epa.std()),
                force_rms=frms, avg_num_neighbors=float(nn),
                atomic_energies_average={str(x): float(v) for x, v in zip(z, e0)})


(out / "statistics.json").write_text(json.dumps(stats(tr), indent=2))
summary = dict(recipe=R.RECIPE_ID, refused_other_recipe=refused, train=check(tr), valid=check(va))
(out / "export_summary.json").write_text(json.dumps(summary, indent=2))
print(f"recipe {R.RECIPE_ID}: {len(tr)} train, {len(va)} validation, {refused} refused (other recipe)")
print(f"weights in train: {summary['train']['weights']}  invalid: {summary['train']['invalid']}+{summary['valid']['invalid']}")
print(f"wrote {out}/train.extxyz valid.extxyz statistics.json export_summary.json")

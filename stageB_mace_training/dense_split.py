"""Split the labelled dense hollow frames into a TRAIN part (appended to a training database) and a held-out TEST part.

    cp data/nimpact/labelled_impacts_all_v2.db data/nimpact/labelled_impacts_all_v3.db
    python dense_split.py

The split is deterministic (hash order), stratified by impact energy and done by WHOLE IMPACT: frames of one impact (a few fs apart, nearly
identical) always stay together, so the test set is never a near-copy of the training set.  One quarter of the impacts of every energy
(at least one) is held out, so every energy is represented in both parts.  Frames whose total moment is outside [--mu-min, --mu-max] are left out of TRAIN (odd spin
states, see the earlier tests) but are still reported; they stay out of TEST too, because their energy labels are the doubtful part.
The test set is written to data/nimpact_dense_test/labelled.db, to be used with:
    python -W ignore nimpact.py compare --model <model> --dir data/nimpact_dense_test
"""
import argparse
import hashlib
import math
from collections import defaultdict
from pathlib import Path

from ase.db import connect

p = argparse.ArgumentParser()
p.add_argument("--src", default="data/nimpact_dense/labelled.db")
p.add_argument("--train-dst", default="data/nimpact/labelled_impacts_all_v3.db")
p.add_argument("--test-dst", default="data/nimpact_dense_test/labelled.db")
p.add_argument("--test-frac", type=float, default=0.25)
p.add_argument("--mu-min", type=float, default=-0.5)
p.add_argument("--mu-max", type=float, default=3.2)
a = p.parse_args()

Path(a.test_dst).parent.mkdir(parents=True, exist_ok=True)
rows = list(connect(a.src).select())
groups, odd = defaultdict(lambda: defaultdict(list)), []
for r in rows:
    mu = float(r.get("mu", 0))
    if not (a.mu_min <= mu <= a.mu_max):
        odd.append((r.kind, mu))
        continue
    impact = r.kind.rsplit("_t", 1)[0]                 # N_hollow_E<energy>_r<run>
    groups[impact.split("_")[2]][impact].append(r)
tr = connect(a.train_dst)
te = connect(a.test_dst)
have_tr = {r.hash for r in tr.select()}
have_te = {r.hash for r in te.select()}
n_tr = n_te = 0
for e, imp in sorted(groups.items()):
    order = sorted(imp, key=lambda t: hashlib.sha1(t.encode()).hexdigest())
    k = max(1, math.ceil(a.test_frac * len(order)))
    test_imp = set(order[:k])
    nt = ne = 0
    for impact, lst in imp.items():
        for r in lst:
            kv = dict(r.key_value_pairs)
            kv["family"] = "nframe-hollow"
            if impact in test_imp:
                ne += 1
                if r.hash not in have_te:
                    te.write(r.toatoms(), key_value_pairs=kv)
                    n_te += 1
            else:
                nt += 1
                if r.hash not in have_tr:
                    tr.write(r.toatoms(), key_value_pairs=kv)
                    n_tr += 1
    print(f"{e}: {len(imp)} impacts -> held out {sorted(test_imp)}; {nt} train frames, {ne} test frames")
for kind, mu in odd:
    print(f"left out (odd spin): {kind} mu {mu:.2f}")
print(f"\n{len(rows)} labelled frames read; written {n_tr} to {a.train_dst}, {n_te} to {a.test_dst}")

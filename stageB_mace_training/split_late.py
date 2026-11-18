import argparse
from pathlib import Path
from ase.db import connect

p = argparse.ArgumentParser()
p.add_argument("--src", nargs="+", default=["data/nimpact_final_n7_rm/labelled.db", "data/nimpact_final_n7_iso/labelled.db"])
p.add_argument("--train-dst", default="data/nimpact/labelled_impacts_all_v4.db")
p.add_argument("--test-dst", default="data/nimpact_late_test/labelled.db")
p.add_argument("--mu-min", type=float, default=-0.5)
p.add_argument("--mu-max", type=float, default=3.2)
a = p.parse_args()
Path(a.test_dst).parent.mkdir(parents=True, exist_ok=True)
tr = connect(a.train_dst)
te = connect(a.test_dst)
have_tr = {r.hash for r in tr.select()}
have_te = {r.hash for r in te.select()}
n_tr = n_te = 0
for src in a.src:
    for r in connect(src).select():
        mu = float(r.get("mu", 0))
        if not (a.mu_min <= mu <= a.mu_max):
            print(f"left out (odd spin): {r.kind} mu {mu:.2f}")
            continue
        parts = r.kind.split("_")          # N_<site>_E<e>_r<run>_t<fs>fs_<rm|iso>
        kv = dict(r.key_value_pairs)
        kv["family"] = "nframe-" + parts[1]
        if parts[3] == "r0":
            if r.hash not in have_tr:
                tr.write(r.toatoms(), key_value_pairs=kv)
                n_tr += 1
        elif r.hash not in have_te:
            te.write(r.toatoms(), key_value_pairs=kv)
            n_te += 1
print(f"written {n_tr} frames to {a.train_dst} (expect 10), {n_te} to {a.test_dst} (expect 9)")

from pathlib import Path
from ase.db import connect

srcs = ["data/nimpact_more_all/labelled.db", "data/nimpact_late_test/labelled.db"]
tr = connect("data/nimpact/labelled_impacts_all_v5.db")
Path("data/nimpact_h34").mkdir(parents=True, exist_ok=True)
te = connect("data/nimpact_h34/labelled.db")
have_tr = {r.hash for r in tr.select()}
have_te = {r.hash for r in te.select()}
n_tr = n_te = 0
for s in srcs:
    for r in connect(s).select():
        mu = float(r.get("mu", 0))
        if not (-0.5 <= mu <= 3.2):
            print(f"left out (odd spin): {r.kind} mu {mu:.2f}")
            continue
        parts = r.kind.split("_")          # N_<site>_E<e>_r<run>_t<fs>fs_<suffix>
        kv = dict(r.key_value_pairs)
        kv["family"] = "nframe-" + parts[1]
        if parts[3] in ("r3", "r4"):
            if r.hash not in have_te:
                te.write(r.toatoms(), key_value_pairs=kv)
                n_te += 1
        elif r.hash not in have_tr:
            tr.write(r.toatoms(), key_value_pairs=kv)
            n_tr += 1
print(f"written {n_tr} frames to v5 (expect 22), {n_te} to data/nimpact_h34 (expect 17)")

import os, hashlib
import numpy as np
from ase.db import connect
src = "data/nimpact_final_n7/pending.db"
dst = "data/nimpact_final_n7_iso/"
want = ["N_top_E20_r0_t65fs", "N_top_E20_r1_t70fs", "N_top_E20_r3_t80fs", "N_top_E20_r4_t75fs",
        "N_top_E30_r3_t65fs", "N_bridge_E30_r0_t90fs", "N_bridge_E50_r0_t50fs"]
os.makedirs(dst, exist_ok=True)
out = connect(dst + "pending.db", append=False)
n = 0
for r in connect(src).select():
    if r.kind not in want:
        continue
    a = r.toatoms()
    nc = len(a) - 1
    iso = []
    for k in range(nc):
        d = a.get_distances(k, [i for i in range(len(a)) if i != k], mic=True)
        if d.min() > 2.3:
            iso.append(k)
    print(f"  {r.kind}: isolated C atoms: {iso}")
    if not iso:
        continue
    b = a[[k for k in range(len(a)) if k not in iso]]
    kv = dict(r.key_value_pairs)
    kv["kind"] = r.kind + "_iso"
    kv["hash"] = hashlib.sha1(np.round(b.positions, 4).tobytes()).hexdigest()[:16]
    out.write(b, **kv)
    n += 1
print(n, "frames written to", dst + "pending.db")

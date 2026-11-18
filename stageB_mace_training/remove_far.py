import os, hashlib
import numpy as np
from ase.db import connect
src = "data/nimpact_final_n7/pending.db"
dst = "data/nimpact_final_n7_rm/"
PARK = 7.0
os.makedirs(dst, exist_ok=True)
out = connect(dst + "pending.db", append=False)
n = 0
for r in connect(src).select():
    a = r.toatoms()
    nc = len(a) - 1
    z = a.positions[:, 2]
    zpl = np.median(z[:nc])
    far = [k for k in range(nc) if abs(z[k] - zpl) > PARK + 0.01]
    if not far:
        continue
    b = a[[k for k in range(len(a)) if k not in far]]
    kv = dict(r.key_value_pairs)
    kv["kind"] = r.kind + "_rm"
    kv["hash"] = hashlib.sha1(np.round(b.positions, 4).tobytes()).hexdigest()[:16]
    out.write(b, **kv)
    n += 1
    print(f"  {r.kind}: removed {len(far)} far C, {len(b)} atoms left")
print(n, "frames written to", dst + "pending.db")

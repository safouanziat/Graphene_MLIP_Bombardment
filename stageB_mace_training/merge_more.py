import os
from ase.db import connect
os.makedirs("data/nimpact_more_all", exist_ok=True)
out = connect("data/nimpact_more_all/labelled.db", append=False)
n = 0
for part in (0, 1):
    for r in connect(f"data/nimpact_more_{part}/labelled.db").select():
        out.write(r.toatoms(), key_value_pairs=dict(r.key_value_pairs))
        n += 1
print(n, "frames in data/nimpact_more_all/labelled.db")

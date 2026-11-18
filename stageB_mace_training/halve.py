import os
from ase.db import connect
rows = list(connect("data/nimpact_more/pending.db").select())
for part in (0, 1):
    os.makedirs(f"data/nimpact_more_{part}", exist_ok=True)
    out = connect(f"data/nimpact_more_{part}/pending.db", append=False)
    for r in rows[part::2]:
        out.write(r.toatoms(), **dict(r.key_value_pairs))
    print(part, len(rows[part::2]), "frames")

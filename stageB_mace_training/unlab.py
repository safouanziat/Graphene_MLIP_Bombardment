from ase.db import connect
p = "data/nimpact_final_n7/"
lab = {r.kind for r in connect(p + "labelled.db").select()}
for r in connect(p + "pending.db").select():
    if r.kind not in lab:
        print(r.kind)

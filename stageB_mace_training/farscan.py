import numpy as np
from ase.db import connect
for r in connect("data/nimpact_final_n7/labelled.db").select():
    a = r.toatoms()
    z = a.positions[:, 2]
    dz = np.abs(z - np.median(z))
    print(f"{r.kind:28s} max |z-sheet| = {dz.max():6.1f} A   atoms > 6 A: {int((dz > 6).sum())}")

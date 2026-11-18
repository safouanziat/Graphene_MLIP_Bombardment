"""N2 molecules at several bond lengths / heights above the sheet -> data/n2ref/pending.db (family n-ref, kind N2_d<bond>_h<height>)."""
import hashlib
import recipe as R   # first: gpaw before ase

import numpy as np
from ase import Atoms
from ase.db import connect

src = next(iter(connect("data/nimpact_final_n7_fix/pending.db").select())).toatoms()
cell = src.cell
nC = len(src) - 1
zpl = float(np.median(src.positions[:nC, 2]))
cx, cy = cell[0][:2] / 2 + cell[1][:2] / 2

jobs = [(d, 5.0) for d in (1.02, 1.06, 1.10, 1.14, 1.18, 1.25, 1.35)] + [(1.10, 3.5), (1.10, 7.0)]
out = connect("data/n2ref/pending.db", append=False)
for d, h in jobs:
    at = Atoms("N2", positions=[[cx - d / 2, cy, zpl + h], [cx + d / 2, cy, zpl + h]], cell=cell, pbc=True)
    at.set_initial_magnetic_moments([3.0, 3.0])
    kind = f"N2_d{d:.2f}_h{h:.1f}"
    out.write(at, family="n-ref", kind=kind, margin=round(d - 1.0, 3),
              hash=hashlib.sha1(np.round(at.positions, 4).tobytes()).hexdigest()[:16])
    print(kind)
print(len(jobs), "frames in data/n2ref/pending.db")

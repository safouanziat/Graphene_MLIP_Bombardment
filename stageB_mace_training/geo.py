import numpy as np
from ase.db import connect
from ase.neighborlist import neighbor_list
r = next(x for x in connect("data/nimpact_final_n7/labelled.db").select() if x.kind == "N_top_E50_r1_t275fs")
a = r.toatoms()
i, j, d = neighbor_list("ijd", a, 3.0)
print("z range of atoms:", round(a.positions[:, 2].min(), 2), round(a.positions[:, 2].max(), 2))
for k in range(len(a)):
    n = int((i == k).sum())
    if n != 3 or abs(a.positions[k, 2] - np.median(a.positions[:, 2])) > 0.8:
        print(k, a[k].symbol, "neighbours<3A:", n, "z:", round(a.positions[k, 2], 2))
print("shortest bond:", round(d.min(), 2))

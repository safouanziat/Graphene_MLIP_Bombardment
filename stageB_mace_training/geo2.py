from ase.db import connect
r = next(x for x in connect("data/nimpact_final_n7/labelled.db").select() if x.kind == "N_top_E50_r1_t275fs")
a = r.toatoms()
print("cell:", a.cell.lengths(), "pbc:", a.pbc)
d = a.get_distances(20, range(len(a)), mic=True)
print("C20 distances to others (A), 6 shortest:", sorted(d)[1:7])
a.wrap()
print("C20 wrapped z:", round(a.positions[20, 2], 2), " sheet mean z:", round(a.positions[:, 2].mean(), 2))

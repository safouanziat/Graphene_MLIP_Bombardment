from ase.db import connect
from mace.calculators import MACECalculator
c = MACECalculator(model_paths="data/ft/omat-0_all_n7/omat-0_all_n7.model", device="cpu", default_dtype="float64")
for r in connect("data/nimpact_final_n7_iso/labelled.db").select():
    a = r.toatoms()
    a.calc = c
    print(f"{r.kind:28s} {r.energy - a.get_potential_energy():8.3f}")

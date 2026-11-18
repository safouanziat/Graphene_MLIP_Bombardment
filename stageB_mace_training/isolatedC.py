import recipe as R   # first: gpaw before ase
from ase import Atoms
from ase.db import connect
from gpaw import GPAW, PW, FermiDirac
cell = next(iter(connect("data/nimpact_final_n7_fix/pending.db").select())).toatoms().cell
for M in (0, 2):
    at = Atoms("C", positions=[[0, 0, 0]], cell=cell, pbc=True)
    at.center()
    at.set_initial_magnetic_moments([float(M)])
    occ = FermiDirac(R.RECIPE["smear"], fixmagmom=True) if M else FermiDirac(R.RECIPE["smear"])
    at.calc = GPAW(mode=PW(R.RECIPE["ecut"]), xc=R.RECIPE["xc"], kpts={"size": R.kmesh(at, R.RECIPE["kdens"]), "gamma": True},
                   occupations=occ, spinpol=bool(M), txt=f"isolatedC_M{M}.txt")
    print("M =", M, " E =", round(at.get_potential_energy(), 4), "eV", flush=True)

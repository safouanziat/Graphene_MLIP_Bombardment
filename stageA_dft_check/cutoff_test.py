# Cutoff test: E_f(V1) at fixed 4x4 geometries (from dft_cheap/) for several plane-wave cutoffs.
from gpaw import GPAW, PW, FermiDirac          # keep first: GPAW must set up MPI before ASE touches it
import os
from ase.io import read
from ase.parallel import parprint, world

os.makedirs("cutoff_test", exist_ok=True)
pr = read("dft_cheap/relaxed/gpaw_pristine_n4.extxyz")
v1 = read("dft_cheap/relaxed/gpaw_V1_n4.extxyz")


def E(at, ec, tag):
    at = at.copy()
    at.pbc = True
    at.set_initial_magnetic_moments([0.3] * len(at))
    at.calc = GPAW(mode=PW(ec), xc="PBE", kpts={"size": (2, 2, 1), "gamma": True},
                   occupations=FermiDirac(0.05), spinpol=True, txt=f"cutoff_test/{tag}_{ec}.txt",
                   convergence={"energy": 1e-4, "density": 1e-4})
    return at.get_potential_energy()


ref = None
for ec in (350, 450, 550):
    ep, ev = E(pr, ec, "pristine"), E(v1, ec, "V1")
    ef = ev - len(v1) * ep / len(pr)
    ref = ef if ref is None else ref
    parprint(f"cutoff {ec} eV: E_f(V1) = {ef:.3f} eV   (change vs 350 eV: {ef - ref:+.3f} eV)", flush=True)
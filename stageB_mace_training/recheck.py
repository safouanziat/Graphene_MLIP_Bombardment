"""How reliable is one DFT force label?  Recompute a stored frame with variations of the recipe and compare the force on the N atom.

    export OMP_NUM_THREADS=1
    mpirun -np 5 python recheck.py --db data/nimpact_val2/labelled.db --kind N_hollow_E55_r0_t25fs

Variants (all else = the recipe): repeat (same settings, checks reproducibility), tight (energy / density convergence 1e-6),
ecut600 (plane-wave cutoff 600 eV), lowspin (N starts at 0 muB instead of 3), unpol (not spin-polarised).
Each runs on the SAME geometry; the table shows energy, total moment, the force on the N (vector) and its difference to the stored label.
If the N force moves by more than ~0.5 eV/A between variants, the DFT label itself is uncertain at that level and a force error of that size
on this frame says nothing about the model.  ~13 min per variant for a hollow frame on 5 ranks.
"""
import argparse
import json
import time
from pathlib import Path

import recipe as R   # first: gpaw before ase.parallel

import numpy as np
from ase.db import connect
from gpaw import GPAW, PW, FermiDirac
from gpaw.mpi import world

p = argparse.ArgumentParser()
p.add_argument("--db", required=True)
p.add_argument("--kind", required=True)
p.add_argument("--variants", nargs="*", default=["repeat", "tight", "ecut600", "lowspin", "unpol"])
p.add_argument("--out", default="recheck")
a = p.parse_args()

rows = [r for r in connect(a.db).select() if r.kind == a.kind]
assert rows, f"no row with kind {a.kind} in {a.db}"
row = rows[0]
at0 = row.toatoms()
n = len(at0) - 1
assert at0.get_chemical_symbols()[n] == "N"
ref = np.array(row.forces)[n]
out = Path(a.out)
out.mkdir(exist_ok=True)
res = {}
if world.rank == 0:
    print(f"{a.kind}: stored label  E = {row.energy:.4f} eV  F_N = {np.round(ref, 3)}  |F| = {np.linalg.norm(ref):.2f}", flush=True)
    print(f"{'variant':10s} {'E (eV)':>12s} {'mu':>6s} {'F_N x':>8s} {'F_N y':>8s} {'F_N z':>8s} {'|dF_N| vs label':>16s} {'min':>6s}", flush=True)
for v in a.variants:
    t0 = time.time()
    at = at0.copy()
    mom = [R.RECIPE["init_magmom"]] * n + [3.0]
    ecut, spin = R.RECIPE["ecut"], True
    conv = {"energy": R.RECIPE["econv"], "density": R.RECIPE["dens_conv"]}
    if v == "tight":
        conv = {"energy": 1e-6, "density": 1e-6}
    elif v == "ecut600":
        ecut = 600.0
    elif v == "lowspin":
        mom = [R.RECIPE["init_magmom"]] * n + [0.3]
    elif v == "unpol":
        spin = False
    at.set_initial_magnetic_moments(mom if spin else [0.0] * len(at))
    at.calc = GPAW(mode=PW(ecut), xc=R.RECIPE["xc"], kpts={"size": R.kmesh(at, R.RECIPE["kdens"]), "gamma": True},
                   occupations=FermiDirac(R.RECIPE["smear"]), spinpol=spin, convergence=conv,
                   txt=str(out / f"{a.kind}_{v}.txt"))
    try:
        e = float(at.get_potential_energy())
        f = np.array(at.get_forces())[n]
        mu = float(at.get_magnetic_moment()) if spin else 0.0
        res[v] = dict(E=e, mu=mu, F_N=f.tolist())
        if world.rank == 0:
            print(f"{v:10s} {e:12.4f} {mu:6.2f} {f[0]:8.3f} {f[1]:8.3f} {f[2]:8.3f} {np.linalg.norm(f - ref):16.3f} {(time.time() - t0) / 60:6.1f}", flush=True)
    except Exception as exc:
        if world.rank == 0:
            print(f"{v:10s} FAILED: {type(exc).__name__}", flush=True)
if world.rank == 0:
    (out / f"{a.kind}.json").write_text(json.dumps(dict(kind=a.kind, label_E=row.energy, label_F_N=ref.tolist(), variants=res), indent=1))
    print(f"\nwritten {out / (a.kind + '.json')}")

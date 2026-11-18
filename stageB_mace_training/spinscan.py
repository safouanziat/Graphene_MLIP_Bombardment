"""Fixed-moment DFT scan of one stored frame: lowest-energy total moment = ground state.

    export OMP_NUM_THREADS=1
    mpirun -np 5 python spinscan.py --db data/nimpact_final_n7_fix/labelled.db --kind N_top_E30_r1_t280fs_pk --moments 1 3

M = 0 runs spin-unpolarised; M != 0 holds the total moment fixed (FermiDirac fixmagmom), starting with M on the N atom.
Works on labelled or pending databases (no stored label needed). Results: <out>/<kind>.json, finished moments are skipped.
--slow uses a small density-mixing step (same converged result, fewer SCF oscillations).
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
p.add_argument("--moments", type=float, nargs="*", default=[0, 1, 3])
p.add_argument("--maxiter", type=int, default=150)
p.add_argument("--slow", action="store_true")
p.add_argument("--out", default="spinscan")
a = p.parse_args()

rows = [r for r in connect(a.db).select() if r.kind == a.kind]
assert rows, f"no row with kind {a.kind} in {a.db}"
at0 = rows[0].toatoms()
n = len(at0) - 1
assert at0.get_chemical_symbols()[n] == "N"
out = Path(a.out)
out.mkdir(exist_ok=True)
jf = out / f"{a.kind}.json"
res = json.loads(jf.read_text()) if jf.exists() else {}
if world.rank == 0:
    print(f"{a.kind}  (stored results: {sorted(res)})", flush=True)
    print(f"{'M (muB)':>8s} {'E (eV)':>12s} {'mu out':>7s} {'F_N x':>8s} {'F_N y':>8s} {'F_N z':>8s} {'min':>6s}", flush=True)
for M in a.moments:
    key = f"{M:g}"
    if key in res:
        continue
    t0 = time.time()
    at = at0.copy()
    spin = M != 0
    mom = [0.0] * len(at)
    if spin:
        mom[n] = float(M)
    at.set_initial_magnetic_moments(mom)
    occ = FermiDirac(R.RECIPE["smear"], fixmagmom=True) if spin else FermiDirac(R.RECIPE["smear"])
    kw = dict(mode=PW(R.RECIPE["ecut"]), xc=R.RECIPE["xc"], kpts={"size": R.kmesh(at, R.RECIPE["kdens"]), "gamma": True},
              occupations=occ, spinpol=spin, maxiter=a.maxiter,
              convergence={"energy": R.RECIPE["econv"], "density": R.RECIPE["dens_conv"]},
              txt=str(out / f"{a.kind}_M{key}.txt"))
    if a.slow:
        from gpaw import Mixer
        kw["mixer"] = Mixer(0.05, 5, 50.0)
    at.calc = GPAW(**kw)
    try:
        e = float(at.get_potential_energy())
        f = np.array(at.get_forces())
        mu = float(at.get_magnetic_moment()) if spin else 0.0
        res[key] = dict(E=e, mu=mu, forces=f.tolist())
        if world.rank == 0:
            print(f"{M:8g} {e:12.4f} {mu:7.2f} {f[n][0]:8.3f} {f[n][1]:8.3f} {f[n][2]:8.3f} {(time.time() - t0) / 60:6.1f}", flush=True)
            jf.write_text(json.dumps(res))
    except Exception as exc:
        if world.rank == 0:
            print(f"{M:8g} FAILED: {type(exc).__name__}", flush=True)
if world.rank == 0 and res:
    emin = min(v["E"] for v in res.values())
    print("lowest:", min(res, key=lambda k: res[k]["E"]), "muB;  E - Emin (eV):",
          {k: round(v["E"] - emin, 3) for k, v in res.items()})

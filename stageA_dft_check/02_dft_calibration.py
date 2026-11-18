#!/usr/bin/env python
"""
DFT-recipe calibration for MLIP training data (cutoff and k-point density).

Takes a few structures (the DFT-relaxed 4x4 geometries of Stage A, plus rattled copies for off-equilibrium forces),
computes every one at six (cutoff, k-density) settings and compares each setting with the most accurate one (REF).

What matters for training is not the absolute energy but
  (a) energy differences between structures  -> 'formation-type' energy  E(struct) - N * E(pristine)/N_pristine
  (b) a constant energy offset per atom      -> std of (E - E_ref)/N over structures
  (c) forces                                 -> RMSE and max of F - F_ref
A setting is usable for ALL training data if  max|dEf| <= 0.02 eV,  std(offset) <= 3 meV/atom,  force RMSE <= 0.02 eV/A.

The k-mesh is set from a k-point DENSITY (points per 1/A of reciprocal length, z fixed to 1), so cells of different
size get an equivalent sampling.  Resumable (results.jsonl); run under MPI with plain python:

    export OMP_NUM_THREADS=1
    nohup nice -n 10 mpirun -np 5 python 02_dft_calibration.py > calib.log 2>&1 &
"""
import sys

if "--toy" not in sys.argv:        # GPAW must set up the MPI world before anything from ASE touches it
    from gpaw import GPAW, PW, FermiDirac  # noqa: F401

import argparse
import json
import time
from pathlib import Path

import numpy as np
from ase.io import read
from ase.parallel import parprint, world

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--src", default="dft_cheap/relaxed", help="folder with gpaw_<name>_n4.extxyz relaxed structures")
p.add_argument("--workdir", default="calib")
p.add_argument("--structures", nargs="+",
               default=["pristine", "pristine_rattled", "V1", "V1_rattled", "SW", "adatom"])
p.add_argument("--rattle", type=float, default=0.08, help="stdev of the random displacement of *_rattled (A)")
p.add_argument("--smear", type=float, default=0.05)
p.add_argument("--econv", type=float, default=1e-4, help="GPAW energy convergence, eV per electron")
p.add_argument("--toy", action="store_true", help="EMT instead of GPAW, only to test the workflow")
p.add_argument("--dry-run", action="store_true", help="list jobs and k-meshes, compute nothing")
args = p.parse_args()

# (name, plane-wave cutoff in eV, k-density).  Cheap -> expensive.  The last one is the reference.
SETTINGS = [("A_350_k2.5", 350, 2.5), ("B_350_k4", 350, 4.0), ("C_450_k4", 450, 4.0),
            ("D_450_k5", 450, 5.0), ("E_600_k4", 600, 4.0), ("REF_600_k5", 600, 5.0)]

WD = Path(args.workdir)
STORE = WD / "results.jsonl"
if world.rank == 0:
    for sub in ("logs", "forces"):
        (WD / sub).mkdir(parents=True, exist_ok=True)
world.barrier()


def kmesh(at, kd):
    """Gamma-centred mesh from a k-point density; the z direction (vacuum) always has 1 point."""
    rec = at.cell.reciprocal()
    n = [max(1, int(np.ceil(2 * np.pi * np.linalg.norm(r) * kd))) for r in rec]
    n[2] = 1
    return tuple(n)


def get_structure(name):
    base = name.replace("_rattled", "")
    f = Path(args.src) / f"gpaw_{base}_n4.extxyz"
    if not f.exists():
        raise SystemExit(f"missing {f}: run Stage A first or give --src")
    at = read(f)
    at.calc = None
    at.pbc = True
    if name.endswith("_rattled"):
        at.rattle(stdev=args.rattle, seed=7)
    return at


def load():
    rows = []
    if STORE.exists():
        for line in STORE.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except ValueError:
                pass
    return rows


def make_calc(ec, mesh, tag):
    if args.toy:
        from ase.calculators.emt import EMT
        return EMT()
    return GPAW(mode=PW(ec), xc="PBE", kpts={"size": mesh, "gamma": True}, occupations=FermiDirac(args.smear),
                spinpol=True, convergence={"energy": args.econv, "density": 1e-4}, txt=str(WD / "logs" / f"{tag}.txt"))


def run():
    done = {(r["setting"], r["struct"]) for r in load()}
    for sname, ec, kd in SETTINGS:
        for st in args.structures:
            at = get_structure(st)
            mesh = kmesh(at, kd)
            if args.dry_run:
                parprint(f"{sname:12s} {st:18s} {len(at):3d} atoms  cutoff {ec} eV  k-density {kd}  mesh {mesh}")
                continue
            if (sname, st) in done:
                continue
            at.set_initial_magnetic_moments([0.3] * len(at))
            at.calc = make_calc(ec, mesh, f"{sname}_{st}")
            t0 = time.time()
            e = at.get_potential_energy()
            f = at.get_forces()
            try:
                mu = float(at.get_magnetic_moment())
            except Exception:
                mu = 0.0
            mins = (time.time() - t0) / 60
            if world.rank == 0:
                np.save(WD / "forces" / f"{sname}_{st}.npy", f)
                with open(STORE, "a") as fh:
                    fh.write(json.dumps(dict(setting=sname, struct=st, ecut=ec, kd=kd, mesh=list(mesh), nat=len(at),
                                             E=float(e), mu=mu, minutes=round(mins, 2),
                                             fmax=float(np.abs(f).max()))) + "\n")
            world.barrier()
            parprint(f"[{sname}] {st:18s} mesh {mesh}  E = {e:.4f} eV  mu = {mu:.2f}  {mins:.1f} min", flush=True)


def summary():
    rows = load()
    R = {(r["setting"], r["struct"]): r for r in rows}
    names = [s[0] for s in SETTINGS]
    ref = names[-1]
    sts = [s for s in args.structures if all((n, s) in R for n in names)]
    if "pristine" not in sts or len(sts) < 2:
        parprint("not enough finished jobs for a summary yet")
        return

    def ef(sn, st):
        pr = R[(sn, "pristine")]
        r = R[(sn, st)]
        return r["E"] - r["nat"] * pr["E"] / pr["nat"]

    parprint("\nComparison with the reference setting %s" % ref)
    parprint(f"{'setting':12s} {'cost(min)':>9s} {'max|dEf| eV':>12s} {'offset std meV/at':>18s} "
             f"{'F rmse eV/A':>12s} {'F max':>7s} {'verdict':>8s}")
    best = None
    for sn, ec, kd in SETTINGS[:-1]:
        d_ef = [ef(sn, s) - ef(ref, s) for s in sts if s != "pristine"]
        off = [(R[(sn, s)]["E"] - R[(ref, s)]["E"]) / R[(sn, s)]["nat"] * 1000 for s in sts]
        dF = np.concatenate([(np.load(WD / "forces" / f"{sn}_{s}.npy") -
                              np.load(WD / "forces" / f"{ref}_{s}.npy")).ravel() for s in sts])
        cost = sum(R[(sn, s)]["minutes"] for s in sts)
        mx, sd, rm, fm = max(abs(x) for x in d_ef), float(np.std(off)), float(np.sqrt(np.mean(dF ** 2))), float(np.abs(dF).max())
        ok = mx <= 0.02 and sd <= 3.0 and rm <= 0.02
        parprint(f"{sn:12s} {cost:9.1f} {mx:12.4f} {sd:18.2f} {rm:12.4f} {fm:7.3f} {'PASS' if ok else 'fail':>8s}")
        if ok and (best is None or cost < best[1]):
            best = (sn, cost)
    mus = {s: [R[(n, s)]["mu"] for n in names] for s in sts}
    flips = [s for s, m in mus.items() if max(m) - min(m) > 0.1]
    parprint("\nmagnetic moment identical (within 0.1 muB) in all settings: " + ("yes" if not flips else "NO for " + ", ".join(flips)))
    parprint("cheapest passing setting: " + (f"{best[0]}" if best else "none: use the most expensive tested setting"))


if __name__ == "__main__":
    run()
    if not args.dry_run:
        summary()
"""Feasibility of MACE impacts with ASE dynamics on this CPU: time per step and energy conservation.

An Ar atom (E_kin eV, over a hexagon centre, 6 A above the sheet) falls on a flat n x n graphene sheet at rest; velocity Verlet
(NVE) for --steps steps.  Prints seconds per step (first call excluded) and the drift of the total energy, for each combination of
sheet size, time step and precision.

    python md_timing.py --model data/ft/omat-0_all_ar/omat-0_all_ar.model --n 6 8 --dt 0.1 0.2 --steps 200 --dtype float32
MLIP_TOY=1 python md_timing.py --n 3 --dt 0.5 --steps 20 --proj C     # workflow test with EMT
"""
import argparse
import os
import time

import numpy as np
from ase import Atoms, units
from ase.build import graphene
from ase.data import atomic_masses, atomic_numbers
from ase.md.verlet import VelocityVerlet

import recipe as R

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--model", default="")
p.add_argument("--n", nargs="+", type=int, default=[6, 8])
p.add_argument("--dt", nargs="+", type=float, default=[0.1, 0.2], help="fs")
p.add_argument("--steps", type=int, default=200)
p.add_argument("--dtype", nargs="+", default=["float32"])
p.add_argument("--ekin", type=float, default=70.0)
p.add_argument("--proj", default="Ar")
p.add_argument("--threads", type=int, default=8)
p.add_argument("--height", type=float, default=6.0)
args = p.parse_args()

if R.TOY:
    from ase.calculators.emt import EMT
else:
    import torch
    torch.set_num_threads(args.threads)
    from mace.calculators import MACECalculator


def build(n):
    sh = graphene(a=2.46, size=(n, n, 1), vacuum=10.0)
    sh.pbc = True
    centre = np.array([sh.cell[0, 0] / 2 + sh.cell[1, 0] / 2, sh.cell[1, 1] / 2])
    i = int(np.argmin(np.linalg.norm(sh.positions[:, :2] - centre, axis=1)))
    d = sh.get_distances(i, range(len(sh)), mic=True)
    d[i] = 9e9
    nb = [int(k) for k in np.argsort(d)[:2]]
    xy = sh.positions[i, :2] + sum(sh.get_distance(i, k, mic=True, vector=True) for k in nb)[:2]
    at = sh.copy()
    at += Atoms(args.proj, positions=[[xy[0], xy[1], sh.positions[:, 2].mean() + args.height]])
    v = np.zeros((len(at), 3))
    m = atomic_masses[atomic_numbers[args.proj]]
    v[-1, 2] = -np.sqrt(2 * args.ekin / m) / units.fs * 1.0     # A/fs -> ASE velocity units handled below
    at.set_velocities(v * units.fs)                              # ASE velocity unit = A / (fs * sqrt(...)): convert via units.fs
    return at


print(f"{'atoms':>6s} {'dt fs':>6s} {'dtype':>8s} {'s/step':>8s} {'max|dE| eV':>11s} {'dE/E_k %':>9s} {'ms/atom/step':>13s}  {'1 ps cost h':>11s}")
for dtype in args.dtype:
    calc = EMT() if R.TOY else MACECalculator(model_paths=os.path.expanduser(args.model), device="cpu", default_dtype=dtype)
    for n in args.n:
        for dt in args.dt:
            at = build(n)
            at.calc = calc
            dyn = VelocityVerlet(at, timestep=dt * units.fs)
            e0 = at.get_potential_energy() + at.get_kinetic_energy()
            at.get_forces()
            t0 = time.time()
            dyn.run(1)                                           # warm-up step
            t1 = time.time()
            es = []
            for _ in range(args.steps):
                dyn.run(1)
                es.append(at.get_potential_energy() + at.get_kinetic_energy())
            sps = (time.time() - t1) / args.steps
            drift = float(np.max(np.abs(np.array(es) - e0)))
            print(f"{len(at):6d} {dt:6.2f} {dtype:>8s} {sps:8.2f} {drift:11.4f} {100 * drift / args.ekin:9.2f} "
                  f"{1000 * sps / len(at):13.1f}  {1000 / dt * sps / 3600:11.1f}")

"""Threshold displacement energy of ONE carbon atom in graphene, per potential (no Ar involved).

A flat n x n sheet at 0 K; the atom nearest the cell centre gets kinetic energy T straight DOWN through the sheet (perpendicular to the
plane, the usual 'Td' geometry).  NVE for --fs; the atom counts as sputtered (a vacancy is left) if it ends > 3 A from the sheet plane
with at most one C neighbour.  Scan T upward: Td = lowest T that sputters.  Literature DFT-MD values for graphene are ~22 eV.

    python -W ignore pka_threshold.py airebo --airebo ../stageA_dft_check/CH.airebo --energies 14 16 18 20 22 24 26 28 30 32 34 36
    python -W ignore pka_threshold.py mace --model data/ft/omat-0_all_ar/omat-0_all_ar.model --energies 20 24 28 32 36 --threads 8
MLIP_TOY=1 python pka_threshold.py mace --energies 10 20 --fs 40 --n 3        # EMT workflow test
"""
import argparse
import json
import os
import shlex
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
from ase import units
from ase.build import graphene
from ase.io import read, write
from ase.neighborlist import neighbor_list

TOY = os.environ.get("MLIP_TOY") == "1"
p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
sp = p.add_subparsers(dest="cmd", required=True)
c = argparse.ArgumentParser(add_help=False)
c.add_argument("--n", type=int, default=5)
c.add_argument("--energies", nargs="+", type=float, required=True)
c.add_argument("--fs", type=float, default=400.0)
c.add_argument("--dt", type=float, default=None)
c.add_argument("--dir", default="down", choices=["down"])
a1 = sp.add_parser("airebo", parents=[c])
a1.add_argument("--airebo", required=True)
a1.add_argument("--lmp", default=os.environ.get("LMP_CMD", "lmp"))
a1.add_argument("--a0", type=float, default=2.419)
a2 = sp.add_parser("mace", parents=[c])
a2.add_argument("--model", default="")
a2.add_argument("--threads", type=int, default=8)
a2.add_argument("--a0", type=float, default=None, help="default: the MACE a0 fitted by stageC_pilot.py (data/stageC/mace_thermal/a0.json)")
args = p.parse_args()

m_c = 12.011


def sheet(a0):
    sh = graphene(a=a0, size=(args.n, args.n, 1), vacuum=10.0)
    sh.pbc = (True, True, False)
    cen = (sh.cell[0, :2] + sh.cell[1, :2]) / 2
    i = int(np.argmin(np.linalg.norm(sh.positions[:, :2] - cen, axis=1)))
    return sh, i


def sputtered(final, i):
    z0 = float(np.median(final.positions[:, 2]))
    dz = final.positions[i, 2] - z0
    ii, jj = neighbor_list("ij", final, 1.85)
    coord = int((ii == i).sum())
    return (abs(dz) > 3.0 and coord <= 1), dz, coord


def run_airebo(T):
    dt = args.dt or 0.1
    sh, i = sheet(args.a0)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        write(td / "s.data", sh, format="lammps-data", specorder=["C"], atom_style="atomic")
        vz = -np.sqrt(2 * T / m_c) * units.fs * 1000.0
        (td / "in").write_text("\n".join([
            "units metal", "boundary p p m", "atom_style atomic", "read_data s.data", f"mass 1 {m_c}",
            "pair_style airebo 3.0", f"pair_coeff * * {os.path.abspath(args.airebo)} C",
            "neighbor 2.0 bin", "min_style cg", "minimize 1e-10 1e-12 2000 20000",
            f"group pka id {i + 1}", f"velocity pka set 0 0 {vz:.8f} units box",
            f"timestep {dt * 1e-3}", "fix i all nve", f"run {int(round(args.fs / dt))}",
            "write_dump all custom final.dump id type x y z modify sort id"]))
        subprocess.run(shlex.split(args.lmp) + ["-in", "in", "-log", "none", "-screen", "none"], cwd=td, check=True)
        fin = read(td / "final.dump", format="lammps-dump-text", index=-1, specorder=["C"])
    return sputtered(fin, i)


calc = None


def run_mace(T):
    global calc
    from ase.md.verlet import VelocityVerlet
    from ase.optimize import FIRE
    dt = args.dt or 0.2
    if calc is None:
        if TOY:
            from ase.calculators.emt import EMT
            calc = EMT()
        else:
            import torch
            torch.set_num_threads(args.threads)
            from mace.calculators import MACECalculator
            calc = MACECalculator(model_paths=os.path.expanduser(args.model), device="cpu", default_dtype="float32")
    a0 = args.a0
    if a0 is None:
        f = Path("data/stageC/mace_thermal/a0.json")
        a0 = json.loads(f.read_text())["a0"] if f.exists() else 2.47
    sh, i = sheet(a0)
    sh.calc = calc
    FIRE(sh, logfile=None).run(fmax=0.01, steps=200)
    v = np.zeros((len(sh), 3))
    v[i, 2] = -np.sqrt(2 * T / m_c)                    # sqrt(eV/amu) is the ASE velocity unit
    sh.set_velocities(v)
    dyn = VelocityVerlet(sh, timestep=dt * units.fs)
    dyn.run(int(round(args.fs / dt)))
    return sputtered(sh, i)


run = run_airebo if args.cmd == "airebo" else run_mace
print(f"{args.cmd}: {args.n}x{args.n} sheet ({2 * args.n ** 2} atoms), kick perpendicular to the plane, {args.fs:g} fs")
print(f"{'T (eV)':>7s} {'sputtered':>10s} {'final dz (A)':>13s} {'C neighbours':>13s} {'min':>6s}")
first = None
for T in args.energies:
    t0 = time.time()
    out, dz, coord = run(T)
    first = T if (out and first is None) else first
    print(f"{T:7.1f} {str(out):>10s} {dz:13.2f} {coord:13d} {(time.time() - t0) / 60:6.1f}", flush=True)
print(f"\nlowest sputtering energy on this grid: {first if first is not None else 'none'} eV")

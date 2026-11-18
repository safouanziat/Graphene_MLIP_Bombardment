#!/usr/bin/env python3
"""Stage A: does AIREBO get the energetics of the observed graphene defects right?  (AIREBO vs DFT, same structures)

For each defect type seen in the bombardment runs (single vacancy, di-vacancy, Frenkel pair, Stone-Wales 5-7, adatom)
the script builds an n x n periodic graphene cell, relaxes it with AIREBO and with GPAW (PBE), and compares formation
energies, local coordination and the DFT magnetic moment.

    # 1. look at the structures, no calculation
    python 12_dft_defect_check.py --dry-run
    # 2. time one DFT single point and get an estimate of the total cost
    mpirun -np 8 gpaw python 12_dft_defect_check.py --bench
    # 3. the real run (resumable: finished jobs are skipped, relaxations restart from their last frame)
    nohup mpirun -np 8 gpaw python 12_dft_defect_check.py > dft_check.log 2>&1 &
    # 4. AIREBO only (seconds)
    python 12_dft_defect_check.py --methods airebo

Everything goes to --workdir (default dft_check/): results.db (ASE database, one row per job), results.csv,
traj/ (relaxation trajectories), relaxed/ (final structures, extended xyz), logs/ (GPAW text output).

Formation energy of a defect with N atoms:  E_f = E(defect) - N * E(pristine) / N_pristine,  each method at its own
equilibrium lattice constant (found by a short scan on the 2-atom cell), same k-mesh for pristine and defect cell.
"""
import argparse
import os
import sys
import time

if "gpaw" in sys.argv:        # GPAW must set up the MPI world before anything from ASE touches it
    from gpaw import GPAW, PW, FermiDirac  # noqa: F401
from pathlib import Path

import json
from types import SimpleNamespace

import numpy as np
from ase import Atoms
from ase.build import graphene
from ase.db import connect
from ase.io import write
from ase.neighborlist import neighbor_list
from ase.optimize import BFGS
from ase.parallel import world

_print = print


def print(*a, **k):                                   # under mpirun only rank 0 talks
    if world.rank == 0:
        _print(*a, **k)

DEFECTS = ["V1", "V2", "SW", "adatom", "frenkel"]
EXTRA = ["V1sym", "V2sym"]            # same defects without the pre-bonded start (to see whether a method reconstructs on its own)
DESCRIPTION = {
    "V1": "single vacancy, start with one dangling pair bonded (5-9)",
    "V2": "di-vacancy, start with the two dangling pairs bonded (5-8-5)",
    "V1sym": "single vacancy, symmetric start",
    "V2sym": "di-vacancy, symmetric start",
    "SW": "Stone-Wales (5-7-7-5) bond rotation",
    "adatom": "C adatom on a bridge site",
    "frenkel": "vacancy + bridge adatom ~4 A away",
}

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--methods", nargs="+", default=["airebo", "gpaw"], choices=["airebo", "gpaw", "emt"],
               help="emt is a toy calculator used only to test the workflow without GPAW")
p.add_argument("--defects", nargs="+", default=DEFECTS, choices=DEFECTS + EXTRA)
p.add_argument("--n", type=int, default=5, help="supercell n x n (2 n^2 atoms)")
p.add_argument("--vacuum", type=float, default=15.0, help="cell height (A)")
p.add_argument("--fmax", type=float, default=0.05, help="relaxation threshold (eV/A)")
p.add_argument("--steps", type=int, default=300, help="maximum relaxation steps")
p.add_argument("--cutoff", type=float, default=450.0, help="GPAW plane-wave cutoff (eV)")
p.add_argument("--kpts", type=int, nargs=3, default=[3, 3, 1], help="GPAW k-mesh of the supercell (Gamma-centred)")
p.add_argument("--smear", type=float, default=0.05, help="Fermi-Dirac width (eV)")
p.add_argument("--econv", type=float, default=1e-5,
               help="GPAW energy convergence, eV per electron (GPAW default 5e-4; 1e-4 is robust for spin-polarised vacancies)")
p.add_argument("--mixer", type=float, nargs=3, metavar=("BETA", "NOLD", "WEIGHT"), default=None,
               help="optional GPAW density mixer, e.g. --mixer 0.05 5 50 (default: GPAW's own)")
p.add_argument("--airebo", default="CH.airebo", help="AIREBO potential file (executable: env LMP_CMD, default lmp)")
p.add_argument("--workdir", default="dft_check")
p.add_argument("--seed", type=int, default=2026, help="seed of the small initial rattle that breaks symmetry")
p.add_argument("--dry-run", action="store_true", help="only build and write the structures")
p.add_argument("--bench", action="store_true", help="time one GPAW single point on the single vacancy and stop")
p.add_argument("--redo", action="store_true", help="ignore stored results")
args = p.parse_args()

WD = Path(args.workdir)
for sub in ("traj", "relaxed", "logs", "start"):
    (WD / sub).mkdir(parents=True, exist_ok=True)
STORE = WD / "results.jsonl"      # control state: read by every rank, appended by rank 0 only
DBPATH = str(WD / "results.db")   # ASE database for later use (ase db results.db), written by rank 0 only


def save(kv):
    if world.rank == 0:
        with open(STORE, "a") as f:
            f.write(json.dumps(kv) + "\n")
        connect(DBPATH, serial=True).write(Atoms(), **kv)
    world.barrier()


def load(**match):
    rows = []
    if STORE.exists():
        for line in STORE.read_text().splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if all(r.get(k) == v for k, v in match.items()):
                rows.append(SimpleNamespace(**r))
    return rows


# ----------------------------------------------------------------------------- calculators
from ase.calculators.calculator import Calculator, all_changes  # noqa: E402


class AireboLMP(Calculator):
    """AIREBO through the `lmp` executable (one `run 0` per call; robust across LAMMPS versions)."""
    implemented_properties = ["energy", "forces"]

    def __init__(self, potential, workdir, lmp=None, **kw):
        super().__init__(**kw)
        self.potential = os.path.abspath(potential)
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.lmp = (lmp or os.environ.get("LMP_CMD", "lmp")).split()

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        import subprocess
        from ase.io.lammpsdata import write_lammps_data
        super().calculate(atoms, properties, system_changes)
        d = self.workdir
        with open(d / "data.lmp", "w") as f:
            write_lammps_data(f, atoms, specorder=["C"], masses=True)
        (d / "in.lmp").write_text(
            "units metal\natom_style atomic\nboundary p p p\nread_data data.lmp\n"
            f"pair_style airebo 3.0 1 1\npair_coeff * * {self.potential} C\n"
            "dump d all custom 1 forces.dump id fx fy fz\ndump_modify d sort id format float %.10g\n"
            "thermo_style custom step pe\nvariable e equal pe\nrun 0\n"
            "print \"ENERGY ${e}\" file energy.txt screen no\n")
        subprocess.run(self.lmp + ["-in", "in.lmp", "-log", "none", "-screen", "none"], cwd=d, check=True)
        energy = float((d / "energy.txt").read_text().split()[1])
        lines = (d / "forces.dump").read_text().splitlines()
        i0 = next(i for i, l in enumerate(lines) if l.startswith("ITEM: ATOMS")) + 1
        forces = np.array([[float(x) for x in l.split()[1:4]] for l in lines[i0:i0 + len(atoms)]])
        self.results = {"energy": energy, "forces": forces}


def make_calc(method, tag, kpts=None):
    if method == "airebo":
        if world.size > 1:
            raise SystemExit("run the airebo part WITHOUT mpirun first: python 01_dft_defect_check.py --methods airebo ...")
        return AireboLMP(args.airebo, WD / "logs" / f"lmp_{tag}")
    if method == "emt":
        from ase.calculators.emt import EMT
        return EMT()
    from gpaw import GPAW, PW, FermiDirac
    extra = {}
    if args.mixer:
        from gpaw import Mixer
        extra["mixer"] = Mixer(args.mixer[0], int(args.mixer[1]), args.mixer[2])
    return GPAW(mode=PW(args.cutoff), xc="PBE", kpts={"size": tuple(kpts or args.kpts), "gamma": True},
                occupations=FermiDirac(args.smear), spinpol=True,
                convergence={"energy": args.econv, "density": 1e-4}, txt=str(WD / "logs" / f"{tag}.txt"), **extra)


# ----------------------------------------------------------------------------- structures
def pristine(a0, n):
    at = graphene(formula="C2", a=a0, size=(n, n, 1), vacuum=args.vacuum / 2)
    at.pbc = True
    return at


def center_index(at):
    c = 0.5 * (at.cell[0] + at.cell[1])
    d = np.linalg.norm(at.positions[:, :2] - c[:2], axis=1)
    return int(np.argmin(d))


def neighbours(at, i, cut=1.7):
    ii, jj = neighbor_list("ij", at, cut)
    return [int(j) for a, j in zip(ii, jj) if a == i]


def bridge_pos(at, i, j, height=1.7):
    mid = at.positions[i] + 0.5 * at.get_distance(i, j, mic=True, vector=True)
    return mid + np.array([0.0, 0.0, height])


def prebond(at, i, j, bond=1.7):
    """Move two dangling atoms towards each other until they are `bond` apart (starting geometry of the reconstruction)."""
    v = at.get_distance(i, j, mic=True, vector=True)
    d = np.linalg.norm(v)
    shift = 0.5 * (d - bond) * v / d
    at.positions[i] += shift
    at.positions[j] -= shift


def build(defect, a0, n, seed):
    at = pristine(a0, n)
    i0 = center_index(at)
    j0 = neighbours(at, i0)[0]
    rng = np.random.default_rng(seed)
    if defect == "pristine":
        return at
    if defect in ("V1", "V1sym"):
        if defect == "V1":
            prebond(at, *neighbours(at, i0)[:2])
        del at[i0]
    elif defect in ("V2", "V2sym"):
        if defect == "V2":
            a = [k for k in neighbours(at, i0) if k != j0]
            b = [k for k in neighbours(at, j0) if k != i0]
            prebond(at, a[0], a[1])
            prebond(at, b[0], b[1])
        for k in sorted([i0, j0], reverse=True):
            del at[k]
    elif defect == "SW":
        mid = at.positions[i0] + 0.5 * at.get_distance(i0, j0, mic=True, vector=True)
        for k in (i0, j0):
            v = at.positions[k] - mid
            at.positions[k] = mid + np.array([-v[1], v[0], v[2]])      # 90 degree rotation in the plane
    elif defect == "adatom":
        at += Atoms("C", positions=[bridge_pos(at, i0, j0)])
    elif defect == "frenkel":
        centre = at.positions[i0].copy()
        ii, jj = neighbor_list("ij", at, 1.7)
        best, bestd = None, 1e9
        for a, b in zip(ii, jj):
            if a < b and a != i0 and b != i0:
                m = at.positions[a] + 0.5 * at.get_distance(a, b, mic=True, vector=True)
                d = abs(np.linalg.norm((m - centre)[:2]) - 4.3)
                if d < bestd:
                    best, bestd = (int(a), int(b)), d
        ad = bridge_pos(at, *best)
        del at[i0]
        at += Atoms("C", positions=[ad])
    at.positions += rng.normal(0.0, 0.03, at.positions.shape)         # break symmetry (Jahn-Teller etc.)
    return at


def coordination(at, cut=1.85):
    ii, _ = neighbor_list("ij", at, cut)
    cn = np.bincount(ii, minlength=len(at))
    return {int(k): int((cn == k).sum()) for k in np.unique(cn)}


# ----------------------------------------------------------------------------- jobs
def stored(method, defect):
    rows = load(method=method, defect=defect, n=args.n)
    return rows[-1] if rows and not args.redo else None


def lattice_constant(method):
    row = stored(method, "lattice")
    if row is not None:
        return float(row.a0)
    xs = np.linspace(2.38, 2.52, 7)
    es = []
    for a in xs:
        at = graphene(formula="C2", a=a, size=(1, 1, 1), vacuum=args.vacuum / 2)
        at.pbc = True
        at.calc = make_calc(method, f"lattice_{method}_{a:.3f}", kpts=[12, 12, 1])
        if method == "gpaw":
            at.set_initial_magnetic_moments([0.0, 0.0])
        es.append(at.get_potential_energy())
    c = np.polyfit(xs, es, 2)
    a0 = float(-c[1] / (2 * c[0]))
    save(dict(method=method, defect="lattice", n=args.n, a0=a0))
    print(f"[{method}] equilibrium lattice constant a0 = {a0:.4f} A", flush=True)
    return a0


def run_job(method, defect, a0):
    tag = f"{method}_{defect}_n{args.n}"
    row = stored(method, defect)
    if row is not None:
        return row
    at = build(defect, a0, args.n, args.seed)
    write(WD / "start" / f"{tag}.extxyz", at)
    t0 = time.time()
    trajf = WD / "traj" / f"{tag}.traj"
    if trajf.exists() and trajf.stat().st_size > 0:                       # restart from the last frame
        from ase.io import read
        try:
            at = read(trajf, index=-1)
            print(f"[{tag}] restarting from the last relaxation frame", flush=True)
        except Exception:
            pass
    at.pbc = True
    at.calc = make_calc(method, tag)
    if method == "gpaw":
        at.set_initial_magnetic_moments([0.3] * len(at))
    if defect == "pristine":
        e = at.get_potential_energy()
        nsteps, fmax_end = 0, 0.0
    else:
        opt = BFGS(at, trajectory=str(trajf), logfile=str(WD / "logs" / f"{tag}.relax.log"), append_trajectory=True)
        opt.run(fmax=args.fmax, steps=args.steps)
        e = at.get_potential_energy()
        nsteps = opt.get_number_of_steps()
        fmax_end = float(np.abs(at.get_forces()).max())
    mag = float(at.get_magnetic_moment()) if method == "gpaw" else 0.0
    conv = fmax_end <= args.fmax or defect == "pristine"
    write(WD / "relaxed" / f"{tag}.extxyz", at)
    cn = coordination(at)
    mins = (time.time() - t0) / 60
    kv = dict(method=method, defect=defect, n=args.n, a0=a0, etot=float(e), nat=len(at), mu=mag,
              converged=bool(conv), nsteps=int(nsteps), minutes=round(mins, 1),
              cn2=cn.get(2, 0), cn3=cn.get(3, 0), cn4=cn.get(4, 0))
    save(kv)
    print(f"[{tag}] E = {e:.3f} eV, {nsteps} steps, {mins:.1f} min, converged={conv}, mu = {mag:.2f}", flush=True)
    return stored(method, defect)


def main():
    if args.dry_run:
        for d in args.defects:
            at = build(d, 2.46, args.n, args.seed)
            write(WD / "start" / f"{d}_n{args.n}.extxyz", at)
            print(f"{d:8s} {len(at):3d} atoms  coordination {coordination(at)}   {DESCRIPTION[d]}")
        print(f"structures written to {WD / 'start'}  (open with: ase gui {WD}/start/V1_n{args.n}.extxyz)")
        return
    if args.bench:
        a0 = 2.466
        at = build("V1", a0, args.n, args.seed)
        at.calc = make_calc("gpaw", "bench")
        at.set_initial_magnetic_moments([0.3] * len(at))
        t0 = time.time()
        e = at.get_potential_energy()
        f = at.get_forces()
        dt = time.time() - t0
        print(f"single vacancy, {len(at)} atoms: first SCF + forces {dt/60:.1f} min  (E = {e:.3f} eV)")
        print("relaxations typically need 20-60 force evaluations, later ones are 3-5 times cheaper than the first;")
        print(f"estimate per relaxed defect: {20*dt/3/3600:.1f}-{60*dt/3/3600:.1f} h  ({len(args.defects)} defects: "
              f"{len(args.defects)*20*dt/3/3600:.0f}-{len(args.defects)*60*dt/3/3600:.0f} h)")
        return
    results = {}
    for method in args.methods:
        a0 = lattice_constant(method)
        pr = run_job(method, "pristine", a0)
        e_atom = pr.etot / pr.nat
        for d in args.defects:
            r = run_job(method, d, a0)
            results[(method, d)] = (r.etot - r.nat * e_atom, r)
    # table
    lines = ["defect,method,natoms,E_formation_eV,magmom_muB,cn2,cn3,cn4,steps,converged,minutes"]
    print("\nFormation energies (eV)")
    head = f"{'defect':9s}" + "".join(f"{m:>10s}" for m in args.methods) + ("      diff" if len(args.methods) > 1 else "")
    print(head)
    for d in args.defects:
        vals = [results[(m, d)][0] for m in args.methods]
        extra = f"{vals[-1] - vals[0]:10.2f}" if len(vals) > 1 else ""
        print(f"{d:9s}" + "".join(f"{v:10.2f}" for v in vals) + extra)
        for m in args.methods:
            ef, r = results[(m, d)]
            lines.append(f"{d},{m},{r.nat},{ef:.4f},{r.mu:.2f},{r.cn2},{r.cn3},{r.cn4},{r.nsteps},{r.converged},{r.minutes}")
    if world.rank == 0:
        (WD / "results.csv").write_text("\n".join(lines) + "\n")
    print(f"\nwrote {WD / 'results.csv'}")
    print("Reference ranges from the literature (PBE): V1 7.4-7.9 eV, V2 7.5-8.0 eV, Stone-Wales 4.5-5.2 eV, bridge adatom 6-7 eV.")
    print("Check the magnetic moment of V1 (about 1-1.5 muB) and that every job converged.")


if __name__ == "__main__":
    main()
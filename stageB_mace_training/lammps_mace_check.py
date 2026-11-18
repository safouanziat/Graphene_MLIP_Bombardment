"""Does this LAMMPS run our MACE model (pair_style mliap unified), and how fast?

  1. converts the model to the ML-IAP format      python -m mace.cli.create_lammps_model <model> --format mliap   (needs cuequivariance in this MACE version)
  2. evaluates one impact snapshot with ASE (MACECalculator) and with LAMMPS (run 0) and compares energy and forces
  3. times --steps NVE steps in LAMMPS (tiny time step, the Ar barely moves: this is a speed test only)

    python lammps_mace_check.py --model data/ft/omat-0_all_ar/omat-0_all_ar.model
    python lammps_mace_check.py --model ... --dtype float32 --threads 8 --pt data/lammps_test/x.pt   # reuse an existing converted file
Everything is written to data/lammps_test/.  Each failing stage prints what failed and stops.
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from ase.io import read, write

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--model", required=True)
p.add_argument("--pt", default="", help="already converted file (skips the conversion)")
p.add_argument("--snap", default="Ar_hollow_E100_t24fs", help="info['name'] of the impact snapshot in data/impacts/selected.extxyz")
p.add_argument("--dtype", default="float64", choices=["float32", "float64"])
p.add_argument("--steps", type=int, default=50)
p.add_argument("--threads", type=int, default=8)
p.add_argument("--lmp", default=os.environ.get("LMP_CMD", "lmp"))
args = p.parse_args()
W = Path("data/lammps_test")
W.mkdir(parents=True, exist_ok=True)
env = dict(os.environ, OMP_NUM_THREADS=str(args.threads), MACE_ALLOW_CPU="true")

# 1. conversion
pt = Path(args.pt) if args.pt else None
if pt is None:
    local = W / Path(args.model).name
    shutil.copy(args.model, local)
    r = subprocess.run([sys.executable, "-m", "mace.cli.create_lammps_model", str(local), "--format", "mliap", "--dtype", args.dtype],
                       capture_output=True, text=True, env=env)
    pt = Path(str(local) + "-mliap_lammps.pt")
    if r.returncode != 0 or not pt.exists():
        print("STAGE 1 FAILED: conversion to the ML-IAP format\n", (r.stderr or r.stdout)[-1500:])
        raise SystemExit(1)
print(f"1. converted model: {pt} ({pt.stat().st_size / 1e6:.1f} MB)")

# 2. one snapshot, ASE versus LAMMPS
geoms = read("data/impacts/selected.extxyz", ":")
at = next((g for g in geoms if g.info.get("name") == args.snap), geoms[0])
print(f"   snapshot: {at.info.get('name')}  ({len(at)} atoms)")
import torch  # noqa: E402
torch.set_num_threads(args.threads)
from mace.calculators import MACECalculator  # noqa: E402
calc = MACECalculator(model_paths=os.path.expanduser(args.model), device="cpu", default_dtype=args.dtype)
a = at.copy()
a.calc = calc
e_ase, f_ase = float(a.get_potential_energy()), np.array(a.get_forces())

syms = ["C", "Ar"]
write(W / "system.data", at, format="lammps-data", specorder=syms, atom_style="atomic")
(W / "in.check").write_text("\n".join([
    "units metal", "boundary p p p", "atom_style atomic", "read_data system.data",
    "mass 1 12.011", "mass 2 39.948",
    f"pair_style mliap unified {pt.resolve()} 0", "pair_coeff * * C Ar",
    "neighbor 1.0 bin", "thermo_style custom step pe ke", "thermo 1",
    "dump d all custom 1 f.dump id type fx fy fz", "dump_modify d sort id", "run 0", "undump d",
    "timestep 0.0001", "velocity all set 0.0 0.0 0.0", "fix nve all nve", f"run {args.steps}", ""]))
r = subprocess.run(args.lmp.split() + ["-in", "in.check", "-log", "log.lammps", "-screen", "none"], cwd=W,
                   capture_output=True, text=True, env=env)
log = (W / "log.lammps").read_text() if (W / "log.lammps").exists() else ""
if r.returncode != 0 or "Loop time" not in log:
    print("STAGE 2 FAILED: LAMMPS did not run the model\n", (r.stderr or "")[-1200:], "\n", log[-1500:])
    raise SystemExit(1)
pe = float(re.search(r"^\s*0\s+(-?[0-9.eE+-]+)\s+[0-9.eE+-]+\s*$", log, re.M).group(1))
dump = (W / "f.dump").read_text().split("ITEM: ATOMS id type fx fy fz\n")[1].strip().splitlines()
f_lmp = np.array([[float(x) for x in l.split()[2:5]] for l in dump])
print(f"2. energy ASE {e_ase:.5f} eV   LAMMPS {pe:.5f} eV   diff {pe - e_ase:+.2e}")
print(f"   forces: max |F_LAMMPS - F_ASE| = {np.abs(f_lmp - f_ase).max():.2e} eV/A   (DFT-scale forces here are ~1-30 eV/A)")
# 3. speed
t = re.findall(r"Loop time of ([0-9.]+) on (\d+) procs for (\d+) steps", log)
loop, procs, n = float(t[-1][0]), t[-1][1], int(t[-1][2])
print(f"3. LAMMPS: {loop / n:.3f} s/step for {len(at)} atoms ({n} steps, {args.threads} OpenMP threads, {args.dtype}); 1 ps at 0.2 fs = {5000 * loop / n / 3600:.1f} h")

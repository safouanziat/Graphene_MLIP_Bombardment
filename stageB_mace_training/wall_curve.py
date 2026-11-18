"""Where do AIREBO+ZBL and MACE part ways on the Ar-C repulsive wall?

Static Ar approach straight above one C atom ('top') of the DFT scan sheet (frozen sheet): interaction energy E(d) - E(d_ref) for
AIREBO+ZBL, for each MACE model, and for the DFT scan points (data/arscan/labelled.db, flat top scans, d >= 1.65 A).
Below 1.65 A there is no DFT reference (PAW cores overlap), so the table shows how far each model has been extrapolated.
Also prints the distance at which the repulsive energy reaches the impact energies (the classical turning point on a frozen atom).

    python -W ignore wall_curve.py --airebo ../stageA_dft_check/CH.airebo --models data/ft/omat-0_all_ar/omat-0_all_ar.model \
        --db data/arscan/labelled.db
"""
import argparse
import os
import shlex
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.db import connect
from ase.io import write

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--airebo", required=True)
p.add_argument("--models", nargs="*", default=[])
p.add_argument("--db", default="data/arscan/labelled.db")
p.add_argument("--arc-table", default="", help="also evaluate AIREBO + this DFT-fitted Ar-C table")
p.add_argument("--lmp", default=os.environ.get("LMP_CMD", "lmp"))
p.add_argument("--dist", nargs="+", type=float, default=[0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.65, 1.8, 1.95, 2.1, 2.3, 2.6, 3.0, 3.5, 4.5])
p.add_argument("--ref", type=float, default=4.5, help="reference distance for the interaction energy")
p.add_argument("--energies", nargs="+", type=float, default=[40, 50, 55, 60, 70])
p.add_argument("--threads", type=int, default=8)
args = p.parse_args()


def airebo_energy(at, table=""):
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        write(td / "s.data", at, format="lammps-data", specorder=["C", "Ar"], atom_style="atomic")
        (td / "in").write_text("\n".join([
            "units metal", "boundary p p p", "atom_style atomic", "read_data s.data",
            "mass 1 12.011", "mass 2 39.948",
            *(["pair_style hybrid airebo 3.0 zbl 4.0 5.0 table linear 5000", f"pair_coeff * * airebo {os.path.abspath(args.airebo)} C NULL",
               "pair_coeff 2 2 zbl 18 18", f"pair_coeff 1 2 table {os.path.abspath(table)} ARC"] if table else
              ["pair_style hybrid airebo 3.0 zbl 4.0 5.0", f"pair_coeff * * airebo {os.path.abspath(args.airebo)} C NULL",
               "pair_coeff 2 2 zbl 18 18", "pair_coeff 1 2 zbl 6.0 18"]),
            "thermo_style custom step pe", "run 0"]))
        r = subprocess.run(shlex.split(args.lmp) + ["-in", "in", "-log", "none", "-screen", "out.txt"], cwd=td,
                           capture_output=True, text=True)
        lines = (td / "out.txt").read_text().splitlines()
        for k, l in enumerate(lines):
            if l.split()[:2] == ["Step", "PotEng"]:
                return float(lines[k + 1].split()[1])
    raise RuntimeError("LAMMPS energy not found: " + r.stderr[-300:])


# geometries: the DFT scan sheet with Ar directly above the C atom nearest to the 'top' site
rows = list(connect(args.db).select(family="ar-scan"))
top = {float(r.kind.split("_d")[1]): r for r in rows if r.kind.startswith("top_d") and "rattled" not in r.kind}
dft = {d: r.energy for d, r in top.items() if getattr(r, 'energy', None) is not None}
base = top[max(top)].toatoms()                      # largest-distance row as the geometry template
ar = [i for i, s in enumerate(base.get_chemical_symbols()) if s == "Ar"][0]
c_idx = int(np.argmin(base.get_distances(ar, [i for i in range(len(base)) if i != ar], mic=True)))
c_idx = [i for i in range(len(base)) if i != ar][c_idx]


def geometry(d):
    at = base.copy()
    at.positions[ar, :2] = at.positions[c_idx, :2]
    at.positions[ar, 2] = at.positions[c_idx, 2] + d
    return at


calcs = {}
if args.models:
    import torch
    torch.set_num_threads(args.threads)
    from mace.calculators import MACECalculator
    for m in args.models:
        calcs[Path(m).stem] = MACECalculator(model_paths=os.path.expanduser(m), device="cpu", default_dtype="float64")

dist = sorted(set(args.dist) | {args.ref})
E = {"airebo+zbl": {}, **({"airebo+DFTpair": {}} if args.arc_table else {}), **{k: {} for k in calcs}, "DFT": {}}
for d in dist:
    at = geometry(d)
    E["airebo+zbl"][d] = airebo_energy(at)
    if args.arc_table:
        E["airebo+DFTpair"][d] = airebo_energy(at, args.arc_table)
    for k, c in calcs.items():
        a2 = at.copy()
        a2.calc = c
        E[k][d] = a2.get_potential_energy()
    key = min(dft, key=lambda x: abs(x - d)) if dft else None
    if key is not None and abs(key - d) < 1e-6:
        E["DFT"][d] = dft[key]
names = [n for n in E if E[n]]
print(f"interaction energy relative to d = {args.ref} A  (eV), Ar straight above one C atom, frozen DFT scan sheet")
print(f"{'d(A)':>6s}" + "".join(f"{n[:18]:>20s}" for n in names))
for d in dist:
    cells = []
    for n in names:
        cells.append(f"{E[n][d] - E[n][args.ref]:20.3f}" if d in E[n] and args.ref in E[n] else f"{'-':>20s}")
    print(f"{d:6.2f}" + "".join(cells))
print("\ndistance where the repulsive energy equals the impact energy (linear interpolation, frozen sheet):")
for n in names:
    ds = [d for d in dist if d in E[n]]
    ev = np.array([E[n][d] - E[n][args.ref] for d in ds])
    out = []
    for e in args.energies:
        k = np.where(ev >= e)[0]
        if len(k) and k.max() < len(ds) - 1:
            k = k.max()
            out.append(f"{e:g} eV: {ds[k] + (ds[k + 1] - ds[k]) * (ev[k] - e) / (ev[k] - ev[k + 1]):.2f} A")
        else:
            out.append(f"{e:g} eV: n/a")
    print(f"  {n:20s} " + "   ".join(out))

# ---- figure: interaction energy vs Ar-C distance (log scale), impact energies as horizontal lines, DFT limit as vertical line
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6.4, 4.4))
style = {"airebo+zbl": dict(color="tab:blue", lw=2), "DFT": dict(color="k", marker="o", ls="none", ms=6)}
for n in names:
    ds = [d for d in dist if d in E[n] and args.ref in E[n]]
    ev = [max(E[n][d] - E[n][args.ref], 1e-3) for d in ds]
    ax.plot(ds, ev, label=n, **style.get(n, dict(lw=2, ls="--")))
for e in args.energies:
    ax.axhline(e, color="0.8", lw=0.8, zorder=0)
    ax.text(max(dist), e, f" {e:g} eV", va="center", fontsize=7, color="0.4")
ax.axvline(1.65, color="tab:red", lw=1, ls=":")
ax.text(1.67, 0.3, "closest DFT\npoint", color="tab:red", fontsize=7)
ax.set_yscale("log")
ax.set_xlabel("Ar-C distance, Ar straight above one C atom (A)")
ax.set_ylabel("interaction energy (eV)")
ax.set_ylim(0.1, 150)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("wall_curve.png", dpi=160)
print("\nfigure: wall_curve.png")

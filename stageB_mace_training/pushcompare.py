"""Compare the energy (and forces) along the push paths of pushscan.py: DFT (labelled.db) vs AIREBO vs MACE model(s).

    python -W ignore pushcompare.py --airebo ../stageA_dft_check/CH.airebo --models data/ft/omat-0_all_ar/omat-0_all_ar.model
Prints E(s) - E(0) per family and the force RMSE of each potential against DFT; saves pushcompare.png.
"""
import argparse
import os
import shlex
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from ase.db import connect
from ase.io import write

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--db", default="data/pushscan/labelled.db")
p.add_argument("--airebo", required=True)
p.add_argument("--models", nargs="*", default=[])
p.add_argument("--lmp", default=os.environ.get("LMP_CMD", "lmp"))
p.add_argument("--threads", type=int, default=8)
args = p.parse_args()


def airebo(at):
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        write(td / "s.data", at, format="lammps-data", specorder=["C"], atom_style="atomic")
        (td / "in").write_text("\n".join([
            "units metal", "boundary p p p", "atom_style atomic", "read_data s.data", "mass 1 12.011",
            "pair_style airebo 3.0", f"pair_coeff * * {os.path.abspath(args.airebo)} C",
            "thermo_style custom step pe", "dump d all custom 1 f.dump id fx fy fz", "dump_modify d sort id", "run 0"]))
        subprocess.run(shlex.split(args.lmp) + ["-in", "in", "-log", "none", "-screen", "out.txt"], cwd=td, check=True)
        lines = (td / "out.txt").read_text().splitlines()
        e = next(float(lines[k + 1].split()[1]) for k, l in enumerate(lines) if l.split()[:2] == ["Step", "PotEng"])
        fl = (td / "f.dump").read_text().splitlines()
        f = np.array([[float(x) for x in l.split()[1:4]] for l in fl[9:9 + len(at)]])
    return e, f


calcs = {}
if args.models:
    import torch
    torch.set_num_threads(args.threads)
    from mace.calculators import MACECalculator
    for m in args.models:
        calcs[Path(m).stem] = MACECalculator(model_paths=os.path.expanduser(m), device="cpu", default_dtype="float64")

rows = sorted(connect(args.db).select(), key=lambda r: (r.family, float(r.kind.split("_s")[1])))
res = {}
for r in rows:
    at = r.toatoms()
    d = res.setdefault(r.family, {"s": [], "DFT": [], "AIREBO": [], **{k: [] for k in calcs}, "fdft": [], "f": {"AIREBO": [], **{k: [] for k in calcs}}})
    d["s"].append(float(r.kind.split("_s")[1]))
    d["DFT"].append(r.energy)
    d["fdft"].append(np.array(r.forces))
    e, f = airebo(at)
    d["AIREBO"].append(e)
    d["f"]["AIREBO"].append(f)
    for k, c in calcs.items():
        a2 = at.copy()
        a2.calc = c
        d[k].append(a2.get_potential_energy())
        d["f"][k].append(a2.get_forces())

names = ["DFT", "AIREBO", *calcs]
for fam, d in res.items():
    print(f"\n{fam}: energy relative to s = 0 (eV)")
    print(f"{'s (A)':>6s}" + "".join(f"{n[:16]:>18s}" for n in names))
    for k, s in enumerate(d["s"]):
        print(f"{s:6.2f}" + "".join(f"{d[n][k] - d[n][0]:18.3f}" for n in names))
    print("  force RMSE vs DFT over all points (eV/A), all atoms / pushed atoms only:")
    for n in names[1:]:
        err = np.concatenate([(a - b).ravel() for a, b in zip(d["f"][n], d["fdft"])])
        errz = np.array([np.sqrt(np.mean((a - b) ** 2)) for a, b in zip(d["f"][n], d["fdft"])])
        print(f"    {n[:18]:18s} {np.sqrt(np.mean(err ** 2)):7.3f}   max over points of per-structure rms {errz.max():7.3f}")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, len(res), figsize=(5.2 * len(res), 4), squeeze=False)
for ax, (fam, d) in zip(axs[0], res.items()):
    for n in names:
        ax.plot(d["s"], np.array(d[n]) - d[n][0], marker="o" if n == "DFT" else None, color="k" if n == "DFT" else None,
                ls="none" if n == "DFT" else "-", label=n)
    ax.set_title(fam)
    ax.set_xlabel("downward displacement s (A)")
    ax.set_ylabel("E(s) - E(0) (eV)")
    ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("pushcompare.png", dpi=160)
print("\nfigure: pushcompare.png")

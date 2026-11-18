"""DFT dimer scans (Ar-C, N-C, N-N) at short range as an arbiter of the MACE wall, and the comparison with MACE.

    python dimer_scan.py generate                       # data/dimers/pending.db, 23 structures
    nohup nice -n 10 mpirun -np 5 python dft_flow.py --pending data/dimers/pending.db --train data/dimers/labelled.db --workdir data/dimers > dimers_dft.log 2>&1 &
    python -W ignore dimer_scan.py compare --models data/ft/omat-0_all_n2/omat-0_all_n2.model

Two atoms along x in a 10 A cube with the recipe settings (PBE, PW 450 eV, spin-polarised, FermiDirac 0.05).  The recipe minimum
distances (Ar-C 1.6, C-N 1.1, N-N 1.0 A) exist because PAW spheres start to overlap below them (PAW labels are not trusted
there).  The scan goes below them on purpose, so rows below the minimum are INDICATIVE only: frozen-core / augmentation-sphere
overlap is a systematic error that no change of cutoff can show.  Rows at or above the minimum are trustworthy.
Energies are compared as E(r) - E(r_ref) with r_ref the largest distance of the scan, so the isolated-atom problem (an isolated
carbon atom does not converge in SCF) never enters; forces are the component along the bond.
A dimer in vacuum is far from the training data of MACE: use the SHAPE of the wall, not the absolute binding energy.
"""
import argparse
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.db import connect

import recipe as R

PAIRS = {
    "dimer-arc": (("Ar", "C"), [1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 2.0, 2.4], 1.6),
    "dimer-nc": (("N", "C"), [0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.6, 2.0], 1.1),
    "dimer-nn": (("N", "N"), [0.9, 1.0, 1.1, 1.2, 1.4, 1.6], 1.0),
}
BOX = 10.0

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("mode", choices=["generate", "compare"])
p.add_argument("--dir", default="data/dimers")
p.add_argument("--models", nargs="*", default=[])
p.add_argument("--dtype", default="float64")
args = p.parse_args()
D = Path(args.dir)


def dimer(a, b, r):
    at = Atoms([a, b], positions=[[BOX / 2 - r / 2, BOX / 2, BOX / 2], [BOX / 2 + r / 2, BOX / 2, BOX / 2]],
               cell=[BOX] * 3, pbc=True)
    mom = [0.3, 0.3]
    if a == "N" and b == "C":
        mom = [3.0, 0.3]                       # N atom is a quartet far away; compute() keeps moments only if one is > 1 muB
    at.set_initial_magnetic_moments(mom)
    return at


if args.mode == "generate":
    D.mkdir(parents=True, exist_ok=True)
    pdb = connect(D / "pending.db")
    have = {r.hash for r in pdb.select()}
    new = 0
    for fam, ((a, b), rs, zmin) in PAIRS.items():
        for r in rs:
            at = dimer(a, b, r)
            h = R.structure_hash(at)
            if h in have:
                continue
            pdb.write(at, hash=h, family=fam, kind=f"r{r:.2f}", rmin_recipe=zmin)
            have.add(h)
            new += 1
    print(f"{new} new rows in {D}/pending.db ({len(have)} total)")
    print("next:\n  export OMP_NUM_THREADS=1\n  nohup nice -n 10 mpirun -np 5 python dft_flow.py --pending "
          f"{D}/pending.db --train {D}/labelled.db --workdir {D} > dimers_dft.log 2>&1 &")
    raise SystemExit

# ---- compare
from mace.calculators import MACECalculator

Z = {"C": 6, "N": 7, "Ar": 18}


def zbl(a, b, r):
    z1, z2 = Z[a], Z[b]
    au = 0.46850 / (z1 ** 0.23 + z2 ** 0.23)
    x = r / au
    phi = 0.18175 * np.exp(-3.19980 * x) + 0.50986 * np.exp(-0.94229 * x) + 0.28022 * np.exp(-0.40290 * x) + 0.02817 * np.exp(-0.20162 * x)
    return 14.3996 * z1 * z2 / r * phi


calcs = {Path(m).stem: MACECalculator(model_paths=m, device="cpu", default_dtype=args.dtype) for m in args.models}
ldb = connect(D / "labelled.db")
lines, plot = [], {}
for fam, ((a, b), rs, zmin) in PAIRS.items():
    rows = sorted(ldb.select(family=fam), key=lambda r: float(r.kind[1:]))
    if len(rows) < 2:
        lines.append(f"\n{fam}: {len(rows)} labelled rows, skipped")
        continue
    ref = rows[-1]
    r_ref = float(ref.kind[1:])
    lines.append(f"\n{a}-{b}: E(r) - E({r_ref:.2f} A) in eV, bond force on atom 2 in eV/A; * = below the recipe minimum {zmin} A (indicative)")
    head = f"{'r':>5s} {'mu':>5s} {'E_dft':>9s}"
    for n in calcs:
        head += f" {n[-14:]:>14s}"
    head += f" {'ZBL':>8s} | {'F_dft':>8s}" + "".join(f" {n[-8:]:>8s}" for n in calcs)
    lines.append(head)
    e_m = {}
    for n, c in calcs.items():
        out = {}
        for r in rows:
            at = r.toatoms()
            at.calc = c
            out[r.id] = (at.get_potential_energy(), at.get_forces()[1, 0])
        e_m[n] = out
    plot[fam] = []
    for r in rows:
        rr = float(r.kind[1:])
        line = f"{rr:5.2f}{'*' if rr < zmin - 1e-9 else ' '}{r.get('mu', 0):5.2f} {r.energy - ref.energy:9.3f}"
        for n in calcs:
            line += f" {e_m[n][r.id][0] - e_m[n][ref.id][0]:14.3f}"
        line += f" {zbl(a, b, rr) - zbl(a, b, r_ref):8.2f} | {r.forces[1, 0]:8.2f}" + "".join(f" {e_m[n][r.id][1]:8.2f}" for n in calcs)
        lines.append(line)
        plot[fam].append((rr, r.energy - ref.energy, {n: e_m[n][r.id][0] - e_m[n][ref.id][0] for n in calcs}, zmin))
text = "\n".join(lines)
print(text)
(D / "dimer_compare.txt").write_text(text + "\n")
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, len(plot), figsize=(5 * max(len(plot), 1), 3.8), squeeze=False)
    for ax, (fam, pts) in zip(axs[0], plot.items()):
        r = np.array([t[0] for t in pts])
        ax.plot(r, [t[1] for t in pts], "ko", label="DFT")
        for n in calcs:
            ax.plot(r, [t[2][n] for t in pts], "-", label=n)
        ax.axvline(pts[0][3], color="g", lw=0.8)
        ax.set_yscale("symlog", linthresh=0.5)
        ax.set_title(f"{fam}  (green: recipe minimum)")
        ax.set_xlabel("r (A)")
    axs[0][0].set_ylabel("E - E(ref) (eV)")
    axs[0][0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(D / "dimer_compare.png", dpi=130)
except Exception as exc:
    print("plot skipped:", exc)
print(f"\nwritten: {D / 'dimer_compare.txt'}, {D / 'dimer_compare.png'}")

"""Evaluate a fine-tuned MACE model against the Stage D DFT labels (N and Ar data).  CPU, a few minutes.

    python -W ignore n_eval.py --model data/ft/omat-0_fold0of5_n1/omat-0_fold0of5_n1.model --holdout n-v1-2n

Prints, and writes next to the model (n_eval.txt, n_eval.png):
  1. per family: energy RMSE (meV/atom) and force RMSE (meV/A) against DFT; rows with a DFT force above --fcap eV/A are
     reported separately (they were down-weighted in training).  The family given with --holdout was NOT in training:
     its line is the generalisation test, every other line is a fit-quality check on training data.
  2. N formation energies of the 'edited' n-defect structures, DFT vs MACE.  Both use their own references (pristine
     relaxed sheet from data/train.db, N2 molecule row of the n-ref family), so constant offsets cancel.  Geometries are
     the unrelaxed single points of the DFT set.
  3. N-scan curves: energy relative to the far point (4.5 A) for the top / bridge / hollow approaches, DFT vs MACE.
"""
import argparse
import os
import re
from pathlib import Path

import numpy as np
from ase.db import connect

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--model", required=True)
p.add_argument("--dtype", default="float64")
p.add_argument("--holdout", default="", help="family that was held out of training (marked in the table)")
p.add_argument("--dbs", nargs="*", default=["data/arscan/labelled.db", "data/nscan/labelled.db", "data/ndefects/labelled.db"])
p.add_argument("--train-db", default="data/train.db")
p.add_argument("--fcap", type=float, default=15.0)
args = p.parse_args()

from mace.calculators import MACECalculator

calc = MACECalculator(model_paths=os.path.expanduser(args.model), device="cpu", default_dtype=args.dtype)
out_dir = Path(args.model).parent
lines = []


def say(s=""):
    print(s)
    lines.append(s)


def predict(row):
    at = row.toatoms()
    at.calc = calc
    return float(at.get_potential_energy()), np.array(at.get_forces())


res = []                                                     # dicts: family, kind, n, E_dft, E_mace, f_dft, f_mace
for db in args.dbs:
    for r in connect(db, serial=True).select():
        if r.natoms < 2:
            continue
        e, f = predict(r)
        res.append(dict(family=r.family, kind=r.get("kind", ""), n=r.natoms, e_dft=float(r.energy), e_mace=e,
                        f_dft=np.array(r.forces), f_mace=f))

# ---- 1. per-family errors
say(f"model: {args.model}")
say("\n1. errors against DFT, per family (E in meV/atom, F in meV/A; rows with DFT fmax > %.0f eV/A listed apart)" % args.fcap)
say(f"{'family':14s} {'rows':>4s} {'E rmse':>8s} {'E bias':>8s} {'F rmse':>8s} | {'rows':>4s} {'F rmse (big forces)':>20s}  note")
fams = sorted({x["family"] for x in res})
for fam in fams:
    rs = [x for x in res if x["family"] == fam]
    ok = [x for x in rs if np.abs(x["f_dft"]).max() <= args.fcap]
    big = [x for x in rs if np.abs(x["f_dft"]).max() > args.fcap]
    de = np.array([(x["e_mace"] - x["e_dft"]) / x["n"] for x in rs]) * 1000
    frm = lambda g: float(np.sqrt(np.mean(np.concatenate([(x["f_mace"] - x["f_dft"]).ravel() for x in g]) ** 2))) * 1000 if g else float("nan")
    note = "HELD OUT (not in training)" if fam == args.holdout else ""
    say(f"{fam:14s} {len(rs):4d} {np.sqrt(np.mean(de ** 2)):8.1f} {de.mean():8.1f} {frm(ok):8.0f} | {len(big):4d} {frm(big):20.0f}  {note}")

# ---- 2. formation energies
pr = [r for r in connect(args.train_db, serial=True).select(family="pristine", kind="relaxed", natoms=32)]
n2 = [x for x in res if x["family"] == "n-ref"]
say("\n2. N formation energies of the unrelaxed 'edited' structures (eV): E - nC*E(C in pristine) - nN*E(N in N2)")
if pr and n2:
    a = pr[0].toatoms()
    a.calc = calc
    epc_d, epc_m = float(pr[0].energy) / 32, float(a.get_potential_energy()) / 32
    epn_d, epn_m = n2[0]["e_dft"] / 2, n2[0]["e_mace"] / 2
    say(f"   references per atom   DFT: C {epc_d:.4f}  N {epn_d:.4f}    MACE: C {epc_m:.4f}  N {epn_m:.4f}")
    say(f"{'family':14s} {'N':>2s} {'DFT':>8s} {'MACE':>8s} {'diff':>7s}  note")
    for fam in [f for f in fams if f.startswith("n-") and f != "n-ref" and f != "n-scan"]:
        for x in res:
            if x["family"] == fam and x["kind"] == "edited":
                rows = [r for r in connect("data/ndefects/labelled.db", serial=True).select(family=fam, kind="edited")]
                sym = list(rows[0].symbols)
                nN, nC = sym.count("N"), sym.count("C")
                ed = x["e_dft"] - nC * epc_d - nN * epn_d
                em = x["e_mace"] - nC * epc_m - nN * epn_m
                say(f"{fam:14s} {nN:2d} {ed:8.2f} {em:8.2f} {em - ed:7.2f}  {'HELD OUT' if fam == args.holdout else ''}")
else:
    say("   pristine relaxed row (data/train.db) or n-ref row missing: skipped")

# ---- 3. N-scan curves
say("\n3. N-scan: energy relative to the 4.5 A point of the same site (eV, whole cell), DFT vs MACE (training data!)")
curves = {}
for site in ("top", "bridge", "hollow"):
    pts = {}
    for x in res:
        m = re.fullmatch(rf"{site}_d([0-9.]+)", x["kind"])
        if x["family"] == "n-scan" and m:
            pts[float(m.group(1))] = x
    if 4.5 not in pts:
        continue
    ref = pts[4.5]
    curves[site] = sorted((d, v["e_dft"] - ref["e_dft"], v["e_mace"] - ref["e_mace"]) for d, v in pts.items())
    say(f"  {site}: " + "  ".join(f"{d:.2f}:{ed:+.2f}/{em:+.2f}" for d, ed, em in curves[site]))
say("  (format d: DFT/MACE)")

(out_dir / "n_eval.txt").write_text("\n".join(lines) + "\n")
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, len(curves), figsize=(4.2 * max(len(curves), 1), 3.6), sharey=True, squeeze=False)
    for ax, (site, c) in zip(axs[0], curves.items()):
        d = np.array([t[0] for t in c])
        ax.plot(d, [t[1] for t in c], "ko", label="DFT")
        ax.plot(d, [t[2] for t in c], "r-", label="MACE")
        ax.set_title(f"N on {site}")
        ax.set_xlabel("N-C distance (A)")
        ax.set_ylim(top=8)
    axs[0][0].set_ylabel("E - E(4.5 A) (eV)")
    axs[0][0].legend()
    fig.tight_layout()
    fig.savefig(out_dir / "n_eval.png", dpi=130)
except Exception as exc:
    print("plot skipped:", exc)
print(f"\nwritten: {out_dir / 'n_eval.txt'}, {out_dir / 'n_eval.png'}")

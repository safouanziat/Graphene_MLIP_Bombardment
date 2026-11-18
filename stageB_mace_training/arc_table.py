"""Fit an Ar-C PAIR potential to the DFT Ar scans and write it as a LAMMPS table (replaces plain ZBL for the Ar-C pair).

Model: for every flat scan row of data/arscan/labelled.db (same frozen sheet, Ar above top / bridge / hollow, closest Ar-C 1.65-4.5 A)
    E_row - E(isolated Ar) = E_sheet + sum over C atoms (periodic images included, r < 5 A) of  phi(r)
    phi(r) = [A1 exp(-b1 r) + A2 exp(-b2 r)] * taper(r)        taper: 1 below 4 A, smooth to 0 at 5 A
E_sheet is an unknown constant (the pristine sheet energy is not in the database) and is fitted too.
For the table, phi is blended into the ZBL potential below 1.6 A (smoothstep 1.1 -> 1.6 A), because DFT stops at 1.65 A.

    python -W ignore arc_table.py --db data/arscan/labelled.db --out data/stageC/arc.table --plot
Then:
    python -W ignore stageC_pilot.py airebo --airebo ... --arc-table data/stageC/arc.table --cases top:40 top:45 ... --outroot data/stageC/airebo_arc
"""
import argparse
from pathlib import Path

import numpy as np
from ase.db import connect
from ase.neighborlist import neighbor_list
from scipy.optimize import least_squares

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--db", default="data/arscan/labelled.db")
p.add_argument("--out", default="data/stageC/arc.table")
p.add_argument("--sites", nargs="+", default=["top", "bridge", "hollow"], help="scan sites used in the fit")
p.add_argument("--terms", type=int, default=2, choices=[1, 2], help="number of exponentials")
p.add_argument("--rc", type=float, default=5.0)
p.add_argument("--taper", type=float, default=1.0, help="width of the smooth cut-off (A), ending at --rc")
p.add_argument("--blend", nargs=2, type=float, default=[1.1, 1.6], help="ZBL -> fit blend range (A)")
p.add_argument("--rmin", type=float, default=0.3)
p.add_argument("--npts", type=int, default=5000)
p.add_argument("--plot", action="store_true")
args = p.parse_args()


def smooth(x):                      # 0 for x<=0, 1 for x>=1, C1
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def taper(r):
    return 1.0 - smooth((r - (args.rc - args.taper)) / args.taper)


def zbl(r, z1=18, z2=6):
    a = 0.46850 / (z1 ** 0.23 + z2 ** 0.23)
    x = r / a
    phi = 0.18175 * np.exp(-3.19980 * x) + 0.50986 * np.exp(-0.94229 * x) + 0.28022 * np.exp(-0.40290 * x) + 0.02817 * np.exp(-0.20162 * x)
    return 14.3996 * z1 * z2 / r * phi


def phi_fit(r, q):
    return (q[0] * np.exp(-q[1] * r) + (q[2] * np.exp(-q[3] * r) if args.terms == 2 else 0.0)) * taper(r)


# ---- data
rows = [r for r in connect(args.db).select(family="ar-scan") if "rattled" not in r.kind and r.kind.split("_d")[0] in args.sites and getattr(r, "energy", None) is not None]
atom = [r for r in connect(args.db).select(family="ar-atom")]
if not rows or not atom:
    raise SystemExit("need flat 'ar-scan' rows and the isolated 'ar-atom' row with energies in " + args.db)
e_ar = atom[0].energy
pairs, y = [], []
for r in rows:
    at = r.toatoms()
    ar = [i for i, s in enumerate(at.get_chemical_symbols()) if s == "Ar"][0]
    i, j, d = neighbor_list("ijd", at, args.rc)
    sel = (i == ar) & (np.array(at.get_chemical_symbols())[j] == "C")
    pairs.append(d[sel])
    y.append(r.energy - e_ar)
y = np.array(y)
print(f"{len(rows)} flat scan rows, closest Ar-C {min(d.min() for d in pairs):.2f} A, {np.mean([len(d) for d in pairs]):.0f} C neighbours within {args.rc} A on average")


def model(par):
    q, e0 = par[:4], par[4]
    return np.array([phi_fit(d, q).sum() for d in pairs]) + e0


def resid(par):
    return model(par) - y


best = None
rng = np.random.default_rng(0)
for t in range(30):
    x0 = [rng.uniform(50, 5000), rng.uniform(1.5, 4.0), rng.uniform(1, 200) if args.terms == 2 else 0.0, rng.uniform(0.5, 2.0), y.min() - 1]
    try:
        s = least_squares(resid, x0, bounds=([0, 0.2, 0, 0.1, -1e5], [1e6, 8, 1e5 if args.terms == 2 else 1e-9, 8, 1e5]))
    except Exception:
        continue
    if best is None or s.cost < best.cost:
        best = s
par = best.x
rms = np.sqrt(np.mean(resid(par) ** 2))
print(f"fit: A1={par[0]:.3g} b1={par[1]:.3f}  A2={par[2]:.3g} b2={par[3]:.3f}   E_sheet offset {par[4]:.3f} eV")
print(f"rms residual {1000 * rms:.0f} meV, max {1000 * np.abs(resid(par)).max():.0f} meV over {len(y)} points")
res_all = resid(par)
for site in args.sites:
    k = [n for n, r in enumerate(rows) if r.kind.startswith(site + "_d")]
    if k:
        print(f"   {site:7s} rms {1000 * np.sqrt(np.mean(res_all[k] ** 2)):4.0f} meV, max {1000 * np.abs(res_all[k]).max():4.0f} meV")
if rms > 0.05:
    print("   WARNING: rms residual above 50 meV: not a faithful pair potential; use only as a diagnostic for the sites fitted")

# ---- table
r = np.linspace(args.rmin, args.rc, args.npts)
s = smooth((r - args.blend[0]) / (args.blend[1] - args.blend[0]))
phi = (1 - s) * zbl(r) + s * phi_fit(r, par[:4])
phi = np.where(r >= args.rc - 1e-9, 0.0, phi)
f = -np.gradient(phi, r)
out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w") as fh:
    fh.write("# Ar-C pair potential: DFT-fitted exponentials, ZBL below ~1.5 A (arc_table.py)\n\nARC\n")
    fh.write(f"N {args.npts} R {r[0]:.6f} {r[-1]:.6f}\n\n")
    for k, (a, b, c) in enumerate(zip(r, phi, f), 1):
        fh.write(f"{k} {a:.6f} {b:.8f} {c:.8f}\n")
print("table:", out)
for d in (1.0, 1.4, 1.65, 2.0, 2.5, 3.0, 4.0):
    k = int(np.argmin(abs(r - d)))
    print(f"   phi({d:.2f} A) = {phi[k]:9.3f} eV   (ZBL {zbl(d):9.3f})")

if args.plot:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(r, np.maximum(phi, 1e-4), label="DFT-fitted pair (table)", lw=2)
    ax.semilogy(r, zbl(r) * np.where(r < 4, 1, np.maximum(0, 5 - r)), "--", label="ZBL (current, no switch shown)")
    ax.axvline(1.65, color="tab:red", ls=":", lw=1)
    ax.set_xlim(0.6, 5)
    ax.set_ylim(1e-3, 300)
    ax.set_xlabel("Ar-C distance (A)")
    ax.set_ylabel("pair energy (eV)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=160)
    print("figure:", out.with_suffix(".png"))

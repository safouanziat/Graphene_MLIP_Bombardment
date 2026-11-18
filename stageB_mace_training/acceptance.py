"""
Acceptance run: does the fine-tuned model give the right defect physics on a SIZE IT WAS NOT TRAINED ON (5x5, 50 atoms)?

  build   relax pristine / V1 / V2 / SW / C-adatom (bridge, top) 5x5 sheets WITH THE MODEL, write the geometries as
          unlabelled rows to data/acceptance/pending.db (family 'acc-<name>')  -> DFT single points, no DFT relaxation
  (label) python dft_flow.py --pending data/acceptance/pending.db --train data/acceptance/labelled.db --workdir data/acceptance
  report  DFT vs model at the SAME geometries: formation energies, forces, adatom bridge-vs-top, 4x4 vs 5x5 cell size,
          optional comparison with other models evaluated on these geometries

    python acceptance.py build  --model data/ft/omat-0_all/omat-0_all.model
    python acceptance.py report --model data/ft/omat-0_all/omat-0_all.model --also omat0=~/.cache/mace/maceomat0mediummodel

MLIP_TOY=1 uses EMT as "model" for testing the workflow (no torch needed).
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.build import graphene
from ase.constraints import FixAtoms, FixedLine
from ase.db import connect
from ase.io import read, write
from ase.optimize import BFGS

import recipe as R

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("mode", choices=["build", "report"])
p.add_argument("--model", default="", help="relaxation model (.model path); ignored with MLIP_TOY=1")
p.add_argument("--also", nargs="*", default=[], help="name=path of further models evaluated at the same geometries")
p.add_argument("--src", default="../stageA_dft_check/dft_cheap/relaxed")
p.add_argument("--dir", default="data/acceptance")
p.add_argument("--train", default="data/train.db", help="4x4 training labels (cell-size comparison)")
p.add_argument("--n", type=int, default=5, help="supercell size")
p.add_argument("--fmax", type=float, default=0.02)
p.add_argument("--dtype", default="float64")
p.add_argument("--threads", type=int, default=8)
args = p.parse_args()
D = Path(args.dir)
GEOM = D / "geometries.extxyz"


def calculator(spec):
    if R.TOY:
        from ase.calculators.emt import EMT
        return EMT()
    import torch
    torch.set_num_threads(args.threads)
    from mace.calculators import MACECalculator
    return MACECalculator(model_paths=os.path.expanduser(spec), device="cpu", default_dtype=args.dtype)


# ----------------------------------------------------------------------------------------------------------------- build
def lattice_constant():
    f = Path(args.src) / "gpaw_pristine_n4.extxyz"
    if R.TOY or not f.exists():
        return 2.46
    return float(np.linalg.norm(read(f).cell[0])) / 4


def sheet(a, n):
    at = graphene(a=a, size=(n, n, 1), vacuum=10.0)
    at.pbc = True
    return at


def nearest_pair(at, i):
    d = at.get_distances(i, range(len(at)), mic=True)
    d[i] = 9e9
    return int(np.argmin(d))


def make_structures(a, n):
    pris = sheet(a, n)
    c = int(np.argmin(np.linalg.norm(pris.positions[:, :2] - pris.cell[:2, :2].sum(0) / 2, axis=1)))
    j = nearest_pair(pris, c)
    out = {"pristine": (pris, [])}
    v1 = pris.copy()
    del v1[c]
    out["V1"] = (v1, [])
    v2 = pris.copy()
    del v2[sorted([c, j])[::-1][0]]
    del v2[sorted([c, j])[0]]
    out["V2"] = (v2, [])
    sw = pris.copy()                      # rotate the c-j bond by 90 degrees about its midpoint, in the sheet plane
    mid = 0.5 * (sw.positions[c] + sw.positions[j])
    for k in (c, j):
        v = sw.positions[k] - mid
        sw.positions[k] = mid + np.array([-v[1], v[0], v[2]])
    out["SW"] = (sw, [])
    for name, site in (("adatom-bridge", 0.5 * (pris.positions[c] + pris.positions[j])), ("adatom-top", pris.positions[c])):
        ad = pris.copy()
        z = ad.positions[:, 2].mean()
        ad += Atoms("C", positions=[[site[0], site[1], z + 1.9]], cell=ad.cell, pbc=True)
        out[name] = (ad, [len(ad) - 1])   # adatom xy is frozen, z free: the two sites stay distinguishable
    return out


def relax(at, adatom, calc):
    at = at.copy()
    at.calc = calc
    cons = [FixedLine(i, [0, 0, 1]) for i in adatom]
    if cons:
        at.set_constraint(cons)
    BFGS(at, logfile=None).run(fmax=args.fmax, steps=400)
    at.set_constraint()
    at.calc = None
    return at


if args.mode == "build":
    D.mkdir(parents=True, exist_ok=True)
    calc = calculator(args.model)
    a = lattice_constant()
    print(f"lattice constant a = {a:.4f} A (from the DFT-relaxed 4x4 pristine), {args.n}x{args.n} cells")
    pdb = connect(D / "pending.db")
    have = {r.hash for r in pdb.select()}
    rows = []
    for name, (at, adatom) in make_structures(a, args.n).items():
        rel = R.normalise(relax(at, adatom, calc))
        h = R.structure_hash(rel)
        ok, margin = R.worst_pair(rel)
        if not ok:
            print(f"  {name}: REJECTED, closest pair violates recipe min_dist (margin {margin:.2f})")
            continue
        rel.calc = calc
        e, f = float(rel.get_potential_energy()), rel.get_forces()
        rel.calc = None
        if h not in have:
            pdb.write(rel, hash=h, family=f"acc-{name}", kind="model-relaxed", margin=round(margin, 3))
        rel.info.update(name=name, hash=h, model_energy=e)
        rel.arrays["model_forces"] = f
        rows.append(rel)
        print(f"  {name:14s} {len(rel):3d} atoms  E_model {e:.3f} eV  max|F| {np.abs(f).max():.3f}  hash {h}")
    write(GEOM, rows)
    print(f"\nwrote {GEOM} and {D}/pending.db ({pdb.count()} rows). Next:\n  python dft_flow.py --pending {D}/pending.db "
          f"--train {D}/labelled.db --workdir {D}")

# ---------------------------------------------------------------------------------------------------------------- report
else:
    geoms = {a.info["name"]: a for a in read(GEOM, ":")}
    lab = {r.hash: r for r in connect(D / "labelled.db", serial=True).select()} if (D / "labelled.db").exists() else {}
    missing = [n for n, a in geoms.items() if a.info["hash"] not in lab]
    if missing:
        print("not yet labelled by DFT:", ", ".join(missing))
    names = [n for n in geoms if n not in missing]
    if "pristine" not in names:
        raise SystemExit("pristine 5x5 needs a DFT label first")

    models = {"model": args.model}
    models.update(dict(s.split("=", 1) for s in args.also))
    ev = {}
    for mname, spec in models.items():
        calc = calculator(spec)
        ev[mname] = {}
        for n in names:
            at = geoms[n].copy()
            at.calc = calc
            ev[mname][n] = (float(at.get_potential_energy()), at.get_forces())

    def formation(energy):
        """eV; reference = pristine energy per atom of the same method (adatom: + one C atom)"""
        mu = energy["pristine"] / len(geoms["pristine"])
        return {n: energy[n] - len(geoms[n]) * mu for n in energy if n != "pristine"}

    dft_e = {n: float(lab[geoms[n].info["hash"]].energy) for n in names}
    dft_f = {n: np.array(lab[geoms[n].info["hash"]].forces) for n in names}
    fd = formation(dft_e)
    fm = {m: formation({n: ev[m][n][0] for n in names}) for m in models}

    print(f"\n5x5 formation energies (eV) at the model-relaxed geometries, DFT = {R.RECIPE_ID} recipe single points")
    print(f"{'structure':14s} {'DFT':>8s} " + " ".join(f"{m:>12s}" for m in models) + "   (error vs DFT in brackets)")
    for n in fd:
        print(f"{n:14s} {fd[n]:8.3f} " + " ".join(f"{fm[m][n]:6.3f}({fm[m][n] - fd[n]:+.2f})" for m in models))

    print("\nforce RMSE vs DFT at these geometries (eV/A)  [DFT residual force of the model geometry in the last column]")
    print(f"{'structure':14s} " + " ".join(f"{m:>12s}" for m in models) + f" {'rms F_DFT':>10s}")
    for n in names:
        print(f"{n:14s} " + " ".join(f"{np.sqrt(np.mean((ev[m][n][1] - dft_f[n]) ** 2)):12.3f}" for m in models)
              + f" {np.sqrt(np.mean(dft_f[n] ** 2)):10.3f}")

    if {"adatom-bridge", "adatom-top"} <= set(fd):
        print("\nadatom ordering  E(top) - E(bridge)  (eV, bridge should be lower, i.e. positive):")
        print(f"  DFT {fd['adatom-top'] - fd['adatom-bridge']:+.3f}   " +
              "   ".join(f"{m} {fm[m]['adatom-top'] - fm[m]['adatom-bridge']:+.3f}" for m in models))

    # 4x4 versus 5x5 (DFT labels, same recipe). The 4x4 relaxed bases are the training rows with kind 'relaxed'.
    if Path(args.train).exists():
        rel = {r.family: r for r in connect(args.train, serial=True).select() if r.get("kind") == "relaxed"}
        if "pristine" in rel:
            mu = float(rel["pristine"].energy) / rel["pristine"].natoms
            f4 = {k: float(r.energy) - r.natoms * mu for k, r in rel.items() if k != "pristine"}
            print("\ncell-size check, DFT formation energy 4x4 (DFT-relaxed) vs 5x5 (model-relaxed geometry), eV:")
            for k4, k5 in (("V1", "V1"), ("V2", "V2"), ("SW", "SW"), ("adatom", "adatom-bridge")):
                if k4 in f4 and k5 in fd:
                    print(f"  {k4:8s} 4x4 {f4[k4]:7.3f}   5x5 {fd[k5]:7.3f}   diff {fd[k5] - f4[k4]:+.3f}")

    res = dict(dft=fd, models={m: fm[m] for m in models})
    (D / "acceptance_results.json").write_text(json.dumps(res, indent=2))
    print(f"\nwrote {D}/acceptance_results.json")

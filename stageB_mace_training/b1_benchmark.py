"""
B1 - zero-shot benchmark of pretrained universal MLIPs against the Stage B DFT labels (no training involved).

    python b1_benchmark.py --models mace-mp-0 mace-mpa-0 mace-omat-0 orb-v3 sevennet
    python b1_benchmark.py --models mace-mp-0 --no-timing         # quick first test
    python b1_benchmark.py --list

Reads data/train.db (all rows of RECIPE 8c357474ed, no GPAW import needed) and writes
    data/b1/<model>.npz        per-structure predictions (reused on reruns; delete to recompute)
    data/b1_results.json       all metrics
and prints a table.  Models that are not installed or cannot download weights are skipped with a message.

Metrics (all against the DFT labels of this project, PBE/GPAW):
  force RMSE per family, and relative to the DFT force RMS of that family
  energy RMSE per atom after a least-squares per-element offset (absolute energies of different codes are not
      comparable; the offset absorbs the reference-energy mismatch)
  formation energy of the relaxed defect structures: E - nC * (E/N of pristine), model vs DFT
  Spearman rank correlation of energies inside each (family, natoms) group - does the model order structures
      of the same composition correctly?
  CPU time per energy+forces call at ~32 and ~512 atoms (pristine graphene)

Notes: foundation models are not spin polarised, V1 labels are (mu ~ 1.5 muB).  Install each model family in its own
venv if torch versions clash; do NOT upgrade torch inside the GPAW environment.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from ase.build import graphene
from ase.db import connect

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--train", default="data/train.db")
p.add_argument("--out", default="data")
p.add_argument("--models", nargs="+", default=["mace-mp-0"])
p.add_argument("--dtype", default="float64", choices=["float32", "float64"])
p.add_argument("--no-timing", action="store_true")
p.add_argument("--list", action="store_true")
p.add_argument("--subset", default="", help="extxyz (e.g. a valid.extxyz): evaluate only the structures whose hash is in it")
p.add_argument("--threads", type=int, default=8, help="torch CPU threads for the timing")
args = p.parse_args()


# --------------------------------------------------------------------------------------------- model loaders
def _mace(name):
    def f():
        from mace.calculators import mace_mp
        return mace_mp(model=name, default_dtype=args.dtype, device="cpu", dispersion=False)
    return f


def _orb(fn):
    def f():
        from orb_models.forcefield import pretrained
        from orb_models.forcefield.calculator import ORBCalculator
        orbff = getattr(pretrained, fn)(device="cpu", precision="float32-high")
        return ORBCalculator(orbff, device="cpu")
    return f


def _seven(name, modal=None):
    def f():
        from sevenn.calculator import SevenNetCalculator
        kw = dict(modal=modal) if modal else {}
        return SevenNetCalculator(name, device="cpu", **kw)
    return f


def _emt():  # pipeline test only
    from ase.calculators.emt import EMT
    return EMT()


LOADERS = {
    "mace-mp-0": _mace("medium"),                    # MACE-MP-0 medium (PBE, Materials Project trajectories)
    "mace-mpa-0": _mace("medium-mpa-0"),             # MPtrj + sAlex
    "mace-omat-0": _mace("medium-omat-0"),           # OMat24 (PBE, rattled/high-T structures)
    "orb-v3": _orb("orb_v3_conservative_inf_omat"),
    "sevennet": _seven("7net-0"),
    "sevennet-mf-ompa": _seven("7net-mf-ompa", "mpa"),
    "emt": _emt,
}
def _ft(path):
    def f():
        from mace.calculators import MACECalculator
        return MACECalculator(model_paths=path, device="cpu", default_dtype=args.dtype)
    return f


for m in args.models:
    if m.startswith("ft:"):
        LOADERS[m] = _ft(m[3:])
if args.list:
    print("\n".join(LOADERS))
    raise SystemExit

# --------------------------------------------------------------------------------------------- data
rows = []
for r in connect(args.train, serial=True).select():
    at = r.toatoms()
    rows.append(dict(hash=r.hash, family=r.family, kind=r.get("kind", ""), atoms=at, E=float(r.energy),
                     F=np.array(r.forces), mu=r.get("mu", 0.0)))
rows.sort(key=lambda d: d["hash"])
SUB = ""
all_rows = list(rows)
if args.subset:
    from ase.io import read as _read
    keep = {a.info["hash"] for a in _read(args.subset, ":")}
    rows = [d for d in rows if d["hash"] in keep]
    SUB = "@" + Path(args.subset).parent.name
    print(f"subset {args.subset}: {len(rows)} structures")
print(f"{len(rows)} labelled structures from {args.train}")
out = Path(args.out)
(out / "b1").mkdir(parents=True, exist_ok=True)
elements = sorted({s for d in rows for s in d["atoms"].get_chemical_symbols()})
fams = sorted({d["family"] for d in rows})
REF = None  # relaxed pristine sheet, evaluated only as the zero of the formation energies when the subset lacks it
if args.subset and not any(d["family"] == "pristine" and d["kind"] == "relaxed" for d in rows):
    cand = [d for d in all_rows if d["family"] == "pristine" and d["kind"] == "relaxed"]
    if cand:
        REF = cand[0]
        rows.append(REF)


def predict(name):
    cache = out / "b1" / (name.replace("/", "_").replace(":", "_") + SUB.replace("/", "_") + ".npz")
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        if list(z["hash"]) == [d["hash"] for d in rows]:
            return z["E"], list(z["F"])
    calc = LOADERS[name]()
    E, F = [], []
    t0 = time.time()
    for d in rows:
        a = d["atoms"].copy()
        a.calc = calc
        E.append(float(a.get_potential_energy()))
        F.append(np.array(a.get_forces()))
    print(f"  {name}: {len(rows)} structures in {time.time() - t0:.0f} s")
    np.savez(cache, hash=np.array([d["hash"] for d in rows]), E=np.array(E), F=np.array(F, dtype=object))
    return np.array(E), F


def spearman(x, y):
    if len(x) < 3:
        return None
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def metrics(E, F, rows):
    res = {}
    ref = None
    if REF is not None:  # last row is the reference, not part of the statistics
        ref = (rows[-1], float(E[-1]))
        rows, E, F = rows[:-1], np.asarray(E)[:-1], list(F)[:-1]
    n = np.array([len(d["atoms"]) for d in rows])
    Ed = np.array([d["E"] for d in rows])
    comp = np.array([[d["atoms"].get_chemical_symbols().count(s) for s in elements] for d in rows], float)
    off, *_ = np.linalg.lstsq(comp, E - Ed, rcond=None)
    dE = (E - Ed - comp @ off) / n * 1000  # meV/atom after element offsets
    for fam in fams:
        idx = [i for i, d in enumerate(rows) if d["family"] == fam]
        if not idx:
            continue
        fd = np.concatenate([rows[i]["F"].ravel() for i in idx])
        fm = np.concatenate([F[i].ravel() for i in idx])
        rmse = float(np.sqrt(np.mean((fd - fm) ** 2)))
        rms = float(np.sqrt(np.mean(fd ** 2)))
        rho = {}
        for natoms in sorted({n[i] for i in idx}):
            sub = [i for i in idx if n[i] == natoms]
            s = spearman(Ed[sub], E[sub])
            if s is not None:
                rho[int(natoms)] = round(s, 3)
        res[fam] = dict(n=len(idx), force_rmse=round(rmse, 4), force_rms_dft=round(rms, 4),
                        force_rel=round(rmse / rms, 3), energy_rmse_meV_atom=round(float(np.sqrt(np.mean(dE[idx] ** 2))), 1),
                        spearman=rho)
    # formation energies of the relaxed bases, relative to the relaxed pristine sheet (C only)
    pr = [i for i, d in enumerate(rows) if d["family"] == "pristine"]
    rel = ([i for i in pr if rows[i]["kind"] == "relaxed"] or [min(pr, key=lambda i: rows[i]["E"] / n[i])]) if pr else []
    zero = None  # (DFT, model) energy per atom of the relaxed pristine sheet
    if ref is not None:
        zero = (ref[0]["E"] / len(ref[0]["atoms"]), ref[1] / len(ref[0]["atoms"]))
    elif rel:
        zero = (Ed[rel[0]] / n[rel[0]], E[rel[0]] / n[rel[0]])
    form = {}
    if zero:
        for i, d in enumerate(rows):
            if d["kind"] == "relaxed" and d["family"] != "pristine" and set(d["atoms"].get_chemical_symbols()) == {"C"}:
                ef_d = d["E"] - n[i] * zero[0]
                ef_m = E[i] - n[i] * zero[1]
                form[d["family"]] = dict(dft=round(ef_d, 3), model=round(ef_m, 3), err=round(ef_m - ef_d, 3))
    import re

    def amp(k):
        m = re.match(r"rattle([0-9.]+)", k or "")
        return float(m.group(1)) if m else 0.0

    bykind = {}
    for lab, sel in (("relaxed", lambda a: a == 0), ("rat<=.06", lambda a: 0 < a <= 0.06),
                     ("rat.10", lambda a: abs(a - 0.10) < 1e-6), ("rat.15", lambda a: a >= 0.15)):
        idx = [i for i, d in enumerate(rows) if sel(amp(d["kind"]))]
        if idx:
            fd = np.concatenate([rows[i]["F"].ravel() for i in idx])
            fm = np.concatenate([F[i].ravel() for i in idx])
            bykind[lab] = dict(n=len(idx), rmse=round(float(np.sqrt(np.mean((fd - fm) ** 2))), 3),
                               dft_rms=round(float(np.sqrt(np.mean(fd ** 2))), 2))
    res["_by_kind"] = bykind
    all_f = np.concatenate([d["F"].ravel() for d in rows])
    all_m = np.concatenate([f.ravel() for f in F])
    res["_all"] = dict(force_rmse=round(float(np.sqrt(np.mean((all_f - all_m) ** 2))), 4),
                       energy_rmse_meV_atom=round(float(np.sqrt(np.mean(dE ** 2))), 1),
                       element_offsets_eV={s: round(float(o), 3) for s, o in zip(elements, off)})
    res["_formation_eV"] = form
    return res


def timing(name):
    try:
        import torch
        torch.set_num_threads(args.threads)
    except ImportError:
        pass
    calc = LOADERS[name]()
    out_t = {}
    for label, size in (("32", (4, 4, 1)), ("512", (16, 16, 1))):
        a = graphene(size=size, vacuum=10.0)  # 2*size atoms
        a.pbc = True
        a.rattle(0.02, seed=1)
        a.calc = calc
        a.get_forces()  # warm-up
        ts = []
        for _ in range(3):
            a.rattle(0.001, seed=2)
            t0 = time.perf_counter()
            a.get_potential_energy(), a.get_forces()
            ts.append(time.perf_counter() - t0)
        out_t[f"{len(a)}_atoms_s"] = round(float(np.median(ts)), 3)
    return out_t


results = {}
for name in args.models:
    if name not in LOADERS:
        print(f"unknown model {name}; --list shows the names")
        continue
    print(f"== {name}")
    try:
        E, F = predict(name)
        results[name + SUB] = metrics(E, F, rows)
        if not args.no_timing:
            results[name + SUB]["cpu_time"] = timing(name)
    except Exception as exc:  # missing package, no network for weights, unsupported element ...
        print(f"  SKIPPED {name}: {type(exc).__name__}: {str(exc)[:200]}")

prev = out / "b1_results.json"
merged = json.loads(prev.read_text()) if prev.exists() else {}
merged.update(results)  # runs in different environments add to the same file
prev.write_text(json.dumps(merged, indent=2))

# ---------------------------------------------------------------------------------------------- table
if results:
    print("\nforce RMSE (eV/A) per family; energy RMSE after element offset (meV/atom)")
    print(f"{'model':18s}" + "".join(f"{f:>9s}" for f in fams) + f"{'ALL F':>9s}{'ALL E':>9s}")
    for m, r in results.items():
        print(f"{m:18s}" + "".join(f"{r[f]['force_rmse']:9.3f}" for f in fams)
              + f"{r['_all']['force_rmse']:9.3f}{r['_all']['energy_rmse_meV_atom']:9.1f}")
    print("\nforce RMSE (eV/A) by structure kind  [DFT force rms in brackets]")
    kinds = ["relaxed", "rat<=.06", "rat.10", "rat.15"]
    first = next((r["_by_kind"] for r in results.values() if r.get("_by_kind")), {})
    print(f"{'model':18s}" + "".join(f"{k + ' [' + str(first[k]['dft_rms']) + ']':>18s}" for k in kinds if k in first))
    for m, r in results.items():
        print(f"{m:18s}" + "".join(f"{r['_by_kind'][k]['rmse']:18.3f}" for k in kinds if k in r.get("_by_kind", {})))
    print("\nformation energy error of relaxed defects (eV), model - DFT   [DFT value]")
    famsF = sorted({k for r in results.values() for k in r["_formation_eV"]})
    for m, r in results.items():
        print(f"{m:18s}" + "  ".join(f"{k} {r['_formation_eV'][k]['err']:+.2f} [{r['_formation_eV'][k]['dft']:.2f}]"
                                       for k in famsF if k in r["_formation_eV"]))
    print("\nSpearman rank correlation of energies inside each (family, natoms) group")
    for m, r in results.items():
        print(f"{m:18s}" + "  ".join(f"{f} {list(r[f]['spearman'].values())}" for f in fams if r[f]["spearman"]))
    if any("cpu_time" in r for r in results.values()):
        print("\nCPU time per energy+forces call (s)")
        for m, r in results.items():
            if "cpu_time" in r:
                print(f"{m:18s}{r['cpu_time']}")
print("\nwrote", out / "b1_results.json")

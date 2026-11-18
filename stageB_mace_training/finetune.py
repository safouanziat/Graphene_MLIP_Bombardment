"""
Fine-tune a MACE foundation model on the Stage B labels (naive fine-tuning, CPU).

    python finetune.py --base omat-0 --fold 0                      # fold 0 of 5, stratified inside families
    python finetune.py --base mpa-0  --holdout-family frenkel      # leave-one-family-out
    python finetune.py --base omat-0 --fold 0 --freeze 1           # ablation: freeze the first layers
    python finetune.py --base omat-0 --fold 0 --dry-run            # print the command only

What it does
  1. runs export.py for the chosen split into data/ft/<run>/  (train.extxyz, valid.extxyz, statistics.json)
  2. runs mace_run_train with the foundation model loaded from the local MACE cache
       multiheads_finetuning=False (no replay data download), E0s=average (our GPAW energies have their own
       reference), per-structure config_weight from export.py (rattled rows 0.3), EMA, float64 by default
  3. writes data/ft/<run>/run.json (all arguments, recipe id, split) so the run can be reproduced

Evaluate afterwards on the held-out structures with the same table as the zero-shot models:
    python b1_benchmark.py --models ft:data/ft/<run>/<name>.model mace-omat-0 --subset data/ft/<run>/valid.extxyz
(the relaxed bases are always in train in the fold split: use --holdout-family for honest defect energies)
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--base", required=True, choices=["mp-0", "mpa-0", "omat-0"])
p.add_argument("--fold", type=int, default=0, help="-1 = train on ALL labelled structures (final model); 10 %% of the rows select the checkpoint")
p.add_argument("--nfolds", type=int, default=5)
p.add_argument("--holdout-family", default="")
p.add_argument("--train-db", default="data/train.db")
p.add_argument("--tag", default="", help="free text appended to the run name")
p.add_argument("--epochs", type=int, default=60, help="the fold-0 run plateaued after ~45 epochs")
p.add_argument("--lr", type=float, default=5e-4)
p.add_argument("--batch", type=int, default=4)
p.add_argument("--freeze", type=int, default=0, help="freeze the first N layers (0 = train everything)")
p.add_argument("--energy-weight", type=float, default=10.0, help="10 gave better energies than 1 at equal forces (held-out V2)")
p.add_argument("--forces-weight", type=float, default=100.0)
p.add_argument("--dtype", default="float64", choices=["float32", "float64"])
p.add_argument("--threads", type=int, default=8)
p.add_argument("--seed", type=int, default=1)
p.add_argument("--e0", default="shifted", choices=["shifted", "average"],
               help="shifted: foundation-model isolated-atom energies minus the fitted GPAW-vs-model offset per element "
                    "(starts at the right energy scale); average: least-squares E0 from our data (starts ~7 eV/atom off)")
p.add_argument("--keep-elements", action="store_true",
               help="keep ALL elements of the foundation model (mace_run_train --foundation_model_elements True); without it the "
                    "fine-tuned model only knows the elements of the training data (carbon) and cannot be used for Ar or N")
p.add_argument("--extra-train", nargs="*", default=[], help="further labelled databases added to the training set "
               "(e.g. data/arscan/labelled.db); a single-atom row in them gives the isolated-atom E0 of that element")
p.add_argument("--extra-fmax-weight", type=float, default=0.0,
               help="extra-train rows with a force above this (eV/A) get weight 0.3 (0 = off); suggested 15 for the N data")
p.add_argument("--all-train", action="store_true",
               help="with --fold -1: keep EVERY row in training (validation = copies of 10 %% of them) instead of letting "
                    "mace_run_train remove a random 10 %% (which can drop singleton rows such as the N2 molecule)")
p.add_argument("--extra-holdout-family", nargs="*", default=[],
               help="families of the extra databases used as validation, e.g. n-v1-2n (honest test of N generalisation)")
p.add_argument("--dry-run", action="store_true")
args = p.parse_args()

CACHE = Path.home() / ".cache" / "mace"
FOUNDATION = {"mp-0": "20231203mace128L1_epoch199model", "mpa-0": "macempa0mediummodel", "omat-0": "maceomat0mediummodel"}
split = f"hold-{args.holdout_family}" if args.holdout_family else ("all" if args.fold < 0 else f"fold{args.fold}of{args.nfolds}")
run = f"{args.base}_{split}" + (f"_freeze{args.freeze}" if args.freeze else "") + (f"_{args.tag}" if args.tag else "")
out = Path("data/ft") / run
out.mkdir(parents=True, exist_ok=True)


def shifted_e0(model_path, base):
    """E0 per element = foundation E0 - (mean model-vs-DFT energy offset per element, fitted on all labelled rows)."""
    import numpy as np
    import torch
    from ase.db import connect

    m = torch.load(model_path, map_location="cpu", weights_only=False)
    zs = [int(z) for z in m.atomic_numbers]
    e0f = np.atleast_2d(m.atomic_energies_fn.atomic_energies.detach().cpu().numpy())[0]
    found = dict(zip(zs, map(float, e0f)))
    rows = sorted(connect(args.train_db, serial=True).select(), key=lambda r: r.hash)
    cache = Path("data/b1") / f"mace-{base}.npz"
    if not cache.exists():
        sys.exit(f"{cache} missing: run  python b1_benchmark.py --models mace-{base} --no-timing  first")
    z = np.load(cache, allow_pickle=True)
    assert list(z["hash"]) == [r.hash for r in rows], "b1 cache does not match train.db: rerun the benchmark"
    elems = sorted({int(x) for r in rows for x in r.numbers})
    comp = np.array([[int((r.numbers == e).sum()) for e in elems] for r in rows], float)
    diff = np.array(z["E"], float) - np.array([r.energy for r in rows])
    off, *_ = np.linalg.lstsq(comp, diff, rcond=None)
    e0 = {e: found[e] - float(o) for e, o in zip(elems, off)}
    print("E0 foundation", {e: round(found[e], 4) for e in elems}, " fitted offset", dict(zip(elems, np.round(off, 4))))
    for dbpath in args.extra_train:                       # isolated atoms (one atom in the cell) give E0 of that element directly
        for r in connect(dbpath, serial=True).select():
            if r.natoms == 1:
                e0[int(r.numbers[0])] = float(r.energy)
                print(f"E0 of Z={int(r.numbers[0])} from the isolated-atom DFT row {dbpath}: {float(r.energy):.6f} eV")
    return "{" + ", ".join(f"{e}: {v:.6f}" for e, v in e0.items()) + "}"


# 1. split
exp = [sys.executable, "export.py", "--train", args.train_db, "--out", str(out), "--nfolds", str(args.nfolds)]
exp += ["--holdout-family", args.holdout_family] if args.holdout_family else ["--fold", str(args.fold)]
if args.extra_train:
    exp += ["--extra-train", *args.extra_train]
if args.extra_fmax_weight > 0:
    exp += ["--extra-fmax-weight", str(args.extra_fmax_weight)]
if args.all_train and args.fold < 0 and not args.holdout_family:
    exp += ["--valid-copies", "0.1"]
if args.extra_holdout_family:
    exp += ["--extra-holdout-family", *args.extra_holdout_family]
print("$", " ".join(exp))
subprocess.run(exp, check=True)

# 2. fine-tuning command
model = CACHE / FOUNDATION[args.base]
if not model.exists() and not args.dry_run:
    sys.exit(f"{model} not found: run  python b1_benchmark.py --models mace-{args.base} --no-timing  once to download it")
cmd = ["mace_run_train", "--name", run, "--seed", str(args.seed),
       "--foundation_model", str(model), "--multiheads_finetuning", "False",
       "--train_file", str((out / "train.extxyz").resolve()),
       "--energy_key", "REF_energy", "--forces_key", "REF_forces",
       "--E0s", shifted_e0(model, args.base) if args.e0 == "shifted" else "average",
       "--loss", "weighted", "--energy_weight", str(args.energy_weight), "--forces_weight", str(args.forces_weight),
       "--lr", str(args.lr), "--weight_decay", "1e-8", "--amsgrad", "--scheduler", "ReduceLROnPlateau",
       "--scheduler_patience", "8", "--ema", "--ema_decay", "0.99",
       "--batch_size", str(args.batch), "--valid_batch_size", str(args.batch),
       "--max_num_epochs", str(args.epochs), "--patience", "40",
       "--default_dtype", args.dtype, "--device", "cpu", "--error_table", "PerAtomRMSE"]
if args.fold < 0 and not args.holdout_family and args.all_train:
    cmd += ["--valid_file", str((out / "valid.extxyz").resolve())]  # all rows train; the validation file holds copies of 10 % of them
elif args.fold < 0 and not args.holdout_family:
    cmd += ["--valid_fraction", "0.1"]  # final model: 10 % random rows are REMOVED from training to pick the checkpoint
elif args.holdout_family:
    # the held-out family is a pure TEST set; checkpoint selection uses 10 % of the training rows instead
    cmd += ["--valid_fraction", "0.1", "--test_file", str((out / "valid.extxyz").resolve())]
else:
    cmd += ["--valid_file", str((out / "valid.extxyz").resolve())]
if args.freeze:
    cmd += ["--freeze", str(args.freeze)]
if args.keep_elements:
    cmd += ["--foundation_model_elements", "True"]
print("$", " ".join(cmd))
meta = dict(run=run, base=args.base, split=split, args=vars(args), command=cmd,
            summary=json.loads((out / "export_summary.json").read_text()))
(out / "run.json").write_text(json.dumps(meta, indent=2))
if args.dry_run:
    sys.exit(0)

env = dict(os.environ, OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads))
t0 = time.time()
with open(out / "train.log", "w") as log:
    proc = subprocess.run(cmd, cwd=out, env=env, stdout=log, stderr=subprocess.STDOUT)
meta["minutes"] = round((time.time() - t0) / 60, 1)
meta["returncode"] = proc.returncode
(out / "run.json").write_text(json.dumps(meta, indent=2))
print(f"finished in {meta['minutes']} min, return code {proc.returncode}; log: {out / 'train.log'}")
print("models:", [str(m) for m in out.glob("*.model")])

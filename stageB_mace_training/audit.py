"""
Homogeneity audit: recompute a random sample of train.db rows with the reference setting (600 eV, k-density 5) and
check that the recipe has a constant offset and the same forces.  Run it now and then (and before every training).

    nohup nice -n 10 mpirun -np 5 python audit.py --n 10 > audit.log 2>&1 &

Pass criteria (same as the calibration): std of the per-atom energy offset <= 1 meV/atom, force RMSE <= 0.02 eV/A,
magnetic moment within 0.3 muB.
"""
import argparse
import json
from pathlib import Path

import recipe as R

import numpy as np
from ase.db import connect
from ase.parallel import parprint, world

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--train", default="data/train.db")
p.add_argument("--out", default="data/audit.jsonl")
p.add_argument("--n", type=int, default=10)
p.add_argument("--seed", type=int, default=1)
args = p.parse_args()

OUT = Path(args.out)
rows = [r for r in connect(args.train, serial=True).select(recipe=R.RECIPE_ID)]
rng = np.random.default_rng(args.seed)
pick = [rows[i] for i in rng.permutation(len(rows))[: args.n]]
done = {json.loads(x)["hash"] for x in OUT.read_text().splitlines()} if OUT.exists() else set()
for r in pick:
    if r.hash in done:
        continue
    e, f, mu = R.compute(r.toatoms(), Path("data/logs") / f"audit_{r.hash}.txt", R.AUDIT["ecut"], R.AUDIT["kdens"])
    if world.rank == 0:
        with open(OUT, "a") as fh:
            fh.write(json.dumps(dict(hash=r.hash, family=r.family, nat=r.natoms, E=e, E0=r.energy, mu=mu, mu0=r.mu,
                                     dF=float(np.sqrt(np.mean((f - r.forces) ** 2))))) + "\n")
    world.barrier()
    parprint(f"audited {r.hash} {r.family}", flush=True)

if world.rank == 0 and OUT.exists():
    A = [json.loads(x) for x in OUT.read_text().splitlines()]
    off = np.array([(a["E"] - a["E0"]) / a["nat"] * 1000 for a in A])
    rm = float(np.sqrt(np.mean([a["dF"] ** 2 for a in A])))
    dmu = max(abs(a["mu"] - a["mu0"]) for a in A)
    ok = off.std() <= 1.0 and rm <= 0.02 and dmu <= 0.3
    print(f"\n{len(A)} audited | offset mean {off.mean():.2f} std {off.std():.2f} meV/atom | force RMSE {rm:.4f} eV/A | "
          f"max |dmu| {dmu:.2f} muB  ->  {'HOMOGENEOUS' if ok else 'CHECK THE RECIPE'}")

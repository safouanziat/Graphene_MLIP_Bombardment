"""
Spin-state check: recompute chosen train.db rows from other starting magnetic moments with the recipe and report which
solution is lowest.  The label of a row must be the lowest-energy spin state; if another start is lower by more than
--tol (meV/atom) the row is flagged and its better result is written to data/spincheck.jsonl (use `--replace` to update
train.db with it).

    mpirun -np 5 python spincheck.py --where "family=V1,mu<0.5"           # rows with a collapsed moment
    mpirun -np 5 python spincheck.py --hash 3a1f...                       # one row
"""
import argparse
import json
from pathlib import Path

import recipe as R

import numpy as np
from ase.calculators.singlepoint import SinglePointCalculator
from ase.db import connect
from ase.parallel import parprint, world

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--train", default="data/train.db")
p.add_argument("--where", default="", help='ASE db selection, e.g. "family=V1,mu<0.5"')
p.add_argument("--hash", nargs="*", default=[])
p.add_argument("--starts", nargs="+", type=float, default=[1.0, 2.0], help="initial moment per atom (muB) to try")
p.add_argument("--tol", type=float, default=2.0, help="meV/atom")
p.add_argument("--replace", action="store_true", help="overwrite the row if a lower state is found")
args = p.parse_args()

db = connect(args.train, serial=True)
rows = [r for r in db.select(args.where)] if args.where else [db.get(hash=h) for h in args.hash]
OUT = Path("data/spincheck.jsonl")
for r in rows:
    at = r.toatoms()
    best = dict(E=r.energy, mu=r.mu, f=r.forces, start=R.RECIPE["init_magmom"])
    for m in args.starts:
        a = at.copy()
        a.set_initial_magnetic_moments([m] * len(a))
        a.calc = R.make_calc(a, Path("data/logs") / f"spin_{r.hash}_{m}.txt")
        e = float(a.get_potential_energy())
        f = np.array(a.get_forces())
        mu = float(a.get_magnetic_moment()) if not R.TOY else 0.0
        d = (e - r.energy) / r.natoms * 1000
        parprint(f"{r.hash} {r.family} {r.kind}: start {m}: E - E_row = {d:+.2f} meV/atom, mu {mu:.2f} (row: {r.mu:.2f})", flush=True)
        if e < best["E"]:
            best = dict(E=e, mu=mu, f=f, start=m)
    gain = (r.energy - best["E"]) / r.natoms * 1000
    flag = gain > args.tol
    if world.rank == 0:
        with open(OUT, "a") as fh:
            fh.write(json.dumps(dict(hash=r.hash, E_row=r.energy, E_best=best["E"], gain_meV_atom=gain, mu_row=r.mu,
                                     mu_best=best["mu"], start=best["start"], flagged=flag)) + "\n")
        if flag and args.replace:
            out = at.copy()
            out.calc = SinglePointCalculator(out, energy=best["E"], forces=best["f"], magmom=best["mu"])
            kv = dict(r.key_value_pairs)
            kv.update(mu=round(best["mu"], 3), spin_start=best["start"])
            db.delete([r.id])
            db.write(out, **kv)
    world.barrier()
    parprint(f"  -> {'LOWER STATE FOUND: row was ' + format(gain, '.1f') + ' meV/atom too high' if flag else 'row is the lowest state'}", flush=True)

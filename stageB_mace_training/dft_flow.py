"""
Label every structure in data/pending.db that is not yet in data/train.db, with the recipe, as a jobflow Flow.

  - structures + labels: ASE SQLite (data/train.db), every row stamped with recipe id, cutoff, k-density, GPAW version
  - provenance (optional): if env MONGO_URI is set, one summary document per job goes to MongoDB
    (database MONGO_DB, default 'mlip'; collection 'dft_labels').  Never put the URI in code: export it in the shell.
  - resumable: a structure with a train.db row (or a failed.jsonl entry) is skipped; kill and restart at will
  - graceful stop: `touch data/STOP` ends the run after the job in progress; --max-jobs / --max-hours also work

    export OMP_NUM_THREADS=1
    nohup nice -n 10 mpirun -np 5 python dft_flow.py > flow.log 2>&1 &
"""
import argparse
import json
import os
import time
from pathlib import Path

import recipe as R                      # first: sets up GPAW / MPI

import numpy as np
from ase.calculators.singlepoint import SinglePointCalculator
from ase.db import connect
from ase.parallel import parprint, world
from jobflow import Flow, JobStore, job, run_locally
from jobflow.core.flow import JobOrder
from maggma.stores import MemoryStore

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--pending", default="data/pending.db")
p.add_argument("--train", default="data/train.db")
p.add_argument("--workdir", default="data")
p.add_argument("--max-jobs", type=int, default=10 ** 9)
p.add_argument("--max-hours", type=float, default=1e9)
p.add_argument("--dry-run", action="store_true")
args = p.parse_args()

WD = Path(args.workdir)
(WD / "logs").mkdir(parents=True, exist_ok=True)
T0 = time.time()


def stop_requested():
    return (WD / "STOP").exists() or (time.time() - T0) / 3600 > args.max_hours


def mongo_push(doc):
    uri = os.environ.get("MONGO_URI")
    if not uri or world.rank != 0:
        return
    try:
        from pymongo import MongoClient
        MongoClient(uri, serverSelectionTimeoutMS=3000)[os.environ.get("MONGO_DB", "mlip")]["dft_labels"].insert_one(doc)
    except Exception as exc:                                  # provenance is best effort, never blocks the labelling
        parprint(f"[mongo] not written: {exc}")


@job
def label(h: str):
    """DFT label of the pending structure with hash h.  Returns a small summary dict (the data go to train.db)."""
    if stop_requested():
        return {"hash": h, "status": "skipped (stop requested)"}
    row = connect(args.pending, serial=True).get(hash=h)
    at = row.toatoms()
    t0 = time.time()
    try:
        e, f, mu = R.compute(at, WD / "logs" / f"{h}.txt")
    except Exception as exc:                                  # SCF failure etc.: record, carry on with the next
        if world.rank == 0:
            with open(WD / "failed.jsonl", "a") as fh:
                fh.write(json.dumps(dict(hash=h, error=repr(exc)[:300])) + "\n")
        world.barrier()
        parprint(f"[{h}] FAILED: {exc!r}")
        return {"hash": h, "status": "failed"}
    mins = (time.time() - t0) / 60
    summary = dict(hash=h, status="ok", family=row.family, kind=row.kind, natoms=len(at), E=e, mu=mu,
                   fmax=float(np.abs(f).max()), minutes=round(mins, 2), **R.db_tags())
    if world.rank == 0:
        out = at.copy()
        out.calc = SinglePointCalculator(out, energy=e, forces=f, magmom=mu)
        connect(args.train, serial=True).write(out, hash=h, family=row.family, kind=row.kind, mu=round(mu, 3),
                                  minutes=round(mins, 2), **R.db_tags())
        mongo_push(dict(summary))
    world.barrier()
    parprint(f"[{h}] {row.family:9s} {row.kind:22s} {len(at):3d} at  E/at = {e / len(at):.4f}  mu = {mu:.2f}  {mins:.1f} min",
             flush=True)
    return {"hash": h, "status": "ok"}


def todo():
    done = {r.hash for r in connect(args.train, serial=True).select()} if Path(args.train).exists() else set()
    if (WD / "failed.jsonl").exists():
        done |= {json.loads(x)["hash"] for x in (WD / "failed.jsonl").read_text().splitlines() if x.strip()}
    return [r.hash for r in connect(args.pending, serial=True).select() if r.hash not in done]


if __name__ == "__main__":
    hs = todo()[: args.max_jobs]
    parprint(f"recipe {R.RECIPE_ID}: {len(hs)} structures to label")
    if hs and not args.dry_run:
        flow = Flow([label(h) for h in hs], name="dft_labels", order=JobOrder.LINEAR)   # fixed order: all MPI ranks must run the same job
        dbg = os.environ.get("FLOW_DEBUG") == "1"                       # FLOW_DEBUG=1: show the traceback of a job that dies outside the SCF try-block
        run_locally(flow, store=JobStore(MemoryStore()), create_folders=False, raise_immediately=dbg, log=dbg)
        parprint("flow finished")

"""
Mirror the SQLite results into MongoDB (e.g. Atlas).  SQLite stays the source of truth; this script only copies.

  collections: structures (one doc per labelled structure), spincheck, audit, failures
  idempotent : every document is upserted on a natural key, so you can rerun it after each new batch
  secret     : the connection string is read from the environment variable MONGO_URI (never from a file, never printed)

    export MONGO_URI='mongodb+srv://...'            # on the machine only; do not commit it
    python import_to_mongo.py --dry-run             # counts only, no connection
    python import_to_mongo.py                       # copy
"""
import argparse
import json
import os
import sys
from pathlib import Path

from ase.db import connect

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--train", default="data/train.db")
p.add_argument("--data", default="data")
p.add_argument("--db-name", default=os.environ.get("MONGO_DB", "graphene_mlip"))
p.add_argument("--dry-run", action="store_true")
p.add_argument("--mock", action="store_true", help="in-memory mongomock instead of a server (testing only)")
args = p.parse_args()

D = Path(args.data)


def jsonl(path):
    f = D / path
    return [json.loads(x) for x in f.read_text().splitlines() if x.strip()] if f.exists() else []


structs = []
for r in connect(args.train, serial=True).select():
    kv = dict(r.key_value_pairs)
    structs.append(dict(
        kv, hash=r.hash, recipe=kv.get("recipe"), natoms=r.natoms,
        energy=float(r.energy), energy_per_atom=float(r.energy) / r.natoms,
        symbols=r.symbols, positions=r.positions.round(6).tolist(), cell=[list(map(float, c)) for c in r.cell],
        forces=r.forces.round(6).tolist(), fmax=float(abs(r.forces).max())))
spin, audit, fails = jsonl("spincheck.jsonl"), jsonl("audit.jsonl"), jsonl("failed.jsonl")
print(f"to copy: {len(structs)} structures, {len(spin)} spin checks, {len(audit)} audit rows, {len(fails)} failures")
if args.dry_run:
    sys.exit(0)

if args.mock:
    import mongomock
    client = mongomock.MongoClient()
else:
    uri = os.environ.get("MONGO_URI")
    if not uri:
        sys.exit("MONGO_URI is not set: export it in the shell first (it is never read from a file).")
    from pymongo import MongoClient
    client = MongoClient(uri, serverSelectionTimeoutMS=8000)
    client.admin.command("ping")                       # fails early, with a clear error, if the network or IP rule is wrong
db = client[args.db_name]

for coll, docs, key in (("structures", structs, ("hash", "recipe")), ("spincheck", spin, ("hash",)),
                        ("audit", audit, ("hash",)), ("failures", fails, ("hash",))):
    if not docs:
        continue
    db[coll].create_index([(k, 1) for k in key], unique=True)
    new = upd = 0
    for d in docs:                                     # a few hundred documents: one upsert each is plenty fast
        res = db[coll].replace_one({k: d.get(k) for k in key}, d, upsert=True)
        new += res.upserted_id is not None
        upd += res.upserted_id is None and res.modified_count > 0
    print(f"{coll:10s} inserted {new}, updated {upd}, total {db[coll].count_documents({})}")
db.structures.create_index("family")
print(f"done: database '{args.db_name}'")

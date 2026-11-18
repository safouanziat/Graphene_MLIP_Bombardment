"""Progress of the labelling: python status.py"""
import json
import subprocess
from pathlib import Path

from ase.db import connect

pend = {r.hash: r for r in connect("data/pending.db", serial=True).select()}
tr = list(connect("data/train.db", serial=True).select()) if Path("data/train.db").exists() else []
done = {r.hash for r in tr}
fl = [json.loads(x) for x in Path("data/failed.jsonl").read_text().splitlines() if x.strip()] if Path("data/failed.jsonl").exists() else []
failed = {f["hash"] for f in fl}
todo = [h for h in pend if h not in done and h not in failed]
mins = [r.minutes for r in tr[1:]] or [0]
mean = sum(sorted(mins)[: int(len(mins) * 0.9) or 1]) / (int(len(mins) * 0.9) or 1)   # mean without the slowest 10 %
run = subprocess.run("pgrep -fc '[d]ft_flow.py'", shell=True, capture_output=True, text=True).stdout.strip()
print(f"labelled {len(done)} | failed {len(failed)} | still to do {len(todo)} | of {len(pend)} pending")
print(f"recipes in train.db: {sorted({r.recipe for r in tr})}")
print(f"typical time {mean:.1f} min -> about {len(todo) * mean / 60:.1f} h left | dft_flow processes: {run} (0 = not running)")
for f in fl:
    p = pend.get(f["hash"])
    print(f"FAILED {f['hash']} {p.family if p else '?'} {p.kind if p else ''}: {f['error'][:120]}")
fam = {}
for r in pend.values():
    fam.setdefault(r.family, [0, 0])[1] += 1
for r in tr:
    fam.setdefault(r.family, [0, 0])[0] += 1
print("per family (done/total): " + ", ".join(f"{k} {v[0]}/{v[1]}" for k, v in sorted(fam.items())))
lowmu = [r for r in tr if r.family in ("V1", "frenkel") and r.mu < 0.9]
print(f"V1/frenkel rows with mu < 0.9 (spin-check candidates): {len(lowmu)}")

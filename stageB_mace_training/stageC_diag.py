"""Print per-run results of the Stage C analysis for one impact site: python stageC_diag.py bridge"""
import csv
import sys

site = sys.argv[1] if len(sys.argv) > 1 else "bridge"
print(f"{'potential':8s} {'run':>3s} {'E eV':>5s} {'min height A':>13s} {'n_vacant':>8s} {'displaced':>9s} {'fate':>12s}  outcome")
for pot in ("airebo", "mace"):
    try:
        rows = list(csv.DictReader(open(f"data/stageC/{pot}/impact_analysis.csv")))
    except FileNotFoundError:
        continue
    for r in rows:
        if r["target"] == site:
            print(f"{pot:8s} {r['run']:>3s} {r['energy_eV']:>5s} {r['min_height']:>13s} {r['n_vacant']:>8s} {r['n_displaced']:>9s} {r['fate']:>12s}  {r['outcome']}")

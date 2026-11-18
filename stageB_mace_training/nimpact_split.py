"""Copy data/nimpact/labelled.db into data/nimpact/labelled_sites.db with one family per impact site
(nframe-top, nframe-bridge, nframe-hollow), so a whole site can be held out of training as an honest test.
    python nimpact_split.py
"""
import argparse
from ase.db import connect

p = argparse.ArgumentParser()
p.add_argument("--src", default="data/nimpact/labelled.db")
p.add_argument("--dst", default="data/nimpact/labelled_sites.db")
p.add_argument("--mu-min", type=float, default=-1e9)
p.add_argument("--mu-max", type=float, default=1e9)
a = p.parse_args()
dst = connect(a.dst)
have = {r.hash for r in dst.select()}
n = {}
for r in connect(a.src).select():
    if r.hash in have:
        continue
    if not (a.mu_min <= float(r.get('mu', 0)) <= a.mu_max):
        print('dropped', r.kind, 'mu', r.get('mu'))
        continue
    site = r.kind.split("_")[1]                      # N_<site>_E<energy>_r<run>_t<fs>fs
    kv = dict(r.key_value_pairs)
    kv["family"] = f"nframe-{site}"
    dst.write(r.toatoms(), **kv)
    n[site] = n.get(site, 0) + 1
print("written to", a.dst, n)

"""Short-range pair energies of a MACE model against the plain ZBL potential: where does the wall come from?  (CPU, seconds)

    python -W ignore wall_check.py --model data/ft/omat-0_all_n1/omat-0_all_n1.model

For each pair (Ar-C, N-C, C-C, N-N) a dimer in vacuum is evaluated from 0.6 to 3.0 A:
  E_mace(r)      total model energy of the dimer minus the two isolated atoms (the model's own E0s cancel)
  E_mace_noZBL   the same with the built-in ZBL repulsion of the foundation model switched off (if that works on this
                 MACE version; otherwise the column is skipped).  E_mace - E_mace_noZBL = what the built-in ZBL adds
  E_zbl          the textbook ZBL / Ziegler-Biersack-Littmark potential for the pair, for comparison
The DFT recipe has no labels below the minimum pair distances (Ar-C 1.6, C-N 1.1, N-N 1.0, C-C 1.0 A): there the model
is foundation model + ZBL + extrapolation, not your data.  The table marks those distances.
Writes wall_check.txt and wall_check.png next to the model.
"""
import argparse
import os
from pathlib import Path

import numpy as np
from ase import Atoms

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--model", required=True)
p.add_argument("--dtype", default="float64")
p.add_argument("--rmin", type=float, default=0.6)
p.add_argument("--rmax", type=float, default=3.0)
p.add_argument("--step", type=float, default=0.05)
args = p.parse_args()

import torch
from mace.calculators import MACECalculator

path = os.path.expanduser(args.model)
calc = MACECalculator(model_paths=path, device="cpu", default_dtype=args.dtype)
calc0 = None
try:                                                          # same model, built-in ZBL replaced by zeros
    class _NoZBL(torch.nn.Module):
        def forward(self, x, node_attrs, *a, **k):
            return torch.zeros(node_attrs.shape[0], dtype=x.dtype)
    calc0 = MACECalculator(model_paths=path, device="cpu", default_dtype=args.dtype)
    for m in calc0.models:
        m.pair_repulsion_fn = _NoZBL()
    t = Atoms("CC", positions=[[0, 0, 0], [0, 0, 1.4]], cell=[12] * 3, pbc=False)
    t.calc = calc0
    t.get_potential_energy()
except Exception as exc:
    print("no-ZBL variant not available on this MACE version:", repr(exc)[:150])
    calc0 = None

ZMIN = {("Ar", "C"): 1.6, ("N", "C"): 1.1, ("C", "C"): 1.0, ("N", "N"): 1.0}
Z = {"C": 6, "N": 7, "Ar": 18}


def zbl(a, b, r):
    """ZBL potential in eV, r in A."""
    z1, z2 = Z[a], Z[b]
    au = 0.46850 / (z1 ** 0.23 + z2 ** 0.23)
    x = r / au
    phi = 0.18175 * np.exp(-3.19980 * x) + 0.50986 * np.exp(-0.94229 * x) + 0.28022 * np.exp(-0.40290 * x) + 0.02817 * np.exp(-0.20162 * x)
    return 14.3996 * z1 * z2 / r * phi


def pair_energy(c, a, b, r):
    box = [14.0] * 3
    d = Atoms([a, b], positions=[[0, 0, 0], [0, 0, r]], cell=box, pbc=False)
    d.calc = c
    e = d.get_potential_energy()
    tot = e
    for s in (a, b):
        at = Atoms(s, positions=[[0, 0, 0]], cell=box, pbc=False)
        at.calc = c
        tot -= at.get_potential_energy()
    return tot


rs = np.arange(args.rmin, args.rmax + 1e-9, args.step)
lines, data = [], {}
for (a, b), zmin in ZMIN.items():
    lines.append(f"\n{a}-{b}   (DFT labels exist from {zmin} A; rows below are extrapolation, marked *)")
    lines.append(f"{'r (A)':>6s} {'E_mace':>10s} {'E_noZBL':>10s} {'E_zbl':>10s} {'mace-zbl':>10s}")
    rows = []
    for r in rs:
        em = pair_energy(calc, a, b, r)
        e0 = pair_energy(calc0, a, b, r) if calc0 is not None else float("nan")
        ez = float(zbl(a, b, r))
        rows.append((r, em, e0, ez))
        mark = "*" if r < zmin - 1e-9 else " "
        if abs(r * 10 - round(r * 10)) < 1e-6 or r < 1.0:        # print every 0.1 A (and every step below 1 A)
            lines.append(f"{r:6.2f}{mark}{em:10.3f} {e0:10.3f} {ez:10.3f} {em - ez:10.3f}")
    data[(a, b)] = np.array(rows)
text = "\n".join(lines)
print(text)
out = Path(args.model).parent
(out / "wall_check.txt").write_text(text + "\n")
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 4, figsize=(17, 3.8))
    for ax, ((a, b), arr) in zip(axs, data.items()):
        ax.semilogy(arr[:, 0], np.clip(arr[:, 1], 1e-3, None), "r-", label="MACE")
        if calc0 is not None:
            ax.semilogy(arr[:, 0], np.clip(arr[:, 2], 1e-3, None), "b--", label="MACE, no built-in ZBL")
        ax.semilogy(arr[:, 0], arr[:, 3], "k:", label="ZBL")
        ax.axvline(ZMIN[(a, b)], color="g", lw=0.8)
        ax.set_title(f"{a}-{b}  (green: shortest DFT label)")
        ax.set_xlabel("r (A)")
        ax.set_ylim(1e-2, 1e4)
    axs[0].set_ylabel("pair energy (eV, repulsive part)")
    axs[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "wall_check.png", dpi=130)
except Exception as exc:
    print("plot skipped:", exc)
print(f"\nwritten: {out / 'wall_check.txt'}, {out / 'wall_check.png'}")

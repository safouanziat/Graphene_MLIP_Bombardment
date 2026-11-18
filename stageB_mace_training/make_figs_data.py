"""Figures that need the labelled databases and a fine-tuned model (run in the working folder stageB_mace_training, env sac_multitm_env):

    python make_figs_data.py                      # model n10
    python make_figs_data.py --model data/ft/omat-0_all_n9/omat-0_all_n9.model --tag n9

Writes fig11_parity.png, fig12_error_vs_distance.png, fig13_label_coverage.png, fig14_dft_frames.png, fig15_data_composition.png
and parity_summary.txt (RMSE and MAE per set).  Test sets are whole impacts that are in no training set."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ase.db import connect

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="data/ft/omat-0_all_n10/omat-0_all_n10.model")
ap.add_argument("--tag", default="n10")
ap.add_argument("--device", default="cpu")
ap.add_argument("--out", default="figures_new")
ap.add_argument("--selftest", action="store_true", help="use ASE EMT instead of MACE (only to test the script)")
a = ap.parse_args()
out = Path(a.out); out.mkdir(exist_ok=True)

plt.rcParams.update({"font.family": ["DejaVu Sans"], "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#4a5b6b", "axes.labelcolor": "#14202b", "xtick.color": "#14202b", "ytick.color": "#14202b",
                     "text.color": "#14202b", "figure.facecolor": "white", "axes.facecolor": "white"})
INK, MUTE, BLUE, ORG, GREY, RED, TEAL = "#14202b", "#4a5b6b", "#2b5d8a", "#b8741a", "#9aa5b1", "#b23a48", "#2a8c82"
SETS = {"h34": "data/nimpact_h34/labelled.db", "late_clean": "data/nimpact_late_clean/labelled.db",
        "dense_test": "data/nimpact_dense_test/labelled.db", "val2": "data/nimpact_val2/labelled.db"}
SETCOL = {"h34": BLUE, "late_clean": ORG, "dense_test": TEAL, "val2": RED}
TRAIN = {"Ar scans": "data/arscan/labelled.db", "N scans": "data/nscan/labelled.db", "N defects": "data/ndefects/labelled.db",
         "N impacts (v6)": "data/nimpact/labelled_impacts_all_v6.db"}

if a.selftest:
    from ase.calculators.emt import EMT
    calc = EMT()
else:
    from mace.calculators import MACECalculator
    calc = MACECalculator(model_paths=a.model, device=a.device, default_dtype="float64")


def frames(path):
    out_ = []
    for r in connect(path).select():
        at = r.toatoms()
        out_.append((at, r))
    return out_


def min_nc(at):
    iN = [i for i, s in enumerate(at.symbols) if s == "N"]
    iC = [i for i, s in enumerate(at.symbols) if s == "C"]
    if not iN or not iC:
        return np.nan
    return min(at.get_distances(i, iC, mic=True).min() for i in iN)


# ---------------------------------------------------------------- predictions on the test sets
rows = []   # dict per frame
for name, p in SETS.items():
    if not Path(p).exists():
        print("missing", p); continue
    for at, r in frames(p):
        eD = float(r.energy); fD = np.array(r.forces)
        b = at.copy(); b.calc = calc
        eM = b.get_potential_energy(); fM = b.get_forces()
        iN = [i for i, s in enumerate(at.symbols) if s == "N"]
        fN_err = np.linalg.norm(fM[iN] - fD[iN], axis=1).mean() / np.sqrt(3) if iN else np.nan
        rows.append(dict(set=name, at=at, n=len(at), eD=eD, eM=eM, fD=fD, fM=fM, iN=iN, fN_err=fN_err, d=min_nc(at)))
print(len(rows), "test frames")

lines = [f"model {a.tag}"]
for name in SETS:
    R = [r for r in rows if r["set"] == name]
    if not R: continue
    fe = np.concatenate([(r["fM"] - r["fD"]).ravel() for r in R])
    ee = np.array([(r["eM"] - r["eD"]) / r["n"] * 1000 for r in R])
    nerr = np.array([r["fN_err"] for r in R])
    lines.append(f"{name:11s} frames {len(R):3d}  force RMSE all atoms {np.sqrt((fe**2).mean()):.3f} eV/A  N force error {np.sqrt(np.nanmean(nerr**2)):.3f}  "
                 f"energy MAE {np.abs(ee).mean():.1f} meV/atom  max |dE| {np.abs(ee).max():.1f}")
Path(out / "parity_summary.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))

# ---------------------------------------------------------------- fig 11: parity
fig, axs = plt.subplots(1, 3, figsize=(15, 4.8))
ax = axs[0]
fD = np.concatenate([r["fD"].ravel() for r in rows]); fM = np.concatenate([r["fM"].ravel() for r in rows])
hb = ax.hexbin(fD, fM, gridsize=60, bins="log", cmap="Blues", mincnt=1)
lim = np.abs(np.concatenate([fD, fM])).max() * 1.05
ax.plot([-lim, lim], [-lim, lim], color=INK, lw=0.8)
ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
rm = np.sqrt(((fM - fD) ** 2).mean())
ax.set_xlabel("DFT force component (eV/Å)"); ax.set_ylabel(f"MACE {a.tag} force component (eV/Å)")
ax.set_title("All atoms, all test frames", loc="left", fontsize=12, fontweight="bold")
ax.text(0.04, 0.95, f"RMSE {rm:.3f} eV/Å\n{len(fD)//3} atoms", transform=ax.transAxes, va="top", fontsize=9.5)
fig.colorbar(hb, ax=ax, label="count (log)", shrink=0.7)

ax = axs[1]
allN = []
for name in SETS:
    R = [r for r in rows if r["set"] == name and r["iN"]]
    x = np.concatenate([r["fD"][r["iN"]].ravel() for r in R]) if R else np.array([])
    y = np.concatenate([r["fM"][r["iN"]].ravel() for r in R]) if R else np.array([])
    ax.scatter(x, y, s=16, color=SETCOL[name], alpha=0.75, label=name, ec="none")
    allN.append((x, y))
xs = np.concatenate([p[0] for p in allN]); ys = np.concatenate([p[1] for p in allN])
lim = max(np.abs(xs).max(), np.abs(ys).max()) * 1.05
ax.plot([-lim, lim], [-lim, lim], color=INK, lw=0.8)
ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
ax.set_xlabel("DFT force component on the N (eV/Å)"); ax.set_ylabel("MACE force component on the N (eV/Å)")
ax.set_title("N atom, by test set", loc="left", fontsize=12, fontweight="bold")
ax.legend(frameon=False, fontsize=9, loc="lower right")

ax = axs[2]
for k, name in enumerate(SETS):
    R = [r for r in rows if r["set"] == name]
    de = np.array([(r["eM"] - r["eD"]) / r["n"] * 1000 for r in R])
    ax.scatter(np.full(len(de), k) + np.random.default_rng(0).uniform(-0.15, 0.15, len(de)), de, s=22, color=SETCOL[name], alpha=0.8, ec="none")
    if len(de): ax.hlines(de.mean(), k - 0.28, k + 0.28, color=INK, lw=1.6)
ax.axhline(0, color=INK, lw=0.8)
ax.set_xticks(range(len(SETS))); ax.set_xticklabels(list(SETS))
ax.set_ylabel("MACE − DFT energy (meV/atom)")
ax.set_title("Energy error per frame", loc="left", fontsize=12, fontweight="bold")
ax.yaxis.grid(True, color="#e3e8ee"); ax.set_axisbelow(True)
fig.suptitle(f"Held-out impact frames: MACE {a.tag} against DFT (whole impacts in no training set)", x=0.01, ha="left", fontsize=13, fontweight="bold", y=1.02)
fig.savefig(out / "fig11_parity.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- fig 12: error against N-C distance
fig, ax = plt.subplots(figsize=(8.2, 4.8))
ax.axvspan(0.5, 1.1, color="#fbe9ea", zorder=0)
for name in SETS:
    R = [r for r in rows if r["set"] == name and np.isfinite(r["d"])]
    ax.scatter([r["d"] for r in R], [r["fN_err"] for r in R], s=34, color=SETCOL[name], alpha=0.85, label=name, ec="white", lw=0.5)
ax.axvline(1.1, color=RED, ls="--", lw=1.3)
ax.text(1.12, ax.get_ylim()[1] * 0.97, "1.1 Å: shortest N-C\ndistance labelled by DFT", color=RED, fontsize=8.5, va="top")
ax.set_xlabel("minimum N-C distance in the frame (Å)"); ax.set_ylabel("force error on the N (eV/Å, norm/√3)")
ax.set_xlim(0.6, max(2.6, np.nanmax([r["d"] for r in rows]) + 0.1)); ax.set_ylim(bottom=0)
ax.legend(frameon=False, fontsize=9)
ax.set_title(f"Where the error is: {a.tag} on held-out frames", loc="left", fontsize=12.5, fontweight="bold")
ax.yaxis.grid(True, color="#e3e8ee"); ax.set_axisbelow(True)
fig.savefig(out / "fig12_error_vs_distance.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- fig 13: label coverage
trd = {}
for k, p in TRAIN.items():
    if Path(p).exists():
        trd[k] = np.array([min_nc(at) for at, _ in frames(p)])
        trd[k] = trd[k][np.isfinite(trd[k])]
trd = {k: v for k, v in trd.items() if len(v)}
td = np.array([r["d"] for r in rows if np.isfinite(r["d"])])
fig, ax = plt.subplots(figsize=(8.6, 4.6))
bins = np.linspace(0.8, 5.0, 43)
bottom = np.zeros(len(bins) - 1)
for (k, v), c in zip(trd.items(), ["#c6dbef", "#9ecae1", "#6baed6", "#2b5d8a"]):
    h, _ = np.histogram(v, bins=bins)
    ax.bar(bins[:-1], h, width=np.diff(bins), align="edge", bottom=bottom, color=c, label=f"training: {k}", edgecolor="white", lw=0.3)
    bottom += h
ht, _ = np.histogram(td, bins=bins)
ax.step(bins[:-1], ht, where="post", color=ORG, lw=2, label="test frames")
ax.axvline(1.1, color=RED, ls="--", lw=1.3)
ax.text(1.12, ax.get_ylim()[1] * 0.97, "1.1 Å", color=RED, fontsize=9, va="top")
ax.set_xlabel("minimum N-C distance in the frame (Å)"); ax.set_ylabel("number of DFT frames")
ax.set_title("What the DFT labels cover", loc="left", fontsize=12.5, fontweight="bold")
ax.legend(frameon=False, fontsize=8.5)
ax.yaxis.grid(True, color="#e3e8ee"); ax.set_axisbelow(True)
fig.savefig(out / "fig13_label_coverage.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- fig 14: gallery of DFT frames along impacts
cand = sorted([r for r in rows if r["iN"] and np.isfinite(r["d"])], key=lambda r: r["d"])
pick = [cand[int(q * (len(cand) - 1))] for q in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)] if cand else []
COL = {"C": GREY, "N": BLUE, "Ar": TEAL}
fig, axs = plt.subplots(2, len(pick), figsize=(2.7 * len(pick), 5.6), gridspec_kw={"height_ratios": [1, 0.7]})
for k, r in enumerate(pick):
    at = r["at"]; iN = r["iN"][0]; c = at.positions[iN]
    dv = at.positions - c
    from ase.geometry import find_mic
    dv, _ = find_mic(dv, at.cell, at.pbc)
    R = 4.5
    sel = [j for j in range(len(at)) if np.hypot(dv[j, 0], dv[j, 1]) < R + 0.6]
    ax, sx = axs[0, k], axs[1, k]
    for i in sel:
        for j in sel:
            if j > i and np.linalg.norm(dv[i] - dv[j]) < 1.7 and at.symbols[i] != "Ar" and at.symbols[j] != "Ar":
                ax.plot([dv[i, 0], dv[j, 0]], [dv[i, 1], dv[j, 1]], color="#7b8794", lw=1.4, zorder=1)
                sx.plot([dv[i, 0], dv[j, 0]], [dv[i, 2], dv[j, 2]], color="#7b8794", lw=1.2, zorder=1)
    for i in sel:
        s = at.symbols[i]
        ax.scatter(dv[i, 0], dv[i, 1], s=170 if s == "N" else 130, color=COL.get(s, GREY), ec="white", lw=1, zorder=3)
        sx.scatter(dv[i, 0], dv[i, 2], s=110 if s == "N" else 80, color=COL.get(s, GREY), ec="white", lw=0.8, zorder=3)
    ax.set_aspect("equal"); ax.set_xlim(-R, R); ax.set_ylim(-R, R); ax.axis("off")
    sx.set_aspect("equal"); sx.set_xlim(-R, R); sx.set_ylim(-4.5, 4.5); sx.axis("off")
    ax.set_title(f"{r['set']}\nN-C {r['d']:.2f} Å", fontsize=9.5, fontweight="bold")
    sx.text(0, -4.6, f"N force error {r['fN_err']:.2f} eV/Å", ha="center", va="top", fontsize=8.5, color=RED if r["fN_err"] > 0.4 else MUTE)
fig.suptitle("DFT-labelled test frames of N impacts, from closest to farthest approach (top view above, side view below; grey C, blue N)",
             x=0.01, ha="left", fontsize=11, fontweight="bold")
fig.savefig(out / "fig14_dft_frames.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- fig 15: data composition
cnt = {}
for k, p in {**TRAIN, **SETS}.items():
    if Path(p).exists():
        cnt[k] = connect(p).count()
fig, ax = plt.subplots(figsize=(8.2, 4.2))
names = list(cnt)
cols = [BLUE if n in TRAIN else ORG for n in names]
ax.barh(range(len(names)), [cnt[n] for n in names], color=cols)
for i, n in enumerate(names):
    ax.text(cnt[n] + 0.5, i, str(cnt[n]), va="center", fontsize=9)
ax.set_yticks(range(len(names))); ax.set_yticklabels(names); ax.invert_yaxis()
ax.set_xlabel("DFT frames")
ax.set_title("DFT frames per database (blue: training, orange: held-out tests)", loc="left", fontsize=12, fontweight="bold")
ax.xaxis.grid(True, color="#e3e8ee"); ax.set_axisbelow(True)
fig.savefig(out / "fig15_data_composition.png", dpi=200, bbox_inches="tight"); plt.close(fig)
print("figures written to", out)

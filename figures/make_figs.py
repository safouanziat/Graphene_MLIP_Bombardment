import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Ellipse
import numpy as np

plt.rcParams.update({"font.family": ["Inter", "DejaVu Sans"], "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#4a5b6b", "axes.labelcolor": "#14202b", "xtick.color": "#14202b", "ytick.color": "#14202b",
                     "text.color": "#14202b", "figure.facecolor": "white", "axes.facecolor": "white"})
INK, BLUE, BLUEF, ORG, ORGF = "#14202b", "#2b5d8a", "#eef3f8", "#b8741a", "#fff4e5"
MODELS = ["n6", "n7", "n8", "n9", "n10"]
SEQ = ["#c6dbef", "#9ecae1", "#6baed6", "#3182bd", "#08519c"]


# ---------------- fig 2: label / retrain loop ----------------
def box(ax, cx, cy, w, h, title, lines, face=BLUEF, edge=BLUE):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=face, ec=edge, lw=1.5))
    ax.text(cx, cy + h / 2 - 0.3, title, ha="center", va="center", fontsize=11, fontweight="bold")
    for i, ln in enumerate(lines):
        ax.text(cx, cy + h / 2 - 0.72 - 0.30 * i, ln, ha="center", va="center", fontsize=9)


def arrow(ax, p, q, rad=0.0, ls="-", color="#4a5b6b"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=14, lw=1.4, color=color, linestyle=ls, connectionstyle=f"arc3,rad={rad}"))


fig, ax = plt.subplots(figsize=(14, 6.6))
ax.set_xlim(0, 14); ax.set_ylim(0, 6.6); ax.axis("off")
W, H = 2.8, 1.9
xs = [1.9, 5.3, 8.7, 12.1]
y1, y2 = 4.9, 1.7
box(ax, xs[0], y1, W, H, "1  MD impacts", ["current MACE model", "whole trajectories kept", "(frames every 5 fs)"])
box(ax, xs[1], y1, W, H, "2  Select frames", ["spread over time and energy", "C-C ≥ 1.0 Å, C-N ≥ 1.1 Å", "(DFT-trusted distances)"])
box(ax, xs[2], y1, W, H, "3  Sanitise", ["remove C > 7 Å from the sheet", "or > 2.3 Å from every atom", "(free C atoms are magnetic)"], ORGF, ORG)
box(ax, xs[3], y1, W, H, "4  DFT labels", ["GPAW PBE, one recipe id", "resumable, failed jobs skipped", "dft_flow.py"])
box(ax, xs[3], y2, W, H, "5  labelled.db", ["energies + forces", "every row carries the recipe id"], ORGF, ORG)
box(ax, xs[2], y2, W, H, "6  Fine-tune", ["omat-0 base, E0s fixed", "extra weight on large forces", "finetune.py"])
box(ax, xs[1], y2, W, H, "7  Held-out tests", ["whole impacts held out", "h34, late (clean), dense, val2", "N₂ reference energy"])
box(ax, xs[0], y2, W, H, "8  Committee runs", ["n6, n7, n8, n10", "45 impacts per model", "outcome table"])
for i in range(3):
    arrow(ax, (xs[i] + W / 2 + 0.02, y1), (xs[i + 1] - W / 2 - 0.02, y1))
arrow(ax, (xs[3], y1 - H / 2 - 0.02), (xs[3], y2 + H / 2 + 0.02))
for i in (3, 2):
    arrow(ax, (xs[i] - W / 2 - 0.02, y2), (xs[i - 1] + W / 2 + 0.02, y2))
arrow(ax, (xs[1] - W / 2 - 0.02, y2), (xs[0] + W / 2 + 0.02, y2))
ax.text((xs[0] + xs[1]) / 2, y2 - H / 2 - 0.3, "\u25b2 passes: committee runs", ha="center", fontsize=8.5, color=BLUE, fontweight="bold")
arrow(ax, (xs[1] - 0.9, y2 + H / 2 + 0.02), (xs[0] + 0.9, y1 - H / 2 - 0.02), ls="--", color=ORG)
ax.text(4.9, 3.3, "fails: new frames from the failing regime", ha="left", va="center", fontsize=9, color=ORG, fontweight="bold")
ax.set_title("Label → fine-tune → test loop", loc="left", fontsize=13, fontweight="bold", pad=6)
fig.savefig("fig2_loop.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# ---------------- fig 3: test errors ----------------
sets = ["h34\n(17 frames)", "late_clean\n(4)", "dense_test\n(10, hollow)", "val2\n(10)"]
err = {"n6": [0.464, 0.272, 0.718, 1.080], "n7": [0.514, 0.373, 0.186, 0.362], "n8": [0.511, 0.349, 0.208, 0.346],
       "n9": [0.429, 0.229, 0.213, 0.363], "n10": [0.426, 0.232, 0.236, 0.349]}
fig, ax = plt.subplots(figsize=(10, 4.8))
w = 0.16
x = np.arange(len(sets))
for k, m in enumerate(MODELS):
    b = ax.bar(x + (k - 2) * w, err[m], w, color=SEQ[k], edgecolor="#08306b" if m == "n10" else "none", linewidth=1.0, label=m)
    for xi, v in zip(x + (k - 2) * w, err[m]):
        ax.text(xi, v + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=7.5, color=INK)
ax.set_xticks(x); ax.set_xticklabels(sets)
ax.set_ylabel("force error on the N atom (eV/Å)")
ax.set_ylim(0, 1.2)
ax.legend(title="model (training data grows left to right)", ncol=5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.0), fontsize=9, title_fontsize=9)
ax.set_title("Held-out impact frames: error against DFT, by model", loc="left", fontsize=13, fontweight="bold", pad=24)
ax.yaxis.grid(True, color="#e3e8ee"); ax.set_axisbelow(True)
fig.text(0.01, -0.02, "norm / √3 convention; whole impacts held out; the near-contact top/bridge regime stays at 0.4-0.5 eV/Å for all models "
         "(0.15-0.3 of it is DFT spin ambiguity). late_clean = 4 frames not in any training set.", fontsize=8, color="#4a5b6b")
fig.savefig("fig3_test_errors.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# ---------------- fig 4: n10 outcome map ----------------
cases = [("top 20 eV", 0, 5, 0), ("top 30 eV", 0, 5, 0), ("top 50 eV", 0, 5, 0),
         ("bridge 20 eV", 2, 3, 0), ("bridge 30 eV", 0, 4, 1), ("bridge 50 eV", 0, 1, 4),
         ("hollow 20 eV", 1, 4, 0), ("hollow 30 eV", 0, 0, 5), ("hollow 50 eV", 0, 0, 5)]
col = {"reflected": "#8da0b3", "bound": "#e08a1e", "transmitted": "#1f9e89"}
fig, ax = plt.subplots(figsize=(9, 4.8))
for i, (lab, r, b, t) in enumerate(cases):
    left = 0
    for name, n in (("reflected", r), ("bound", b), ("transmitted", t)):
        if n:
            ax.barh(i, n, left=left, color=col[name], edgecolor="white", linewidth=1.5, height=0.72)
            ax.text(left + n / 2, i, str(n), ha="center", va="center", color="white", fontweight="bold", fontsize=10)
            left += n
ax.set_yticks(range(len(cases))); ax.set_yticklabels([c[0] for c in cases]); ax.invert_yaxis()
ax.set_xlim(0, 5); ax.set_xticks([]); ax.spines["bottom"].set_visible(False); ax.spines["left"].set_visible(False); ax.tick_params(left=False)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=col[k], label=k) for k in col], ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.02))
ax.set_title("N impacts on graphene, model n10: outcome of 5 runs per case", loc="left", fontsize=13, fontweight="bold", pad=10)
fig.text(0.01, -0.08, "top: always bound in-plane (3 C), 1 C ejected.  hollow: transmits from 30 eV.  bridge at 20-30 eV depends on the model (see table).", fontsize=8, color="#4a5b6b")
fig.savefig("fig4_outcomes_n10.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# ---------------- fig 5: N2 reference error history ----------------
names = ["n2", "n3", "n4", "n5", "n6", "n7", "n8", "n9", "n10"]
vals = [4.5, 42, 14, 38, 37, 9.2, 2.4, 56.9, 7.5]
fig, ax = plt.subplots(figsize=(8.5, 4.2))
cols = ["#b7c3cf"] * 8 + ["#08519c"]
bars = ax.bar(names, vals, color=cols)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v:g}", ha="center", fontsize=9)
ax.set_ylabel("N₂ reference error (meV/atom)")
ax.set_ylim(0, 68)
ax.yaxis.grid(True, color="#e3e8ee"); ax.set_axisbelow(True)
ax.annotate("single N₂ row in the training set:\nthe error jumps between retrains", xy=(7, 56.9), xytext=(2.1, 60), fontsize=9, ha="left", va="center",
            arrowprops=dict(arrowstyle="-", color="#4a5b6b", lw=0.8))
ax.annotate("9 DFT N₂ frames added", xy=(8, 12), xytext=(8, 24), fontsize=9, ha="center", color="#08519c", fontweight="bold",
            arrowprops=dict(arrowstyle="-|>", color="#08519c", lw=1.0))
ax.set_title("The N₂ reference energy was unstable until it had more than one row", loc="left", fontsize=12.5, fontweight="bold", pad=8)
fig.savefig("fig5_n2_reference.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# ---------------- fig 6: AIREBO SW minima ----------------
runs = [("vacuum 10 Å, seed 2026", 5.8103, 1.82), ("seed 4", 5.8086, 1.82), ("seed 5", 5.8086, 1.82), ("seed 7", 5.8094, 1.82),
        ("seed 2", 6.0152, 1.20), ("seed 3", 6.0152, 1.20), ("seed 6", 6.0155, 1.22), ("seed 8", 6.0158, 1.24),
        ("vacuum 15 Å, seed 2026", 6.0225, 1.29), ("seed 1", 6.0271, 1.37), ("vacuum 20 Å, seed 1", 6.0271, 1.37)]
fig, ax = plt.subplots(figsize=(9, 4.8))
for i, (lab, e, z) in enumerate(runs):
    c = "#08519c" if z > 1.5 else "#e08a1e"
    ax.plot([e], [i], "o", color=c, markersize=8)
    ax.text(6.075, i, f"z spread {z:.2f} Å", va="center", fontsize=8.5, color="#4a5b6b")
ax.axvline(5.5397, color="#1f9e89", lw=2)
ax.text(5.565, -1.2, "DFT 5.54 eV (flat, z spread 0.04 \u00c5)", color="#1f9e89", ha="left", va="center", fontsize=9, fontweight="bold")
ax.set_yticks(range(len(runs))); ax.set_yticklabels([r[0] for r in runs], fontsize=9); ax.invert_yaxis()
ax.set_xlim(5.4, 6.3); ax.set_ylim(len(runs) - 0.5, -1.9)
ax.set_xlabel("Stone-Wales formation energy with AIREBO (eV), 5×5 cell")
ax.xaxis.grid(True, color="#e3e8ee"); ax.set_axisbelow(True)
ax.set_title("AIREBO relaxes the SW defect into two different rippled minima", loc="left", fontsize=12.5, fontweight="bold", pad=8)
fig.savefig("fig6_sw_minima.png", dpi=200, bbox_inches="tight"); plt.close(fig)
print("done")

# ---------------- fig 1: stages ----------------
fig, ax = plt.subplots(figsize=(14, 3.4))
ax.set_xlim(0, 14); ax.set_ylim(0, 3.4); ax.axis("off")
stages = [("Stage A", ["Can AIREBO be trusted?", "V1, V2, SW, adatom, Frenkel", "AIREBO vs GPAW (PBE)"]),
          ("Stage B", ["DFT labels + MACE fine-tune", "omat-0 base", "one fixed DFT recipe"]),
          ("Stage C", ["Paired Ar impacts", "AIREBO+ZBL vs fine-tuned MACE", "same small sheet"]),
          ("Stage D", ["N impacts (no AIREBO for N)", "top / bridge / hollow, 20-50 eV", "DFT frames as ground truth"])]
xs4 = [1.9, 5.3, 8.7, 12.1]
for cx, (t, ls) in zip(xs4, stages):
    box(ax, cx, 1.7, 2.8, 1.9, t, ls)
for i in range(3):
    arrow(ax, (xs4[i] + 1.42, 1.7), (xs4[i + 1] - 1.42, 1.7))
ax.set_title("Graphene_MLIP_Bombardment: four stages", loc="left", fontsize=13, fontweight="bold", pad=6)
fig.savefig("fig1_pipeline.png", dpi=200, bbox_inches="tight"); plt.close(fig)

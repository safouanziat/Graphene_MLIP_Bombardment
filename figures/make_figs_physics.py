"""Figures that need no data files: impact sites, collision kinematics, Stage A errors, N defect structures.
    python make_figs_physics.py      (needs numpy, matplotlib, ase)
Writes fig7_sites.png, fig8_kinematics.png, fig9_stageA_errors.png, fig10_n_defects.png"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, RegularPolygon
import numpy as np
from ase.build import graphene

plt.rcParams.update({"font.family": ["DejaVu Sans"], "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#4a5b6b", "axes.labelcolor": "#14202b", "xtick.color": "#14202b", "ytick.color": "#14202b",
                     "text.color": "#14202b", "figure.facecolor": "white", "axes.facecolor": "white"})
INK, MUTE, BLUE, ORG, GREY, RED = "#14202b", "#4a5b6b", "#2b5d8a", "#b8741a", "#9aa5b1", "#b23a48"
DCC = 1.42   # C-C bond length (A)

# ------------------------------------------------------------------ fig 7: impact sites
fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1, 1.25]})
a1, a2 = np.array([2.46, 0.0]), np.array([1.23, 2.13])
atoms = []
for i in range(-3, 4):
    for j in range(-3, 4):
        c0 = i * a1 + j * a2
        for k in range(6):
            ang = np.deg2rad(30 + 60 * k)
            atoms.append(c0 + DCC * np.array([np.cos(ang), np.sin(ang)]))
pts = np.unique(np.round(np.array(atoms), 3), axis=0)
for c in pts:
    for d in pts:
        if 0.1 < np.linalg.norm(c - d) < 1.6:
            ax.plot([c[0], d[0]], [c[1], d[1]], color=GREY, lw=1.6, zorder=1)
ax.scatter(pts[:, 0], pts[:, 1], s=120, color="#d5dbe2", ec=GREY, zorder=2)
cen = np.array([0.0, 0.0])                         # hollow = centre of a hexagon
top = DCC * np.array([0.0, 1.0])                   # top = on a C atom (vertex at 90 deg)
brg = 0.5 * (DCC * np.array([np.cos(np.deg2rad(30)), np.sin(np.deg2rad(30))]) + DCC * np.array([np.cos(np.deg2rad(-30)), np.sin(np.deg2rad(-30))]))
ax.add_patch(Circle(cen, DCC, fill=False, ls="--", ec=BLUE, lw=1.2, zorder=3))
for pt, lab, col, dx, dy in [(top, "top", RED, 0.0, 0.55), (brg, "bridge", ORG, 0.15, -0.55), (cen, "hollow", BLUE, 0.0, -0.55)]:
    ax.scatter(*pt, s=210, color=col, zorder=4, ec="white", lw=1.5)
    ax.text(pt[0] + dx, pt[1] + dy, lab, color=col, fontsize=11, fontweight="bold", ha="center", va="center", zorder=5,
            bbox=dict(fc="white", ec="none", pad=1.0, alpha=0.85))
ax.set_aspect("equal"); ax.set_xlim(-4.2, 4.2); ax.set_ylim(-3.5, 3.4); ax.axis("off")
ax.set_title("Impact sites on the sheet", loc="left", fontsize=12.5, fontweight="bold")
ax.text(0.5, -0.04, f"distance from the impact point to the nearest C: top 0, bridge {DCC/2:.2f}, hollow {DCC:.2f} Å", transform=ax.transAxes, ha="center", va="top", fontsize=8.5, color=MUTE)

rows = [("top, 20-50 eV", 0.85, 1.04, RED), ("bridge, 50 eV", 0.90, 1.06, ORG), ("bridge, 20-30 eV", 1.08, 1.17, ORG),
        ("hollow, 20 eV", 1.14, 1.54, BLUE), ("hollow, 30-50 eV", 1.27, 1.40, BLUE)]
bx.axvspan(0.6, 1.1, color="#fbe9ea", zorder=0)
for k, (lab, lo, hi, col) in enumerate(rows):
    y = len(rows) - 1 - k
    bx.plot([lo, hi], [y, y], color=col, lw=7, solid_capstyle="round", zorder=3)
    bx.text(hi + 0.04, y, f"{lo:.2f}-{hi:.2f}", va="center", fontsize=8.5, color=MUTE)
bx.axvline(1.1, color=RED, lw=1.4, ls="--")
bx.text(1.1, len(rows) - 0.35, "1.1 Å: shortest N-C distance\nlabelled by DFT", color=RED, fontsize=8.5, ha="center", va="bottom")
bx.text(0.65, -0.95, "below this the model is\nnot checked by DFT", color=RED, fontsize=8.5, va="center")
bx.set_yticks(range(len(rows))); bx.set_yticklabels([r[0] for r in rows][::-1])
bx.set_xlim(0.6, 1.8); bx.set_ylim(-1.4, len(rows) + 0.6)
bx.set_xlabel("minimum N-C distance reached in the impact (Å)")
bx.set_title("How close the N gets, and what DFT checked", loc="left", fontsize=12.5, fontweight="bold")
bx.yaxis.grid(True, color="#e3e8ee"); bx.set_axisbelow(True)
fig.savefig("fig7_sites.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# ------------------------------------------------------------------ fig 8: kinematics
mC = 12.011
gam = lambda m: 4 * m * mC / (m + mC) ** 2
fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.5, 4.4))
m = np.linspace(1, 100, 400)
ax.plot(m, gam(m), color=BLUE, lw=2)
for lab, mm in [("N", 14.007), ("Ar", 39.948)]:
    ax.scatter(mm, gam(mm), s=70, color=ORG, zorder=3)
    ax.annotate(f"{lab}: {gam(mm):.2f}", (mm, gam(mm)), xytext=(8, -16 if lab == "N" else 8), textcoords="offset points", fontsize=10, fontweight="bold")
ax.axvline(mC, color=GREY, ls=":"); ax.text(mC + 1, 0.12, "C", color=MUTE)
ax.set_xlabel("projectile mass (u)"); ax.set_ylabel("largest share of the energy given to a C atom")
ax.set_ylim(0, 1.05); ax.set_title("Head-on collision with a carbon atom", loc="left", fontsize=12.5, fontweight="bold")
ax.yaxis.grid(True, color="#e3e8ee"); ax.set_axisbelow(True)

x = np.arange(2)
w = 0.34
Td = {"MACE": 20.5, "AIREBO": 25.5}
for k, (nm, td) in enumerate(Td.items()):
    vals = [td / gam(39.948), td / gam(14.007)]
    col = BLUE if nm == "MACE" else ORG
    bars = bx.bar(x + (k - 0.5) * w, vals, w, color=col, label=f"displacement threshold {td} eV ({nm})")
    for xi, v in zip(x + (k - 0.5) * w, vals):
        bx.text(xi, v + 0.8, f"{v:.0f}", ha="center", fontsize=9)
bx.plot([-0.5 * w - w / 2, -0.5 * w + w / 2], [50, 50], color=BLUE, lw=2.5)
bx.plot([0.5 * w - w / 2, 0.5 * w + w / 2], [52.5, 52.5], color=ORG, lw=2.5)
bx.annotate("MD threshold for Ar vacancies:\nAIREBO 50-55 eV, MACE below 50 eV", xy=(0.5 * w, 52.5), xytext=(0.55, 44), fontsize=9, ha="left",
            arrowprops=dict(arrowstyle="-", color=MUTE, lw=0.8))
bx.set_xticks(x); bx.set_xticklabels(["Ar on top site\n(vacancy)", "N on top site\n(substitution)"])
bx.set_ylabel("projectile energy needed (eV)"); bx.set_ylim(0, 82)
bx.legend(frameon=False, fontsize=8.5, loc="upper center", ncol=1)
bx.set_title("Binary-collision estimate of the threshold", loc="left", fontsize=12.5, fontweight="bold")
bx.yaxis.grid(True, color="#e3e8ee"); bx.set_axisbelow(True)
fig.text(0.01, -0.03, "bars: E = Td / gamma, gamma = 4 m1 m2 / (m1 + m2)^2 (head-on, two-body). Horizontal marks: MD threshold for Ar vacancies. "
         "The ordering agrees; the absolute value of the MD threshold is higher than the estimate.", fontsize=8, color=MUTE)
fig.savefig("fig8_kinematics.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# ------------------------------------------------------------------ fig 9: Stage A errors
d = [("Monovacancy\nV1", -0.32), ("Stone-Wales", 0.17), ("Divacancy\nV2", 0.67), ("Frenkel pair", 0.93), ("C adatom", -2.23)]
fig, ax = plt.subplots(figsize=(8.2, 4.2))
for i, (lab, v) in enumerate(d):
    ax.bar(i, v, 0.55, color=BLUE if v > 0 else ORG)
    ax.text(i, v + (0.08 if v > 0 else -0.08), f"{v:+.2f}", ha="center", va="bottom" if v > 0 else "top", fontsize=10, fontweight="bold")
ax.axhline(0, color=INK, lw=1)
ax.set_xticks(range(len(d))); ax.set_xticklabels([x[0] for x in d])
ax.set_ylabel("AIREBO - DFT formation energy (eV)"); ax.set_ylim(-2.8, 1.5)
ax.text(-0.45, 1.25, "above 0: AIREBO too costly", ha="left", color=BLUE, fontsize=9)
ax.text(-0.45, -1.0, "below 0: AIREBO too cheap", ha="left", color=ORG, fontsize=9)
ax.set_title("Stage A: AIREBO defect energies against GPAW PBE (4×4 cell)", loc="left", fontsize=12.5, fontweight="bold")
ax.yaxis.grid(True, color="#e3e8ee"); ax.set_axisbelow(True)
fig.savefig("fig9_stageA_errors.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# ------------------------------------------------------------------ fig 10: N defect structures (idealised, unrelaxed)
from ase.geometry import find_mic
def base():
    g = graphene(formula="C2", a=2.46, size=(6, 6, 1), vacuum=3.0)
    g.center(axis=2)
    return g
def nbrs(at, i, cut=1.7):
    dv, dd = find_mic(at.positions - at.positions[i], at.cell, at.pbc)
    return [j for j in range(len(at)) if j != i and dd[j] < cut]
def undercoord(at):
    return [i for i in range(len(at)) if len(nbrs(at, i)) == 2]
g0 = base()
i0 = int(np.argmin(np.linalg.norm(g0.positions - g0.get_center_of_mass(), axis=1)))
n0 = nbrs(g0, i0)
structs = []
a = g0.copy(); a[i0].symbol = "N"; structs.append(("graphitic N", a, [], -0.23, a.positions[i0]))
a = g0.copy(); mid = 0.5 * (a.positions[i0] + a.positions[n0[0]]); a.append("N"); a.positions[-1] = mid + np.array([0, 0, 1.45])
structs.append(("N adatom, bridge", a, [], -0.50, mid))
vac = [g0.positions[i0].copy()]
a = g0[[k for k in range(len(g0)) if k != i0]]; uc = undercoord(a)
for k in uc: a[k].symbol = "N"
structs.append((f"V1 + {len(uc)}N (pyridinic-like)", a, vac, +0.31, vac[0]))
rem = [i0, n0[0]]; vac = [g0.positions[k].copy() for k in rem]
a = g0[[k for k in range(len(g0)) if k not in rem]]; uc = undercoord(a)
for k in uc: a[k].symbol = "N"
structs.append((f"V2 + {len(uc)}N", a, vac, +0.43, 0.5 * (vac[0] + vac[1])))
COL = {"C": GREY, "N": BLUE}
R = 4.3
fig, axs = plt.subplots(2, len(structs), figsize=(3.4 * len(structs), 6.8), gridspec_kw={"height_ratios": [1, 0.42]})
for k, (name, at, vac, err, c) in enumerate(structs):
    ax, sx = axs[0, k], axs[1, k]
    dv, dd = find_mic(at.positions - c, at.cell, at.pbc)
    sel = [j for j in range(len(at)) if np.hypot(dv[j, 0], dv[j, 1]) < R + 0.8]
    for i in sel:
        for j in nbrs(at, i):
            if j in sel and j > i:
                ax.plot([dv[i, 0], dv[j, 0]], [dv[i, 1], dv[j, 1]], color="#7b8794", lw=1.8, zorder=1)
                sx.plot([dv[i, 0], dv[j, 0]], [dv[i, 2], dv[j, 2]], color="#7b8794", lw=1.5, zorder=1)
    for i in sel:
        col = COL[at.symbols[i]]; sz = 330 if at.symbols[i] == "N" else 250
        ax.scatter(dv[i, 0], dv[i, 1], s=sz, color=col, ec="white", lw=1.2, zorder=3)
        sx.scatter(dv[i, 0], dv[i, 2], s=sz * 0.6, color=col, ec="white", lw=1.0, zorder=3)
    for v in vac:
        dvv, _ = find_mic((v - c)[None, :], at.cell, at.pbc)
        ax.add_patch(Circle((dvv[0, 0], dvv[0, 1]), 0.7, fill=False, ls="--", ec=RED, lw=1.5, zorder=4))
    ax.set_aspect("equal"); ax.set_xlim(-R, R); ax.set_ylim(-R, R); ax.axis("off")
    ax.set_title(name, fontsize=10.5, fontweight="bold")
    sx.set_aspect("equal"); sx.set_xlim(-R, R); sx.set_ylim(-1.0, 2.2); sx.axis("off")
    sx.text(0, -1.35, f"MACE - DFT: {err:+.2f} eV", ha="center", va="top", fontsize=10, fontweight="bold", color=RED if abs(err) > 0.25 else MUTE)
fig.suptitle("N in the sheet: the structures behind the formation-energy test (idealised, unrelaxed; grey C, blue N, dashed = removed C; below: side view)",
             x=0.01, ha="left", fontsize=11, fontweight="bold")
fig.savefig("fig10_n_defects.png", dpi=200, bbox_inches="tight"); plt.close(fig)
print("done")

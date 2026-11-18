"""
Draw the EXACT architecture of a MACE model file: every box is a module that really exists in the loaded model, with its
irreps, tensor-product paths and parameter count read from the file (nothing is typed in by hand).

    python plot_architecture.py                              # the three foundation models in ~/.cache/mace
    python plot_architecture.py --models ft=data/ft/omat-0_all/omat-0_all.model --out docs/architecture

per model it writes   <out>/<name>.png          the diagram
                      <out>/<name>_spec.json    what was extracted (the diagram is drawn from this file only)
                      <out>/<name>_modules.txt  full str(model), for reference
and   python plot_architecture.py --from-spec docs/architecture/<name>_spec.json   redraws without torch.
"""
import argparse
import json
import os
import re
from pathlib import Path

DEFAULT = {"MACE-MP-0": "20231203mace128L1_epoch199model", "MACE-MPA-0": "macempa0mediummodel",
           "MACE-OMAT-0": "maceomat0mediummodel"}


def npar(m):
    return int(sum(p.numel() for p in m.parameters()))


def first_line(m, n=90):
    s = str(m).splitlines()[0]
    return s if len(s) <= n else s[: n - 1] + "…"


def children(m):
    return [dict(name=n, kind=type(c).__name__, text=first_line(c), params=npar(c)) for n, c in m.named_children()]


def mlp_dims(m):
    """e3nn FullyConnectedNet prints as FullyConnectedNet[10, 64, 64, 64, 512]"""
    r = re.search(r"\[([0-9, ]+)\]", str(m).splitlines()[0])
    return [int(x) for x in r.group(1).split(",")] if r else []


def safe(f, default=None):
    try:
        return f()
    except Exception:
        return default


def extract(model, name):
    spec = dict(name=name, r_max=float(model.r_max), params=npar(model),
                n_species=safe(lambda: len(model.atomic_numbers)), n_layers=len(model.interactions))
    spec["embedding"] = dict(kind=type(model.node_embedding).__name__, text=first_line(model.node_embedding),
                             params=npar(model.node_embedding))
    re_ = model.radial_embedding
    spec["radial"] = dict(kind=type(re_).__name__, params=npar(re_),
                          n_basis=safe(lambda: len(re_.bessel_fn.bessel_weights)),
                          cutoff_p=safe(lambda: int(re_.cutoff_fn.p)), text=first_line(re_))
    spec["sh"] = dict(kind=type(model.spherical_harmonics).__name__,
                      irreps=safe(lambda: str(model.interactions[0].conv_tp.irreps_in2)),
                      text=first_line(model.spherical_harmonics))
    spec["layers"] = []
    for i, (b, p, r) in enumerate(zip(model.interactions, model.products, model.readouts)):
        L = dict(index=i, block=dict(kind=type(b).__name__, params=npar(b), children=children(b),
                                     irreps_in=safe(lambda: str(b.conv_tp.irreps_in1)),
                                     irreps_out=safe(lambda: str(b.conv_tp.irreps_out)),
                                     radial_mlp=safe(lambda: mlp_dims(b.conv_tp_weights), [])),
                 product=dict(kind=type(p).__name__, params=npar(p), children=children(p),
                              correlation=safe(lambda: int(p.symmetric_contractions.contractions[0].correlation)),
                              text=first_line(p)),
                 readout=dict(kind=type(r).__name__, params=npar(r), text=first_line(r), children=children(r)))
        spec["layers"].append(L)
    spec["scale_shift"] = safe(lambda: dict(scale=float(model.scale_shift.scale), shift=float(model.scale_shift.shift)))
    spec["e0"] = safe(lambda: [float(x) for x in model.atomic_energies_fn.atomic_energies.flatten()[:8]])
    return spec


# ---------------------------------------------------------------------------------------------------------- drawing
def compact(ir):
    """'128x0e+128x1o+128x2e+128x3o' -> '128×(0e 1o 2e 3o)';  repeated irreps are summed first"""
    if not ir:
        return str(ir)
    tot = {}
    for t in ir.split("+"):
        m = re.match(r"\s*(\d+)x(\w+)", t)
        if not m:
            return ir
        tot[m.group(2)] = tot.get(m.group(2), 0) + int(m.group(1))
    if len(set(tot.values())) == 1:
        return f"{next(iter(tot.values()))}×(" + " ".join(tot) + ")"
    return " + ".join(f"{v}×{k}" for k, v in tot.items())


def ttl(c):
    t = f"{c['name']}  ({c['kind']})"
    return t if len(t) <= 38 else c["name"]


def clip(t, n):
    return t if len(t) <= n else t[: n - 1] + "…"


def draw(spec, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    nl = len(spec["layers"])
    rows = max(len(l["block"]["children"]) + len(l["product"]["children"]) for l in spec["layers"])
    H = 1.0 + 0.56 * rows + 3 * 0.85 + 0.6 + 1.5
    W = 5.3 * nl + 2.6
    fig, ax = plt.subplots(figsize=(W, H))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    fmt = lambda n: f"{n:,}".replace(",", " ")

    def box(x, y, w, h, title, sub, color, fs=8.6):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=color, ec="#555", lw=0.9))
        ax.text(x + 0.08, y + h - 0.13, title, fontsize=fs, weight="bold", va="top")
        if sub:
            ax.text(x + 0.08, y + 0.07, sub, fontsize=fs - 1.4, va="bottom", color="#333")

    def arrow(x1, y1, x2, y2, c="#333"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=10, color=c, lw=1))

    ax.text(0.1, H - 0.2, f"{spec['name']}   |   {fmt(spec['params'])} parameters   |   r_cut {spec['r_max']:g} Å   |   "
            f"{spec['n_layers']} interaction layers   |   {spec['n_species']} element channels in the embedding",
            fontsize=11.5, weight="bold", va="top")
    # left column: input and embedding
    top = H - 1.0
    bh = 0.78
    e = spec["embedding"]
    box(0.1, top - bh, 2.2, bh, "Input", "positions, atomic numbers Z\nneighbour list within r_cut", "#eef")
    box(0.1, top - 2.3 * bh, 2.2, bh, f"Node embedding", f"{e['kind']}\n{fmt(e['params'])} params", "#dfe")
    arrow(1.2, top - bh, 1.2, top - 1.3 * bh)
    r = spec["radial"]
    box(0.1, top - 3.6 * bh, 2.2, bh * 1.1, "Radial basis (bonds)",
        f"{r.get('n_basis')} Bessel fns × polynomial\ncutoff p={r.get('cutoff_p')} → feeds every layer", "#eef")
    s = spec["sh"]
    box(0.1, top - 4.9 * bh, 2.2, bh * 1.1, "Spherical harmonics", f"{s.get('irreps')}\n(bond directions)", "#eef")
    # layers
    for l in spec["layers"]:
        x0 = 2.6 + 5.3 * l["index"]
        w = 5.0
        y = top
        b = l["block"]
        ax.text(x0, y + 0.1, f"Layer {l['index'] + 1}", fontsize=11, weight="bold")
        y -= 0.1
        box(x0, y - 0.8, w, 0.8, f"Interaction: {b['kind']}", f"in {compact(b['irreps_in'])}  →  out {compact(b['irreps_out'])}\n{fmt(b['params'])} params", "#fed", 8.6)
        y -= 0.8
        for c in b["children"]:
            sub = c["text"]
            if c["name"] == "conv_tp_weights" and b.get("radial_mlp"):
                sub = "radial MLP " + " → ".join(map(str, b["radial_mlp"]))
            box(x0 + 0.25, y - 0.55, w - 0.25, 0.5, ttl(c), clip(sub, 56), "#fff3e8", 8.2)
            ax.text(x0 + w - 0.08, y - 0.14, f"{fmt(c['params'])} params", fontsize=7.6, ha="right", va="top", color="#333")
            y -= 0.56
        p = l["product"]
        y -= 0.1
        box(x0, y - 0.8, w, 0.8, f"Product basis: {p['kind']}", f"body order {None if p['correlation'] is None else p['correlation'] + 1} (correlation {p['correlation']})\n{fmt(p['params'])} params", "#ffe9a8", 8.6)
        y -= 0.8
        for c in p["children"]:
            box(x0 + 0.25, y - 0.55, w - 0.25, 0.5, ttl(c), clip(c["text"], 56), "#fff6d8", 8.2)
            ax.text(x0 + w - 0.08, y - 0.14, f"{fmt(c['params'])} params", fontsize=7.6, ha="right", va="top", color="#333")
            y -= 0.56
        ro = l["readout"]
        y -= 0.1
        box(x0, y - 0.8, w, 0.8, f"Readout: {ro['kind']}", f"→ one number per atom (atomic energy contribution)\n{fmt(ro['params'])} params", "#dfe", 8.6)
        ybot = min(globals().get("_ybot", 1e9), y - 0.8)
        globals()["_ybot"] = ybot
        if l["index"] < nl - 1:
            arrow(x0 + w, top - 0.4, x0 + w + 0.3, top - 0.4)
    arrow(2.3, top - 1.65 * bh, 2.6, top - 0.4)
    # footer
    ss = spec.get("scale_shift") or {}
    sum_txt = ("Σ readouts of all layers  →  atomic energy;  × scale " + f"{ss['scale']:.4g}" + "  + shift " + f"{ss['shift']:.4g}"
               if ss else "Σ readouts of all layers → atomic energy")
    box(2.6, 0.2, W - 3.0, 0.75, "Total energy  E = Σ_atoms [ scale · (Σ layer readouts) + shift + E0(Z) ]",
        sum_txt + "     |     forces  F = −∂E/∂r  (automatic differentiation)", "#ffd", 9)
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="*", default=[], help="name=path; default: the three foundation models")
    ap.add_argument("--out", default="docs/architecture")
    ap.add_argument("--from-spec", nargs="*", default=[], help="redraw from *_spec.json (no torch needed)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for f in a.from_spec:
        sp = json.loads(Path(f).read_text())
        draw(sp, out / f"{sp['name']}.png")
        print("redrawn", out / f"{sp['name']}.png")
    if a.from_spec and not a.models:
        raise SystemExit
    models = dict(s.split("=", 1) for s in a.models) if a.models else {k: f"~/.cache/mace/{v}" for k, v in DEFAULT.items()}
    import torch
    for name, path in models.items():
        m = torch.load(os.path.expanduser(path), map_location="cpu", weights_only=False)
        sp = extract(m, name)
        (out / f"{name}_spec.json").write_text(json.dumps(sp, indent=1))
        (out / f"{name}_modules.txt").write_text(str(m))
        draw(sp, out / f"{name}.png")
        print(f"{name}: {sp['params']:,} parameters -> {out / (name + '.png')}")

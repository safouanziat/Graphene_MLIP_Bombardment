"""N impacts on a graphene sheet with the fine-tuned MACE model (no classical reference: AIREBO has no nitrogen), and the DFT frame check.

    python -W ignore nimpact.py generate --model data/ft/omat-0_all_n2/omat-0_all_n2.model
    export OMP_NUM_THREADS=1
    nohup nice -n 10 mpirun -np 5 python dft_flow.py --pending data/nimpact/pending.db --train data/nimpact/labelled.db --workdir data/nimpact > nimpact_dft.log 2>&1 &
    python -W ignore nimpact.py compare --model data/ft/omat-0_all_n2/omat-0_all_n2.model

generate: one N atom (default 20, 30, 50 eV) falls on a 4x4 sheet (32 C) at top / bridge / hollow, 0 K start, MACE via ASE (VelocityVerlet,
0.2 fs, float32), 300 fs, --runs impacts per condition (random shift of the impact point, --jitter).  For every impact it prints the outcome
(reflected / bound in-plane / adatom / transmitted / ...), the closest N-C distance and the number of ejected C.  A budget of --max-dft frames is
then chosen evenly over all impacts (frames shortly before, at and after contact; recipe rules C-N >= 1.1, C-C >= 1.0 A; the N is parked at
--park A above the plane when it is further; C atoms further than --park A from the sheet or --iso A from every other atom are removed (free C atoms are magnetic in DFT), so the deformed sheet after the impact is tested too), rescaled to the DFT lattice and written to
data/nimpact/pending.db (family 'nframe', initial moments C 0.3 / N 3.0 muB as in the N scans).  Full trajectories: data/nimpact/traj/<tag>.extxyz.
compare: DFT vs MACE on the labelled frames: force errors overall, on the N, on the C atoms within 3 A of the N, and energy differences between
frames of the same impact.  Frames closer than the recipe minimum are not part of the test (no trusted DFT): say so when quoting results.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from ase import Atoms, units
from ase.build import graphene
from ase.data import atomic_masses, atomic_numbers
from ase.db import connect
from ase.io import read, write

import recipe as R

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("mode", choices=["generate", "compare"])
p.add_argument("--model", required=True)
p.add_argument("--dir", default="data/nimpact")
p.add_argument("--src", default="../stageA_dft_check/dft_cheap/relaxed")
p.add_argument("--n", type=int, default=4)
p.add_argument("--proj", default="N")
p.add_argument("--energies", nargs="+", type=float, default=[20.0, 30.0, 50.0])
p.add_argument("--sites", nargs="+", default=["top", "bridge", "hollow"], choices=["top", "bridge", "hollow"])
p.add_argument("--runs", type=int, default=2)
p.add_argument("--run-start", type=int, default=0, help="first run index (the run index seeds the random impact point): use a new range for new, independent impacts")
p.add_argument("--jitter", type=float, default=0.15)
p.add_argument("--fs", type=float, default=300.0)
p.add_argument("--dump-fs", type=float, default=5.0)
p.add_argument("--height", type=float, default=6.0)
p.add_argument("--window", type=float, default=150.0, help="frames up to this long after first N-C contact (< 3 A) are candidates")
p.add_argument("--park", type=float, default=7.0)
p.add_argument("--iso", type=float, default=2.3, help="C atoms further than this from every other atom are removed from DFT frames")
p.add_argument("--frames", type=int, default=6, help="candidate frames kept per impact")
p.add_argument("--post", type=float, nargs="*", default=None, metavar="FS",
               help="POST-CONTACT mode: instead of approach/closest/aftermath, take the frame at each of these delays (fs) after the closest allowed approach, e.g. --post 20 60 120; every run of every condition contributes all of them")
p.add_argument("--max-dft", type=int, default=36, help="total number of frames written for DFT")
p.add_argument("--a0-mace", type=float, default=None)
p.add_argument("--threads", type=int, default=8)
p.add_argument("--dtype", default="float64", help="compare only; the dynamics always use float32")
args = p.parse_args()
D = Path(args.dir)
MODEL = os.path.expanduser(args.model)


def get_calc(dtype):
    import torch
    torch.set_num_threads(args.threads)
    from mace.calculators import MACECalculator
    return MACECalculator(model_paths=MODEL, device="cpu", default_dtype=dtype)


def site_xy(sheet, mode):
    cen = (sheet.cell[0, :2] + sheet.cell[1, :2]) / 2
    i = int(np.argmin(np.linalg.norm(sheet.positions[:, :2] - cen, axis=1)))
    if mode == "top":
        return sheet.positions[i, :2].copy()
    d = sheet.get_distances(i, range(len(sheet)), mic=True)
    d[i] = 9e9
    nb = [int(k) for k in np.argsort(d)[:2]]
    if mode == "bridge":
        return sheet.positions[i, :2] + sheet.get_distance(i, nb[0], mic=True, vector=True)[:2] / 2
    v = sum(sheet.get_distance(i, k, mic=True, vector=True) for k in nb)   # hexagon centre
    return sheet.positions[i, :2] + v[:2]


def to_dft_frame(fr, a_dft, a_src):
    at = fr.copy()
    at.calc = None
    cell = np.array(at.cell)
    sc = a_dft / a_src
    at.positions[:, :2] *= sc
    cell[0] *= sc
    cell[1] *= sc
    at.set_cell(cell, scale_atoms=False)
    nc = len(at) - 1
    zpl = float(np.median(at.positions[:nc, 2]))
    zn = at.positions[-1, 2] - zpl
    if abs(zn) > args.park:
        at.positions[-1, 2] = zpl + np.sign(zn) * args.park
    # free C atoms are magnetic in DFT and break the SCF: REMOVE far / isolated C (do not park them)
    iso = getattr(args, "iso", 2.3)
    drop = []
    for k in range(nc):
        d = at.get_distances(k, [j for j in range(len(at)) if j != k], mic=True)
        if d.min() > iso or abs(at.positions[k, 2] - zpl) > args.park:
            drop.append(k)
    at = at[[k for k in range(len(at)) if k not in drop]]
    nc = len(at) - 1
    at.wrap()
    at = R.normalise(at)
    at.set_initial_magnetic_moments([0.3] * nc + [3.0])
    return at, zn


def run_impact(calc, a_src, site, energy, run):
    from ase.md.verlet import VelocityVerlet
    from ase.optimize import FIRE
    rng = np.random.default_rng(10 * run + int(energy) + 1000 * ["top", "bridge", "hollow"].index(site))
    sh = graphene(a=a_src, size=(args.n, args.n, 1), vacuum=10.0)
    sh.pbc = (True, True, False)
    sh.calc = calc
    FIRE(sh, logfile=None).run(fmax=0.01, steps=200)
    at = sh.copy()
    at.calc = None
    xy = site_xy(at, site) + rng.uniform(-args.jitter, args.jitter, 2)
    z0 = at.positions[:, 2].mean()
    at += Atoms(args.proj, positions=[[xy[0], xy[1], z0 + args.height]])
    at.calc = calc
    v = np.zeros((len(at), 3))
    v[-1, 2] = -np.sqrt(2 * energy / atomic_masses[atomic_numbers[args.proj]])
    at.set_velocities(v)
    dt = 0.2
    dyn = VelocityVerlet(at, timestep=dt * units.fs)
    every = int(round(args.dump_fs / dt))
    frames = []
    for s in range(int(round(args.fs / dt)) + 1):
        if s % every == 0:
            f = at.copy()
            f.calc = None
            frames.append(f)
        dyn.run(1)
    return frames


def outcome(frames):
    last, prev = frames[-1], frames[-2]
    n = len(last) - 1
    zpl = float(np.median(last.positions[:n, 2]))
    h = last.positions[n, 2] - zpl
    vz = (last.positions[n, 2] - prev.positions[n, 2]) / args.dump_fs
    d = last.get_distances(n, range(n), mic=True)
    nb = int((d < 1.9).sum())
    ejected = int((np.abs(last.positions[:n, 2] - zpl) > 3.0).sum())
    dmin = min(float(f.get_distances(n, range(n), mic=True).min()) for f in frames)
    if h > 3.5 and vz > 0:
        kind = "reflected"
    elif h < -3.5:
        kind = "transmitted"
    elif nb >= 1 and abs(h) < 0.7:
        kind = f"bound in-plane ({nb} C)"
    elif nb >= 1:
        kind = f"bound {'above' if h > 0 else 'below'} the plane ({nb} C)"
    else:
        kind = f"free near the sheet (h={h:.1f})"
    return kind, h, nb, ejected, dmin


if args.mode == "generate":
    D.mkdir(parents=True, exist_ok=True)
    (D / "traj").mkdir(exist_ok=True)
    f = Path(args.src) / "gpaw_pristine_n4.extxyz"
    a_dft = float(np.linalg.norm(read(f).cell[0])) / args.n if f.exists() else 2.4704
    a_m = args.a0_mace
    if a_m is None:
        j = Path("data/stageC/mace_thermal/a0.json")
        a_m = json.loads(j.read_text())["a0"] if j.exists() else 2.47
    calc = get_calc("float32")
    summary, per_impact = [], {}
    print(f"MACE lattice constant {a_m:.4f} A, DFT {a_dft:.4f} A; {len(args.sites) * len(args.energies) * args.runs} impacts")
    print(f"{'impact':22s} {'outcome':34s} {'dmin N-C':>8s} {'ejected C':>9s}")
    for site in args.sites:
        for e in args.energies:
            for run in range(args.run_start, args.run_start + args.runs):
                tag = f"N_{site}_E{e:g}_r{run}"
                frames = run_impact(calc, a_m, site, e, run)
                write(D / "traj" / f"{tag}.extxyz", frames)
                kind, h, nb, ej, dmin = outcome(frames)
                print(f"{tag:22s} {kind:34s} {dmin:8.2f} {ej:9d}", flush=True)
                summary.append(dict(tag=tag, site=site, E=e, run=run, outcome=kind, height=round(h, 2), nbonds=nb, ejected=ej, dmin=round(dmin, 3)))
                # candidate frames of this impact: only frames with the N within 3 A of a C atom and allowed by the recipe
                cand = []
                for k, fr in enumerate(frames):
                    at, zn = to_dft_frame(fr, a_dft, a_m)
                    ok, margin = R.worst_pair(at)
                    dd = float(at.get_distances(len(at) - 1, range(len(at) - 1), mic=True).min())
                    if ok and dd < 3.0:
                        cand.append((k, at, ok, margin, dd, zn))
                if not cand:
                    continue
                # the three frames that matter: approach (N-C closest to 2 A before the minimum), the closest ALLOWED approach, aftermath (+60 fs)
                kmin = min(range(len(cand)), key=lambda i: cand[i][4])
                pre = [i for i in range(kmin) if cand[i][4] > cand[kmin][4] + 0.3]
                pre = min(pre, key=lambda i: abs(cand[i][4] - 2.0)) if pre else None
                post = [i for i in range(kmin + 1, len(cand)) if cand[i][0] * args.dump_fs >= cand[kmin][0] * args.dump_fs + 60]
                post = post[0] if post else (len(cand) - 1 if len(cand) - 1 > kmin else None)
                sel = [i for i in (pre, kmin, post) if i is not None]
                if args.post:
                    t0 = cand[kmin][0] * args.dump_fs
                    sel = []
                    for off in args.post:
                        later = [i for i in range(kmin + 1, len(cand)) if cand[i][0] * args.dump_fs >= t0 + off - 1e-9]
                        if later and later[0] not in sel:
                            sel.append(later[0])
                per_impact[tag] = [cand[i] for i in sel]
    (D / "impacts_summary.json").write_text(json.dumps(summary, indent=1))
    # budget: run 0 of every condition first (all its frames), then the other runs (closest-approach frame only)
    order = []
    if args.post:                                    # post-contact mode: every run contributes all its frames, round-robin so a cap keeps the spread
        for rank in range(len(args.post)):
            order += [(tag, lst[rank]) for tag, lst in per_impact.items() if rank < len(lst)]
    for tag, lst in ([] if args.post else per_impact.items()):
        if tag.endswith("_r0"):
            order += [(tag, c) for c in lst]
    for tag, lst in ([] if args.post else per_impact.items()):
        if not tag.endswith("_r0"):
            order += [(tag, lst[min(1, len(lst) - 1)])]
    seen, pdb = set(), connect(D / "pending.db")
    have = {r.hash for r in pdb.select()}
    new, allf = 0, []
    for tag, (k, at, ok, margin, dd, zn) in order:
        if len(allf) >= args.max_dft:
            break
        h = R.structure_hash(at)
        if h in seen:
            continue
        seen.add(h)
        t = k * args.dump_fs
        kind = f"{tag}_t{t:.0f}fs"
        allf.append(at)
        if h not in have:
            pdb.write(at, hash=h, family="nframe", kind=kind, margin=round(margin, 3), t_fs=t, nc_min=round(dd, 3), n_height=round(zn, 2))
            have.add(h)
            new += 1
        print(f"    DFT frame {kind:34s} N-C min {dd:5.2f} A  N height {zn:6.2f} A  margin {margin:5.2f}")
    write(D / "geometries.extxyz", allf)
    print(f"\n{len(allf)} frames chosen, {new} new rows in {D}/pending.db")
    print("next:\n  export OMP_NUM_THREADS=1\n  nohup nice -n 10 mpirun -np 5 python dft_flow.py --pending "
          f"{D}/pending.db --train {D}/labelled.db --workdir {D} > nimpact_dft.log 2>&1 &")
    raise SystemExit

# ---- compare
calc = get_calc(args.dtype)
ldb = connect(D / "labelled.db")
rows = list(ldb.select())
print(f"{len(rows)} labelled frames")
res = []
for r in rows:
    at = r.toatoms()
    at.calc = calc
    e, f = float(at.get_potential_energy()), np.array(at.get_forces())
    res.append((r, e, f))
n_of = lambda r: len(r.numbers) - 1
allerr, nerr, nearerr = [], [], []
by = {}
print(f"\n{'frame':34s} {'N-C':>5s} {'|F_N| dft':>9s} {'dF on N':>8s} {'F rms err':>9s}")
for r, e, f in res:
    n = n_of(r)
    err = f - np.array(r.forces)
    at = r.toatoms()
    d = at.get_distances(n, range(n), mic=True)
    near = np.where(d < 3.0)[0]
    allerr.append(err.ravel())
    nerr.append(err[n])
    if len(near):
        nearerr.append(err[near].ravel())
    tag = r.kind.rsplit("_t", 1)[0]
    by.setdefault(tag, []).append((float(r.get("t_fs", 0)), e - float(r.energy)))
    print(f"{r.kind:34s} {d.min():5.2f} {np.linalg.norm(np.array(r.forces)[n]):9.2f} {np.linalg.norm(err[n]):8.2f} {np.sqrt(np.mean(err ** 2)):9.2f}")
rms = lambda l: float(np.sqrt(np.mean(np.concatenate([np.ravel(x) for x in l]) ** 2))) if l else float("nan")
dE = []
for tag, lst in by.items():
    lst.sort()
    dE += [x[1] - lst[0][1] for x in lst[1:]]
dft_rms = float(np.sqrt(np.mean(np.concatenate([np.array(r.forces).ravel() for r, _, _ in res]) ** 2)))
print(f"\nDFT force rms over all frames: {dft_rms:.2f} eV/A")
print(f"MACE force error rms: all atoms {rms(allerr):.3f}   on the N {rms(nerr):.3f}   C within 3 A of the N {rms(nearerr):.3f} eV/A")
print(f"energy-difference error within an impact (vs first frame): rms {rms(dE):.3f} eV  ({len(dE)} differences)")

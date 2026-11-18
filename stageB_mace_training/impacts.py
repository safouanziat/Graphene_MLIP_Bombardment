"""
Impact test: do the models give the right forces/energies on Ar-on-graphene configurations (far from equilibrium)?

  generate  short AIREBO+ZBL Ar impacts on a SMALL periodic sheet (same LAMMPS setup as the MD pipeline: pair_style hybrid
            airebo 3.0 zbl 4.0 5.0), keep a few snapshots per impact (Ar close to the sheet, recipe distance rules met),
            write them as unlabelled rows to data/impacts/pending.db (family 'impact')  -> DFT single points
  (label)   python dft_flow.py --pending data/impacts/pending.db --train data/impacts/labelled.db --workdir data/impacts
  report    DFT vs models (and vs AIREBO+ZBL itself) on those snapshots: forces overall, on the Ar atom, on C atoms near Ar,
            and energy differences between snapshots of the same impact

The small sheet gets hot quickly (it is only a test set, not a physical impact); snapshots are taken early. Impacts closer
than the recipe limit (Ar-C 1.6 A, the ZBL regime where PAW labels are not trusted) are skipped, so the hardest
close-contact frames are NOT part of this test. Say so when quoting results.

    python impacts.py generate --airebo ../stageA_dft_check/CH.airebo --energies 30 70 --targets top bridge
    python impacts.py report --models omat=data/ft/omat-0_all/omat-0_all.model mpa=data/ft/mpa-0_all/mpa-0_all.model

MLIP_TOY=1: no LAMMPS; a straight-line Ar descent and EMT stand-ins test the bookkeeping only (projectile N).
"""
import argparse
import os
import shlex
import subprocess
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.build import graphene
from ase.data import atomic_masses, atomic_numbers
from ase.db import connect
from ase.io import read, write
from ase import units

import recipe as R

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("mode", choices=["generate", "report"])
p.add_argument("--dir", default="data/impacts")
p.add_argument("--src", default="../stageA_dft_check/dft_cheap/relaxed")
p.add_argument("--n", type=int, default=4, help="sheet size n x n unit cells (4 -> 32 C atoms + Ar)")
p.add_argument("--proj", default="Ar")
p.add_argument("--energies", nargs="+", type=float, default=[30.0, 70.0])
p.add_argument("--targets", nargs="+", default=["top", "bridge"], choices=["top", "bridge", "hollow"])
p.add_argument("--a0", type=float, default=None, help="lattice constant (A); default DFT 4x4 value. The MD repo uses 2.419 (AIREBO)")
p.add_argument("--frames", type=int, default=4, help="snapshots kept per impact")
p.add_argument("--height", type=float, default=6.0)
p.add_argument("--timestep", type=float, default=0.01, help="fs, as in the MD pipeline")
p.add_argument("--steps", type=int, default=12000)
p.add_argument("--dump-every", type=int, default=400)
p.add_argument("--max-height", type=float, default=7.0, help="keep snapshots with Ar at most this far above the sheet")
p.add_argument("--max-dmin", type=float, default=3.5, help="keep snapshots with Ar-C distance at most this")
p.add_argument("--airebo", default="CH.airebo")
p.add_argument("--lmp", default=os.environ.get("LMP_CMD", "lmp"))
p.add_argument("--models", nargs="*", default=[], help="name=path of MACE models (report)")
p.add_argument("--dtype", default="float64")
p.add_argument("--threads", type=int, default=8)
args = p.parse_args()
D = Path(args.dir)
SEL = D / "selected.extxyz"


def lattice_constant():
    if args.a0:
        return args.a0
    f = Path(args.src) / "gpaw_pristine_n4.extxyz"
    return 2.46 if (R.TOY or not f.exists()) else float(np.linalg.norm(read(f).cell[0])) / 4


def target_site(sheet, mode):
    c = sheet.cell
    centre = np.array([c[0, 0] / 2 + c[1, 0] / 2, c[1, 1] / 2])
    i = int(np.argmin(np.linalg.norm(sheet.positions[:, :2] - centre, axis=1)))
    if mode == "top":
        return sheet.positions[i]
    d = sheet.get_distances(i, range(len(sheet)), mic=True)
    d[i] = 9e9
    if mode == "hollow":                       # centre of a hexagon: atom + the two bond vectors to two of its neighbours
        nb = [int(k) for k in np.argsort(d)[:2]]
        v = sum(sheet.get_distance(i, k, mic=True, vector=True) for k in nb)
        return sheet.positions[i] + v
    return 0.5 * (sheet.positions[i] + sheet.positions[int(np.argmin(d))])


def write_input(path, e_kin, pid, data, dump):
    z = atomic_numbers[args.proj]
    vz = -np.sqrt(2.0 * e_kin / atomic_masses[z]) * units.fs * 1000.0           # A/ps (metal units), downwards
    dt = args.timestep * 1e-3
    lines = [f"# {args.proj} {e_kin:g} eV", "units metal", "boundary p p m", "atom_style charge", f"read_data {data}",
             f"mass 1 {atomic_masses[6]:.4f}", f"mass 2 {atomic_masses[z]:.4f}",
             "pair_style hybrid airebo 3.0 zbl 4.0 5.0", f"pair_coeff * * airebo {os.path.abspath(args.airebo)} C NULL",
             f"pair_coeff 2 2 zbl {z} {z}", f"pair_coeff 1 2 zbl 6.0 {z}",
             "neighbor 2.0 bin", "neigh_modify every 1 delay 0 check yes",
             f"group proj id {pid}", f"velocity proj set 0.0 0.0 {vz:.8f} units box", f"timestep {dt:.8g}", "fix integ all nve",
             f"fix pe all print {args.dump_every} \"$(step) $(pe:%.8f)\" file pe.txt screen no",
             "thermo 2000", "thermo_style custom step temp pe ke etotal",
             f"dump traj all custom {args.dump_every} {dump} id type x y z fx fy fz", "dump_modify traj sort id",
             f"run {args.steps}"]
    Path(path).write_text("\n".join(lines) + "\n")


def run_impact(sheet, mode, e_kin, tag):
    """returns list of frames (Atoms; info['airebo_energy'], arrays['airebo_forces'] when available)"""
    at = sheet.copy()
    site = target_site(at, mode)
    z0 = at.positions[:, 2].mean()
    at.append(Atoms(args.proj, positions=[[site[0], site[1], z0 + args.height]])[0])
    if R.TOY:                                                             # straight descent, no dynamics
        frames = []
        for k in range(args.steps // args.dump_every + 1):
            f = at.copy()
            f.positions[-1, 2] = z0 + args.height - 0.12 * k
            frames.append(f)
        return frames
    rd = D / "runs" / tag
    rd.mkdir(parents=True, exist_ok=True)
    at.set_initial_charges(np.zeros(len(at)))
    write(rd / "system.data", at, format="lammps-data", specorder=["C", args.proj], atom_style="charge")
    write_input(rd / "in.impact", e_kin, len(at), "system.data", "traj.dump")
    r = subprocess.run(shlex.split(args.lmp) + ["-in", "in.impact", "-log", "log.lammps", "-screen", "none"], cwd=rd,
                       capture_output=True, text=True)
    if r.returncode != 0 or not (rd / "traj.dump").exists():
        raise RuntimeError(f"LAMMPS failed for {tag}: " + (r.stderr[-800:] or "see " + str(rd / "log.lammps")))
    frames = read(rd / "traj.dump", format="lammps-dump-text", index=":", specorder=["C", args.proj])
    pe = {}
    if (rd / "pe.txt").exists():
        for line in (rd / "pe.txt").read_text().splitlines():
            q = line.split()
            if len(q) == 2:
                try:
                    pe[int(q[0])] = float(q[1])
                except ValueError:
                    pass
    out = []
    for i, f in enumerate(frames):
        try:
            f.arrays["airebo_forces"] = f.get_forces()
        except Exception:
            pass
        step = i * args.dump_every
        if step in pe:
            f.info["airebo_energy"] = pe[step]
        f.calc = None
        out.append(f)
    return out


def snapshot_info(f):
    n = len(f) - 1
    c = np.mean(f.positions[:n, 2])
    d = f.get_distances(n, range(n), mic=True)
    return float(f.positions[n, 2] - c), float(d.min())


if args.mode == "generate":
    D.mkdir(parents=True, exist_ok=True)
    a = lattice_constant()
    sheet = graphene(a=a, size=(args.n, args.n, 1), vacuum=10.0)
    sheet.pbc = True
    pdb = connect(D / "pending.db")
    have = {r.hash for r in pdb.select()}
    kept, summary = [], []
    for mode in args.targets:
        for e in args.energies:
            tag = f"{args.proj}_{mode}_E{int(e)}"
            frames = run_impact(sheet, mode, e, tag)
            cand = []
            for i, f in enumerate(frames):
                h, dmin = snapshot_info(f)
                nf = R.normalise(f)
                ok, margin = R.worst_pair(nf)
                if h <= args.max_height and dmin <= args.max_dmin and ok:
                    cand.append((i, nf, dmin, margin))
            rej_close = sum(1 for f in frames if snapshot_info(f)[1] < R.RECIPE["min_dist"].get(f"Ar-C", 1.6))
            pick = sorted({int(round(x)) for x in np.linspace(0, len(cand) - 1, min(args.frames, len(cand)))}) if cand else []
            for j in pick:
                i, nf, dmin, margin = cand[j]
                t = i * args.dump_every * args.timestep
                h = R.structure_hash(nf)
                if h not in have:
                    pdb.write(nf, hash=h, family="impact", kind=f"{mode}_E{int(e)}_t{t:.0f}fs", margin=round(margin, 3))
                    have.add(h)
                nf.info.update(name=f"{tag}_t{t:.0f}fs", hash=h, t_fs=t, dmin=dmin, e_kin=e, target=mode)
                kept.append(nf)
            summary.append(f"  {tag}: {len(frames)} frames, {len(cand)} usable, {len(pick)} kept, "
                           f"{rej_close} skipped as closer than the Ar-C limit")
    write(SEL, kept)
    print("\n".join(summary))
    print(f"\n{len(kept)} snapshots -> {SEL}, {pdb.count()} rows in {D}/pending.db\nnext:\n  python dft_flow.py --pending "
          f"{D}/pending.db --train {D}/labelled.db --workdir {D}")

else:
    from ase.calculators.emt import EMT
    geoms = read(SEL, ":")
    labp = D / "labelled.db"
    lab = {r.hash: r for r in connect(labp, serial=True).select()} if labp.exists() else {}
    geoms = [g for g in geoms if g.info["hash"] in lab]
    print(f"{len(geoms)} labelled snapshots")
    if not geoms:
        raise SystemExit
    methods = {}
    for s in args.models:
        k, v = s.split("=", 1)
        methods[k] = v
    if any("airebo_energy" in g.info for g in geoms):
        methods["airebo+zbl"] = None
    if R.TOY:
        methods = {"toy": None}

    def evaluate(name, spec):
        if name == "airebo+zbl":
            return ([g.info["airebo_energy"] for g in geoms], [g.arrays["airebo_forces"] for g in geoms])
        if R.TOY:
            calc = EMT()
        else:
            import torch
            torch.set_num_threads(args.threads)
            from mace.calculators import MACECalculator
            calc = MACECalculator(model_paths=os.path.expanduser(spec), device="cpu", default_dtype=args.dtype)
        E, F = [], []
        for g in geoms:
            a = g.copy()
            a.calc = calc
            E.append(float(a.get_potential_energy()))
            F.append(np.array(a.get_forces()))
        return E, F

    dft_E = [float(lab[g.info["hash"]].energy) for g in geoms]
    dft_F = [np.array(lab[g.info["hash"]].forces) for g in geoms]
    by_impact = {}
    for i, g in enumerate(geoms):
        by_impact.setdefault((g.info["target"], g.info["e_kin"]), []).append(i)

    rows = {}
    skipped, store = {}, {}
    for name, spec in methods.items():
        try:
            E, F = evaluate(name, spec)
        except ValueError as exc:                       # e.g. "18 is not in list": the model's element table has no Ar
            skipped[name] = str(exc)
            print(f"  {name}: cannot evaluate these snapshots ({exc}); skipped")
            continue
        allf, arf, nearf, farf, dE = [], [], [], [], []
        for i, g in enumerate(geoms):
            n = len(g) - 1
            err = F[i] - dft_F[i]
            d = g.get_distances(n, range(n), mic=True)
            near = np.where(d < 3.0)[0]
            far = np.where(d >= 3.0)[0]
            allf.append(err.ravel())
            arf.append(err[n])
            if len(near):
                nearf.append(err[near].ravel())
            if len(far):
                farf.append(err[far].ravel())
        for idx in by_impact.values():
            idx = sorted(idx, key=lambda i: geoms[i].info["t_fs"])
            for i in idx[1:]:
                dE.append((E[i] - E[idx[0]]) - (dft_E[i] - dft_E[idx[0]]))
        store[name] = (E, F)
        rms = lambda l: float(np.sqrt(np.mean(np.concatenate([np.ravel(x) for x in l])**2))) if l else float("nan")
        rows[name] = (rms(allf), rms(arf), rms(nearf), rms(farf), rms(dE) if dE else float("nan"))
    rms_dft = float(np.sqrt(np.mean(np.concatenate([f.ravel() for f in dft_F]) ** 2)))
    print(f"\nDFT force rms over all snapshots: {rms_dft:.2f} eV/A")
    print(f"{'method':14s} {'F all':>8s} {'F on Ar':>9s} {'C <3A of Ar':>12s} {'C >=3A':>8s} {'dE rms (eV)':>12s}")
    for k, v in rows.items():
        print(f"{k:14s} " + " ".join(f"{x:{w}.3f}" for x, w in zip(v, (8, 9, 12, 8, 12))))
    for k, why in skipped.items():
        print(f"{k:14s} not evaluated: {why}")
    print("\nper snapshot: |F_Ar| from DFT, then the force error ON THE Ar and the rms force error over all atoms, per method (eV/A)")
    names = list(store)
    print(f"{'snapshot':22s} {'ArC':>5s} {'|F_Ar|':>7s} " + " ".join(f"{k[:10]:>10s}(Ar|all)" for k in names))
    for i, g in enumerate(geoms):
        n = len(g) - 1
        dar = g.get_distances(n, range(n), mic=True).min()
        row = f"{g.info['target'] + '_E' + str(int(g.info['e_kin'])) + '_t' + str(int(g.info['t_fs'])):22s} {dar:5.2f} {np.linalg.norm(dft_F[i][n]):7.2f} "
        for k in names:
            err = store[k][1][i] - dft_F[i]
            row += f" {np.linalg.norm(err[n]):6.2f}|{np.sqrt(np.mean(err ** 2)):4.2f}       "
        print(row)
    print("\n(F columns: RMS force error vs DFT in eV/A; dE: error of the energy difference to the first snapshot of the same impact)")

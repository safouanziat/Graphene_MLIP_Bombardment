"""Stage C pilot: the same Ar impacts on a small graphene sheet with AIREBO+ZBL (LAMMPS) and with the Ar-aware MACE (ASE).

Both potentials follow the same protocol, each on ITS OWN relaxed and thermalised sheet (AIREBO a0 = 2.419 A, MACE a0 fitted here):
  thermalise (Langevin/NVT, 300 K)  ->  Ar placed --height above the sheet  ->  NVE impact (--nve-fs)  ->
  Langevin quench of the sheet (--quench-ps)  ->  minimisation  ->  graphene_defective_<tag>.xyz
The folder layout is the one of 02_defect_engineering_stats.py, so 03_analyze_impacts.py classifies both sets unchanged.
Pairing between the potentials is by case (impact site type + energy + run number), not by trajectory.

    python -W ignore stageC_pilot.py airebo --airebo ../stageA_dft_check/CH.airebo --runs 3 --parallel 4
    python -W ignore stageC_pilot.py mace --model data/ft/omat-0_all_ar/omat-0_all_ar.model --threads 8
    python -W ignore stageC_pilot.py mace --model ... --dt 0.1 --cases hollow:70 --outroot data/stageC/mace_dt01   # dt check
    python -W ignore stageC_pilot.py analyse --md-repo ../../graphene-defect-engineering-md
Workflow test without GPAW/MACE (EMT, C projectile):  MLIP_TOY=1 python stageC_pilot.py mace --proj C --n 3 --nve-fs 20 ...
"""
import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from ase import Atoms, units
from ase.build import graphene
from ase.data import atomic_masses, atomic_numbers
from ase.io import read, write

TOY = os.environ.get("MLIP_TOY") == "1"
A0_AIREBO = 2.419
ROOT = Path("data/stageC")


# ----------------------------------------------------------------------------------------------- geometry
def perfect_sheet(n, a0):
    sh = graphene(a=a0, size=(n, n, 1), vacuum=10.0)
    sh.pbc = (True, True, False)
    return sh


def site_xy(sheet, mode):
    """xy of 'top' / 'bridge' / 'hollow' around the atom nearest to the cell centre (same rule for both potentials)."""
    c = (sheet.cell[0, :2] + sheet.cell[1, :2]) / 2
    i = int(np.argmin(np.linalg.norm(sheet.positions[:, :2] - c, axis=1)))
    d = sheet.get_distances(i, range(len(sheet)), mic=True)
    d[i] = 9e9
    nb = [int(k) for k in np.argsort(d)[:3]]
    if mode == "top":
        return sheet.positions[i, :2].copy()
    v = [sheet.get_distance(i, k, mic=True, vector=True)[:2] for k in nb]
    if mode == "bridge":
        return sheet.positions[i, :2] + v[0] / 2
    # hollow: centre of a hexagon = atom + two of its bond vectors
    return sheet.positions[i, :2] + v[0] + v[1]


def parse_cases(items):
    out = []
    for s in items:
        m, e = s.split(":")
        out.append((m, float(e)))
    return out


def tag_of(proj, mode, e, run):
    return f"{proj}_{mode}_E{e:g}_run{run}"


def seeds_for(proj, e, mode, run, base=2026):
    ss = np.random.SeedSequence([base, atomic_numbers[proj], int(round(e * 10)), ["top", "bridge", "hollow"].index(mode), run])
    r = np.random.default_rng(ss)
    return dict(velocity=int(r.integers(1, 8e8)), langevin=int(r.integers(1, 8e8)))


def vz_down(proj, e_kin):
    """A/fs"""
    return -np.sqrt(2 * e_kin / atomic_masses[atomic_numbers[proj]]) * units.fs


# ----------------------------------------------------------------------------------------------- AIREBO (LAMMPS)
def pair_lines(a, z):
    """AIREBO for C-C; Ar-Ar ZBL; Ar-C ZBL (MD repo) or, with --arc-table, the DFT-fitted Ar-C pair table."""
    if getattr(a, "arc_table", ""):
        return ["pair_style hybrid airebo 3.0 zbl 4.0 5.0 table linear 5000",
                f"pair_coeff * * airebo {os.path.abspath(a.airebo)} C NULL",
                f"pair_coeff 2 2 zbl {z} {z}", f"pair_coeff 1 2 table {os.path.abspath(a.arc_table)} ARC"]
    return ["pair_style hybrid airebo 3.0 zbl 4.0 5.0", f"pair_coeff * * airebo {os.path.abspath(a.airebo)} C NULL",
            f"pair_coeff 2 2 zbl {z} {z}", f"pair_coeff 1 2 zbl 6.0 {z}"]


def lammps_input(proj, e_kin, a, seeds, pid):
    fs = 1e-3
    dt_th, dt = a.therm_dt * fs, a.dt * fs
    n_th, n_q = int(round(a.therm_ps * 1000 / a.therm_dt)), int(round(a.quench_ps * 1000 / a.therm_dt))
    n_imp = int(round(a.nve_fs / a.dt))
    dump_every = max(1, int(round(a.dump_fs / a.dt)))
    z = atomic_numbers[proj]
    tdamp = 100 * fs
    T = a.temperature
    vz = vz_down(proj, e_kin) * 1000.0           # A/ps
    return "\n".join([
        f"# {proj} {e_kin:g} eV on a {a.n}x{a.n} sheet, T = {T:g} K",
        "units metal", "boundary p p m", "atom_style charge", "read_data system.data",
        f"mass 1 {atomic_masses[6]:.4f}", f"mass 2 {atomic_masses[z]:.4f}",
        *pair_lines(a, z),
        "neighbor 2.0 bin", "neigh_modify every 1 delay 0 check yes",
        f"group proj id {pid}", "group sheet subtract all proj",
        "compute tsheet sheet temp", "thermo_style custom step time c_tsheet pe ke etotal", "thermo 1000",
        "min_style cg", "minimize 1.0e-10 1.0e-12 2000 20000",
        "reset_timestep 0", f"timestep {dt_th:.8g}",
        f"velocity sheet create {T} {seeds['velocity']} mom yes rot no dist gaussian",
        f"fix therm sheet nvt temp {T} {T} {tdamp:.8g}", f"run {n_th}", "unfix therm",
        f"velocity proj set 0.0 0.0 {vz:.8f} units box",
        f"timestep {dt:.8g}", "fix integ all nve",
        f"dump traj all custom {dump_every} traj.dump id type x y z", "dump_modify traj sort id",
        "variable epot equal pe", "variable ekin equal ke",
        f"fix en all print {max(1, int(round(10 / a.dt)))} \"$(step) $(time) $(pe) $(ke) $(etotal)\" file energy.dat screen no",
        f"run {n_imp}", "undump traj", "unfix integ", "unfix en",
        f"timestep {dt_th:.8g}", f"fix quench sheet langevin {T} {T} {tdamp:.8g} {seeds['langevin']}",
        "fix qnve sheet nve", f"run {n_q}", "unfix quench", "unfix qnve",
        "min_style cg", "minimize 1.0e-8 1.0e-10 3000 30000",
        "write_dump all custom final.dump id type x y z modify sort id",
    ]) + "\n"


def airebo_job(job, a, outdir, workdir):
    mode, e, run = job
    proj = a.proj
    tag = tag_of(proj, mode, e, run)
    final_xyz = outdir / f"graphene_defective_{tag}.xyz"
    sheet = perfect_sheet(a.n, A0_AIREBO)
    xy = site_xy(sheet, mode)
    atoms = sheet.copy()
    atoms += Atoms(proj, positions=[[xy[0], xy[1], sheet.positions[:, 2].mean() + a.height]])
    atoms.set_initial_charges(np.zeros(len(atoms)))
    rd = workdir / tag
    rd.mkdir(parents=True, exist_ok=True)
    write(rd / "system.data", atoms, format="lammps-data", specorder=["C", proj], atom_style="charge")
    write(rd / "initial.xyz", sheet)
    sd = seeds_for(proj, e, mode, run)
    (rd / "in.stageC").write_text(lammps_input(proj, e, a, sd, len(atoms)))
    (rd / "meta.json").write_text(json.dumps(dict(potential="airebo+zbl", target=mode, energy_eV=e, run=run,
                                                   impact_xy=[float(xy[0]), float(xy[1])], a0=A0_AIREBO, seeds=sd,
                                                   dt_fs=a.dt, n=a.n), indent=2))
    if a.dry_run:
        return tag, "dry-run", 0, str(rd)
    t0 = time.time()
    cmd = shlex.split(a.lmp) + ["-in", "in.stageC", "-log", "log.lammps", "-screen", "none"]
    pr = subprocess.run(cmd, cwd=rd, capture_output=True, text=True)
    wall = time.time() - t0
    if pr.returncode != 0 or not (rd / "final.dump").exists():
        tail = (rd / "log.lammps").read_text().splitlines()[-12:] if (rd / "log.lammps").exists() else [pr.stderr[-800:]]
        return tag, "failed", wall, "\n".join(tail)
    spec = ["C", proj]
    write(outdir / f"bombardment_{tag}.traj", read(rd / "traj.dump", format="lammps-dump-text", index=":", specorder=spec))
    write(final_xyz, read(rd / "final.dump", format="lammps-dump-text", index=-1, specorder=spec))
    return tag, "ok", wall, f"{wall / 60:.1f} min"


def cmd_airebo(a):
    outdir, workdir = Path(a.outroot) / "results", Path(a.outroot) / "work"
    outdir.mkdir(parents=True, exist_ok=True)
    workdir.mkdir(parents=True, exist_ok=True)
    jobs = [(m, e, r) for (m, e) in parse_cases(a.cases) for r in range(1, a.runs + 1)]
    todo = [j for j in jobs if a.force or not (outdir / f"graphene_defective_{tag_of(a.proj, *j[:2], j[2])}.xyz").exists()]
    print(f"AIREBO: {len(jobs)} impacts, {len(todo)} to run, {a.parallel} at a time", flush=True)
    with ThreadPoolExecutor(max_workers=a.parallel) as pool:
        futs = [pool.submit(airebo_job, j, a, outdir, workdir) for j in todo]
        for k, f in enumerate(as_completed(futs), 1):
            tag, st, wall, msg = f.result()
            print(f"[{k}/{len(todo)}] {tag}: {st} {msg}", flush=True)


# ----------------------------------------------------------------------------------------------- MACE (ASE)
def make_calc(a):
    if TOY:
        from ase.calculators.emt import EMT
        return EMT()
    import torch
    torch.set_num_threads(a.threads)
    from mace.calculators import MACECalculator
    return MACECalculator(model_paths=os.path.expanduser(a.model), device="cpu", default_dtype=a.dtype)


def fit_a0(calc):
    """Lattice constant of the potential: parabola through E(a) on a 3x3 periodic cell."""
    As = np.linspace(2.38, 2.56, 7)
    E = []
    for x in As:
        s = graphene(a=x, size=(3, 3, 1), vacuum=10.0)
        s.pbc = True
        s.calc = calc
        E.append(s.get_potential_energy())
    c = np.polyfit(As, E, 2)
    return float(-c[1] / (2 * c[0]))


def thermal_snapshots(calc, a, a0, nsnap, thermal_dir):
    """One Langevin run, snapshots every 0.25 ps starting at 1 ps (reused as independent thermal states)."""
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
    thermal_dir.mkdir(parents=True, exist_ok=True)
    paths = [thermal_dir / f"snap{k}.xyz" for k in range(nsnap)]
    if all(p.exists() for p in paths) and not a.force:
        return paths
    sh = perfect_sheet(a.n, a0)
    sh.calc = calc
    rng = np.random.default_rng(a.seed)
    MaxwellBoltzmannDistribution(sh, temperature_K=2 * a.temperature, rng=rng)
    Stationary(sh)
    dyn = Langevin(sh, a.therm_dt * units.fs, temperature_K=a.temperature, friction=0.02 / units.fs, rng=rng)
    start = int(round(a.snap_start_ps * 1000 / a.therm_dt))
    every = int(round(250 / a.therm_dt))
    t0 = time.time()
    dyn.run(start)
    for k in range(nsnap):
        if k:
            dyn.run(every)
        write(paths[k], sh)
        print(f"  thermal snapshot {k}: T = {sh.get_temperature():.0f} K  ({(time.time() - t0) / 60:.1f} min)", flush=True)
    return paths


def mace_job(job, a, calc, snaps, a0, outdir, workdir, k):
    from ase.md.langevin import Langevin
    from ase.md.verlet import VelocityVerlet
    from ase.optimize import FIRE
    mode, e, run = job
    proj = a.proj
    tag = tag_of(proj, mode, e, run)
    if (outdir / f"graphene_defective_{tag}.xyz").exists() and not a.force:
        print(f"{tag}: already done", flush=True)
        return
    rd = workdir / tag
    rd.mkdir(parents=True, exist_ok=True)
    pristine = perfect_sheet(a.n, a0)
    write(rd / "initial.xyz", pristine)
    xy = site_xy(pristine, mode)
    sheet = read(snaps[k % len(snaps)])
    sheet.pbc = (True, True, False)
    nc = len(sheet)
    zs = float(np.median(sheet.positions[:, 2]))
    at = sheet.copy()
    at += Atoms(proj, positions=[[xy[0], xy[1], zs + a.height]])
    vel = np.zeros((len(at), 3))
    if sheet.has("momenta"):
        vel[:nc] = sheet.get_velocities()
    vel[-1, 2] = vz_down(proj, e) / units.fs           # A/fs -> ASE velocity unit
    at.set_velocities(vel)
    at.calc = calc
    dt = a.dt * units.fs
    dyn = VelocityVerlet(at, timestep=dt)
    nsteps = int(round(a.nve_fs / a.dt))
    stride_traj = max(1, int(round(a.dump_fs / a.dt)))
    stride_en = max(1, int(round(10 / a.dt)))
    frames, en = [], []
    t0 = time.time()
    e0 = None
    for s in range(nsteps + 1):
        if s % stride_traj == 0:
            fr = at.copy()
            fr.calc = None
            frames.append(fr)
        if s % stride_en == 0:
            ep, ek = at.get_potential_energy(), at.get_kinetic_energy()
            e0 = ep + ek if e0 is None else e0
            en.append((s, s * a.dt, ep, ek, ep + ek, at.positions[-1, 2] - np.median(at.positions[:nc, 2])))
            if s % (stride_en * 10) == 0:
                print(f"  {tag} t={s * a.dt:6.1f} fs  z_Ar={en[-1][5]:6.2f}  drift={en[-1][4] - e0:+.4f} eV  "
                      f"[{(time.time() - t0) / 60:.1f} min]", flush=True)
        if s < nsteps:
            dyn.run(1)
    with open(rd / "energy.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["step", "t_fs", "epot", "ekin", "etot", "z_proj_above_plane"])
        w.writerows(en)
    write(outdir / f"bombardment_{tag}.traj", frames)
    drift = max(abs(x[4] - e0) for x in en)
    # quench: a reflected projectile no longer interacts (> MACE cutoff), so it is set aside; otherwise it stays in the system
    z_p = at.positions[-1, 2] - np.median(at.positions[:nc, 2])
    vzp = at.get_velocities()[-1, 2]
    away = abs(z_p) > 8.0 and np.sign(z_p) == np.sign(vzp)   # reflected or transmitted, leaving the sheet
    ar_pos = at.positions[-1].copy()
    work = at[:nc] if away else at
    work.calc = calc
    qd = Langevin(work, a.therm_dt * units.fs, temperature_K=a.temperature, friction=0.02 / units.fs,
                  rng=np.random.default_rng(seeds_for(proj, e, mode, run)["langevin"]), fixcm=False)
    qd.run(int(round(a.quench_ps * 1000 / a.therm_dt)))
    FIRE(work, logfile=None).run(fmax=0.05, steps=a.min_steps)
    final = work.copy()
    final.calc = None
    if away:
        final += Atoms(proj, positions=[ar_pos])
    write(outdir / f"graphene_defective_{tag}.xyz", final)
    (rd / "meta.json").write_text(json.dumps(dict(potential="mace", model=a.model, target=mode, energy_eV=e, run=run,
                                                   impact_xy=[float(xy[0]), float(xy[1])], a0=a0, dt_fs=a.dt,
                                                   snapshot=int(k % len(snaps)), max_energy_drift_eV=float(drift),
                                                   projectile_set_aside_for_quench=bool(away),
                                                   wall_min=(time.time() - t0) / 60), indent=2))
    print(f"{tag}: done in {(time.time() - t0) / 60:.1f} min, max |E_tot drift| during NVE = {drift:.4f} eV", flush=True)


def cmd_mace(a):
    outdir, workdir = Path(a.outroot) / "results", Path(a.outroot) / "work"
    outdir.mkdir(parents=True, exist_ok=True)
    workdir.mkdir(parents=True, exist_ok=True)
    thermal_dir = Path(a.thermal_dir) if a.thermal_dir else Path(a.outroot).parent / "mace_thermal"
    calc = make_calc(a)
    a0f = thermal_dir / "a0.json"
    if a0f.exists() and not a.force:
        a0 = json.loads(a0f.read_text())["a0"]
    else:
        a0 = fit_a0(calc)
        thermal_dir.mkdir(parents=True, exist_ok=True)
        a0f.write_text(json.dumps(dict(a0=a0)))
    print(f"MACE lattice constant a0 = {a0:.4f} A (AIREBO {A0_AIREBO})", flush=True)
    cases = parse_cases(a.cases)
    jobs = [(m, e, r) for (m, e) in cases for r in range(1, a.runs + 1)]
    snaps = thermal_snapshots(calc, a, a0, a.snapshots, thermal_dir)
    for k, j in enumerate(jobs):
        mace_job(j, a, calc, snaps, a0, outdir, workdir, k)


# ----------------------------------------------------------------------------------------------- analysis
def cmd_analyse(a):
    script = Path(a.md_repo) / "03_analyze_impacts.py"
    if not script.is_file():
        sys.exit(f"03_analyze_impacts.py not found in {a.md_repo} (use --md-repo)")
    rows = {}
    for name, root in (("airebo", Path(a.airebo_root)), ("mace", Path(a.mace_root))):
        if not (root / "results").is_dir():
            print(f"{name}: no results in {root}")
            continue
        out = root / "impact_analysis.csv"
        subprocess.run([sys.executable, str(script), "--indir", str(root / "results"), "--workdir", str(root / "work"),
                        "--out", str(out), "--summary", str(root / "yield_summary.csv"), "--reanalyse", "--jobs", "2"],
                       check=True, stdout=subprocess.DEVNULL)
        for r in csv.DictReader(open(out)):
            rows.setdefault((r["target"], float(r["energy_eV"])), {}).setdefault(name, []).append(r)
    print(f"\n{'case':16s} {'potential':8s} {'runs':>4s} {'n_vacant':>16s} {'closest_C min/med':>18s} {'fate':>22s}  outcomes")
    for key in sorted(rows):
        for name in ("airebo", "mace"):
            rs = rows[key].get(name, [])
            if not rs:
                continue
            nv = [int(r["n_vacant"]) for r in rs]
            cc = [float(r["closest_C"]) for r in rs]
            fates = {}
            for r in rs:
                fates[r["fate"]] = fates.get(r["fate"], 0) + 1
            outs = sorted({r["outcome"] for r in rs})
            print(f"{key[0] + ' ' + format(key[1], 'g') + ' eV':16s} {name:8s} {len(rs):4d} {str(nv):>16s} "
                  f"{min(cc):8.2f}/{np.median(cc):<8.2f} {str(fates):>22s}  {'; '.join(outs)}")
    for root in (Path(a.mace_root), Path(a.mace_root).parent / "mace_dt01"):
        for mf in sorted((root / "work").glob("*/meta.json")) if (root / "work").is_dir() else []:
            m = json.loads(mf.read_text())
            if 'max_energy_drift_eV' not in m:
                continue
            print(f"{root.name}/{mf.parent.name}: dt {m['dt_fs']} fs, max |dE_tot| {m['max_energy_drift_eV']:.4f} eV, "
                  f"{m['wall_min']:.0f} min")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--proj", default="Ar")
    common.add_argument("--n", type=int, default=8, help="n x n unit cells (2 n^2 C atoms)")
    common.add_argument("--cases", nargs="+", default=["hollow:70", "bridge:70", "hollow:100", "top:100"])
    common.add_argument("--runs", type=int, default=1, help="runs per case (different thermal states)")
    common.add_argument("--temperature", type=float, default=300.0)
    common.add_argument("--height", type=float, default=6.0)
    common.add_argument("--nve-fs", type=float, default=750.0)
    common.add_argument("--dump-fs", type=float, default=5.0)
    common.add_argument("--therm-ps", type=float, default=2.0)
    common.add_argument("--quench-ps", type=float, default=0.5)
    common.add_argument("--force", action="store_true")
    x = sp.add_parser("airebo", parents=[common])
    x.add_argument("--airebo", required=True)
    x.add_argument("--arc-table", default="", help="DFT-fitted Ar-C pair table from arc_table.py (replaces ZBL for Ar-C)")
    x.add_argument("--lmp", default=os.environ.get("LMP_CMD", "lmp"))
    x.add_argument("--dt", type=float, default=0.05)
    x.add_argument("--therm-dt", type=float, default=0.5)
    x.add_argument("--parallel", type=int, default=2)
    x.add_argument("--outroot", default=str(ROOT / "airebo"))
    x.add_argument("--dry-run", action="store_true")
    x.set_defaults(fn=cmd_airebo)
    y = sp.add_parser("mace", parents=[common])
    y.add_argument("--model", default="")
    y.add_argument("--dtype", default="float32")
    y.add_argument("--threads", type=int, default=8)
    y.add_argument("--dt", type=float, default=0.2)
    y.add_argument("--therm-dt", type=float, default=1.0)
    y.add_argument("--snapshots", type=int, default=4)
    y.add_argument("--snap-start-ps", type=float, default=1.0)
    y.add_argument("--min-steps", type=int, default=300)
    y.add_argument("--seed", type=int, default=2026)
    y.add_argument("--outroot", default=str(ROOT / "mace"))
    y.add_argument("--thermal-dir", default="")
    y.set_defaults(fn=cmd_mace)
    z = sp.add_parser("analyse")
    z.add_argument("--md-repo", required=True)
    z.add_argument("--airebo-root", default=str(ROOT / "airebo"))
    z.add_argument("--mace-root", default=str(ROOT / "mace"))
    z.set_defaults(fn=cmd_analyse)
    a = p.parse_args()
    if TOY and getattr(a, "proj", None) == "Ar":
        sys.exit("toy mode (EMT) has no Ar: use --proj C")
    a.fn(a)


if __name__ == "__main__":
    main()

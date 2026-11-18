"""
The ONE DFT recipe for every training label of this project (Stage B and later).

Chosen from the calibration in ../stageA_dft_check/02_dft_calibration.py (see README.md, "DFT recipe").
Nothing else in the workflow defines DFT settings: every script imports them from here, every database row is stamped
with RECIPE_ID, and export.py refuses rows whose stamp does not match.  Change a number here -> new RECIPE_ID -> old
rows are no longer accepted (they must be recomputed), so two recipes can never be mixed by accident.

GPAW must be imported before ASE touches the MPI world, so import this module FIRST in every entry script.
Set MLIP_TOY=1 to use EMT instead of GPAW (workflow tests only; such rows get the recipe id 'toy').
"""
import hashlib
import json
import os

import numpy as np

TOY = os.environ.get("MLIP_TOY") == "1"
if not TOY:
    from gpaw import GPAW, PW, FermiDirac  # noqa: F401  (must come before any ase.parallel use)

RECIPE = dict(
    code="gpaw", xc="PBE", mode="pw", ecut=450.0, kdens=5.0, gamma_centred=True,
    smearing="FermiDirac", smear=0.05, spinpol=True, init_magmom=0.3,
    econv=1e-4, dens_conv=1e-4,
    cell_z=20.0,            # fixed vacuum direction; the sheet (mean z of C atoms) is centred at z = cell_z/2
    # Closest allowed pair distance (A).  Closer pairs belong to the ZBL regime: PAW labels there are not trusted.
    # Starting values (C-C 1.0 lets a thermal rattle of a 1.42 A bond through); revisit after the first audit.
    min_dist={"C-C": 1.0, "C-N": 1.1, "N-N": 1.0, "Ar-C": 1.6, "Ar-N": 1.6, "Ar-Ar": 2.0},
)
AUDIT = dict(ecut=600.0, kdens=5.0)   # the reference setting of the calibration, used only by audit.py

RECIPE_ID = "toy" if TOY else hashlib.sha1(json.dumps(RECIPE, sort_keys=True).encode()).hexdigest()[:10]


def kmesh(atoms, kdens):
    """Gamma-centred mesh from a k-point density; the vacuum direction always has 1 point (same as calibration)."""
    rec = atoms.cell.reciprocal()
    n = [max(1, int(np.ceil(2 * np.pi * np.linalg.norm(r) * kdens))) for r in rec]
    n[2] = 1
    return tuple(n)


def normalise(atoms):
    """Put a structure into the recipe geometry: cell_z fixed, sheet centred, periodic in xy only conceptually (pbc True)."""
    at = atoms.copy()
    at.calc = None
    c = [i for i, s in enumerate(at.get_chemical_symbols()) if s == "C"]
    zc = at.positions[c, 2].mean() if c else at.positions[:, 2].mean()
    at.positions[:, 2] += RECIPE["cell_z"] / 2 - zc
    cell = np.array(at.cell)
    cell[2] = [0.0, 0.0, RECIPE["cell_z"]]
    at.set_cell(cell, scale_atoms=False)
    at.pbc = True
    return at


def worst_pair(atoms):
    """(ok, margin): margin = min over pairs of (distance - allowed minimum for that element pair); ok if margin >= 0."""
    from ase.neighborlist import neighbor_list
    i, j, d = neighbor_list("ijd", atoms, 3.0)
    sym = atoms.get_chemical_symbols()
    margin = 3.0
    for a, b, dist in zip(i, j, d):
        key = "-".join(sorted((sym[a], sym[b])))
        margin = min(margin, dist - RECIPE["min_dist"].get(key, 1.6))
    return margin >= 0, float(margin)


def structure_hash(atoms):
    """Identity of a structure (independent of the recipe): same geometry -> same hash."""
    h = hashlib.sha1()
    h.update(",".join(atoms.get_chemical_symbols()).encode())
    h.update(np.round(atoms.positions, 4).tobytes())
    h.update(np.round(np.array(atoms.cell), 4).tobytes())
    return h.hexdigest()[:16]


def make_calc(atoms, txt, ecut=None, kdens=None):
    ecut = RECIPE["ecut"] if ecut is None else ecut
    kdens = RECIPE["kdens"] if kdens is None else kdens
    if TOY:
        from ase.calculators.emt import EMT
        return EMT()
    extra = {}
    mx = os.environ.get("MLIP_MIXER")        # e.g. "0.02 5 100" = beta nold weight; changes the SCF path, not the result
    if mx:
        from gpaw import Mixer
        beta, nold, weight = mx.split()
        extra["mixer"] = Mixer(float(beta), int(nold), float(weight))
    eg = os.environ.get("MLIP_EIGENSOLVER")  # e.g. "dav": Davidson, slower per step but more robust than the default RMM-DIIS
    if eg:
        extra["eigensolver"] = eg
    conv = {"energy": RECIPE["econv"], "density": RECIPE["dens_conv"]}
    mi = os.environ.get("MLIP_MAXITER")      # SCF iteration cap; a structure that does not converge is recorded as failed and skipped
    if mi:
        extra["maxiter"] = int(mi)
    return GPAW(mode=PW(ecut), xc=RECIPE["xc"], kpts={"size": kmesh(atoms, kdens), "gamma": True},
                occupations=FermiDirac(RECIPE["smear"]), spinpol=RECIPE["spinpol"],
                convergence=conv, txt=str(txt), **extra)


def compute(atoms, txt, ecut=None, kdens=None):
    """Energy, forces, total magnetic moment of `atoms` with the recipe (or an overridden cutoff / k-density)."""
    at = atoms.copy()
    m0 = at.get_initial_magnetic_moments()
    # Default: the recipe moment on every atom (all rows of Stages A-C). Only deliberately set large moments (> 1 muB, e.g. the N atom
    # at 3 muB from scans.py --proj-magmom) are kept, so stored small moments in old files cannot change an old label.
    if not (len(m0) and np.max(np.abs(m0)) > 1.0):
        at.set_initial_magnetic_moments([RECIPE["init_magmom"]] * len(at))
    at.calc = make_calc(at, txt, ecut, kdens)
    e = float(at.get_potential_energy())
    f = np.array(at.get_forces())
    try:
        mu = float(at.get_magnetic_moment())
    except Exception:
        mu = 0.0
    return e, f, mu


def db_tags(extra=None):
    """Key-value pairs stamped on every labelled row."""
    kv = dict(recipe=RECIPE_ID, ecut=RECIPE["ecut"], kdens=RECIPE["kdens"], smear=RECIPE["smear"],
              xc=RECIPE["xc"], econv=RECIPE["econv"], mixer=os.environ.get("MLIP_MIXER", "default"))
    if not TOY:
        import gpaw
        kv["gpaw"] = gpaw.__version__
    kv.update(extra or {})
    return kv

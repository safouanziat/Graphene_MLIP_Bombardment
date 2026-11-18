# Stage A: does AIREBO reproduce DFT defect energetics?

Formation energies of five defects in a 4×4 graphene sheet (31–33 atoms), relaxed with AIREBO (LAMMPS) and with DFT
(GPAW, PBE, spin-polarised), same starting structures.

| defect | atoms | AIREBO (eV) | DFT (eV) | AIREBO − DFT (eV) | DFT moment (μB) |
|---|---|---|---|---|---|
| V1 (monovacancy) | 31 | 7.60 | 7.92 | −0.32 | 1.58 |
| Stone-Wales | 32 | 6.24 | 6.07 | +0.17 | 0.00 |
| V2 (divacancy) | 30 | 9.24 | 8.57 | +0.67 | 0.00 |
| C adatom | 33 | 4.26 | **6.49** | **−2.23** | 0.00 |
| Frenkel pair | 32 | 15.08 | 14.14 | +0.93 | 0.00 |

(`dft_cheap/results.csv`; formation energy = E(defect) − N·E(pristine)/N_pristine, adatom: relative to one graphene C.)

## Reading
- AIREBO is within ~0.3 eV for V1 and Stone-Wales, but **underestimates the adatom energy by 2.2 eV** (it binds the adatom
  far too strongly) and overestimates V2 and the Frenkel pair by 0.7–0.9 eV.
- Impact damage is made of exactly these defects, so the AIREBO damage statistics inherit these biases. This motivates
  Stage B.

## Caveat: these DFT numbers are "cheap"
They use a reduced plane-wave cutoff and a coarse k-mesh to afford the relaxations. The calibration in
`02_dft_calibration.py` (see `../stageB_mace_training/README.md`) shows that this carries about ±0.15 eV of k-sampling
error for V1/V2 and up to 0.6 eV in the least accurate settings, and that the V1 magnetic state is only reproduced at the
final recipe (450 eV, k-density 5). Qualitative conclusions (the sign and the size of the adatom error) are unaffected;
the Stage B recipe is used for all later energies.

## Convergence checks (cheap settings, from `dft_conv/` and `cutoff_test.py`)
Cell size, 5x5 sheets (48-50 atoms), cheap DFT settings (350 eV, 3x3x1 k-mesh; confirmed for V2 in `run_checks.sh`, assumed identical for the earlier V1 and Stone-Wales runs) and AIREBO:

| defect | AIREBO 4x4 / 5x5 (eV) | DFT 4x4 / 5x5 (eV) | AIREBO − DFT at 5x5 (eV) |
|---|---|---|---|
| V1 | 7.60 / 7.59 | 7.92 / 7.70 | −0.11 |
| Stone-Wales | 6.24 / 5.81 | 6.07 / 5.54 | +0.27 |
| V2 | 9.24 / 8.78 | 8.57 / 8.13 | +0.65 |

The sign and size of the V2 error are the same in both cells (+0.67 at 4x4, +0.65 at 5x5); the V1 gap shrinks and the Stone-Wales gap grows a little. Absolute V2 and
Stone-Wales energies depend on the cell by 0.4-0.5 eV for both methods, which is also seen in the Stage B recipe (V2 −0.40, SW −0.51 eV, 4x4 → 5x5).
The adatom and Frenkel pair were not repeated at 5x5, so their errors (−2.2 and +0.9 eV) are 4x4 numbers.

AIREBO alone in 5x5 cells (`airebo_vac15/results.csv`, eV): V1 7.59, Stone-Wales 6.02, V2 8.78, adatom 4.25, Frenkel 15.06, against 4x4 values of 7.60 / 6.24 / 9.24 / 4.26 / 15.08.
The adatom and Frenkel energies of AIREBO do not change with the cell, so the adatom error is not a 4x4 artefact: against the final-recipe 5x5 DFT adatom (6.42 eV, Stage B acceptance run) it is −2.2 eV.
**Inconsistency to resolve:** the 5x5 Stone-Wales energy of AIREBO is 6.02 eV in `airebo_vac15` but 5.81 eV in `dft_conv` (V1 and V2 agree to 0.01 eV).
The SW relaxation takes 87 steps and may end in slightly different minima or cell settings; the AIREBO − DFT gap for SW at 5x5 is therefore between +0.27 and +0.48 eV, not a single number.

Plane-wave cutoff (V1 at fixed 4x4 geometries, 2x2x1 k-mesh): E_f = 7.919 eV (350 eV), 7.897 eV (450 eV), 7.903 eV (550 eV), i.e. a change of at most 0.022 eV.
The cutoff is not the limiting parameter; the k-mesh is (see `02_dft_calibration.py` and the Stage B README: only 450 eV with k-density 5 reproduced the V1 magnetic moment of the reference).

Literature PBE ranges printed by the script for orientation: V1 7.4-7.9, V2 7.5-8.0, Stone-Wales 4.5-5.2, bridge adatom 6-7 eV. In this project the DFT V1 (7.92) is at the upper end, V2 (8.57 cheap, 8.07 at 5x5 with the final recipe) slightly above it, and Stone-Wales (6.07) above the range at 4x4, 5.45 at 5x5 with the final recipe.

## Files
| file / folder | content |
|---|---|
| `01_dft_defect_check.py` | defect relaxations, AIREBO vs DFT |
| `02_dft_calibration.py` | cutoff / k-mesh calibration that fixed the Stage B recipe (six settings A-E and REF, six structures each; log `calib.log`, output `calib/`) |
| `cutoff_test.py`, `cutoff_test/` | plane-wave cutoff test for V1 (350 / 450 / 550 eV); GPAW text outputs of the six runs |
| `dft_cheap/` | the relaxed 4x4 structures (`relaxed/`), `results.csv`, `results.jsonl`, `results.db` |
| `dft_conv/` | 5x5 cell-size check (V1, Stone-Wales, V2): AIREBO and DFT relaxations, `results.csv`, `results_V1_SW.csv` |
| `run_checks.sh`, `checks.log`, `stageA.log`, `stageA_conv.log` | driver script for the V2 5x5 and cutoff runs, and the logs of the three Stage A runs |
| `airebo_vac15/` | AIREBO relaxations of all five defects in 5x5 cells (48-51 atoms; folder name suggests a 15 A vacuum, not checked): `results.csv`, `logs/`, `relaxed/`, `traj/` |
| `dft_check/` | earliest GPAW run folder; only `logs/bench.txt` (a benchmark log) is in it, `relaxed/` is empty |

The AIREBO potential file (`CH.airebo`) is not distributed here; use the LAMMPS-supplied file.

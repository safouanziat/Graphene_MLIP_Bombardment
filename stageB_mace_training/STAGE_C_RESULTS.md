# Stage C pilot: Ar impacts on graphene, AIREBO+ZBL vs Ar-aware MACE

All numbers below were measured in this project; nothing is taken from the literature except the two reference values marked
*(literature)*. Everything is a **pilot**: few runs per case, one small sheet, 0 K or 300 K single sheets (see "Limits").

## Setup

| | AIREBO + ZBL | MACE |
|---|---|---|
| engine | LAMMPS (`pair_style hybrid airebo 3.0 zbl 4.0 5.0`, as in the MD repo; Ar-C = ZBL only) | ASE velocity Verlet, CPU, float32, 8 threads |
| model | CH.airebo | `omat-0` foundation model fine-tuned on the project's DFT data **plus Ar scans**, all foundation elements kept (`--foundation_model_elements True`) |
| sheet | 8x8 hexagonal cells = 128 C (+1 Ar), a0 = 2.419 A | same size, own a0 fitted by the potential |
| thermal state | 300 K, 2 ps NVT, new state per run | 300 K Langevin, 4 reused snapshots (1.0-1.75 ps) |
| impact | Ar 6 A above the sheet, NVE 750 fs, dt 0.05 fs | Ar 6 A above the sheet, NVE 750 fs, dt 0.2 fs |
| after impact | Langevin quench 0.5 ps + minimisation | same (FIRE) |
| classification | `03_analyze_impacts.py` of the MD repo, unchanged | same |

Impact sites: `top` (on an atom), `bridge` (middle of a bond), `hollow` (hexagon centre). Pairing between potentials is by site,
energy and run number, not by trajectory. MACE cost: about 47-51 min per 750 fs impact on 8 threads.
Reference (MD repo, 32x32 sheet = 2048 C, random impact points, AIREBO+ZBL): vacancy threshold about 63 eV.

## 1. Does the 8x8 sheet stand in for the repo's 32x32 sheet? (AIREBO only)

Same site/energy cases on both sizes (8x8: 5 runs, 32x32: 3 runs, 32x32 with the repo's 2 ps quench):

| case | 8x8 | 32x32 |
|---|---|---|
| hollow 70 eV | no vacancy, Ar transmitted 5/5 | same, 3/3 |
| hollow 100 eV | no vacancy, transmitted 5/5 | same, 3/3 |
| bridge 70 eV | no vacancy, reflected 3 / transmitted 2 | no vacancy, reflected 1 / transmitted 2 |
| top 100 eV | single vacancy 5/5, transmitted | same, 3/3 |

Outcomes agree for the cases tested. The fate mix at bridge 70 eV differs, but with 3-5 runs this is not significant.
Absolute yields of the 8x8 sheet are not comparable with the repo's 32x32 yields (a small sheet heats much more).

## 2. Top-site vacancy threshold, AIREBO vs MACE (8x8, 300 K)

Vacancies per runs (single vacancy in every case where one forms):

| Ar energy (eV) | AIREBO + ZBL | AIREBO + DFT-fitted Ar-C pair (diagnostic, section 4) | MACE |
|---|---|---|---|
| 40 | 0/5 | 0/5 | |
| 45 | 0/5 | 0/5 | |
| 50 | 0/5 | 0/5 | **3/3** |
| 55 | 2/5 | 2/5 | 2/2 |
| 60 | 4/5 | 4/5 | 4/4 |
| 70 | 5/5 | | |
| 100 | 5/5 | | |

At 60 eV Ar was reflected in 5/5 AIREBO runs and transmitted in 3/4 MACE runs (borderline, p about 0.05). At 50 eV MACE gave a vacancy
in 3/3 runs where AIREBO gave none in 15 runs at 40-50 eV. Counts are small: this shows a lower MACE threshold, not its value.

Energy conservation of the MACE impacts (dt 0.2 fs, 750 fs, flat 8x8 sheet): max |E_tot drift| 0.7-4.9 meV for the 50-60 eV runs.

## 3. Where do the potentials differ? Static Ar-C wall (frozen DFT scan sheet, Ar above one C atom)

Interaction energy relative to d = 4.5 A (eV); DFT = the project's PBE scan points:

| d (A) | AIREBO+ZBL | MACE | DFT |
|---|---|---|---|
| 1.00 | 45.7 | 44.8 | - |
| 1.30 | 20.2 | 17.7 | - |
| 1.65 | 8.99 | 8.19 | 8.23 |
| 1.95 | 4.83 | 3.82 | 3.87 |
| 2.10 | 3.59 | 2.49 | 2.54 |
| 2.60 | 1.38 | 0.47 | 0.53 |
| 3.00 | 0.63 | 0.08 | 0.12 |

MACE follows the DFT points to about 0.05 eV between 1.65 and 3.0 A. AIREBO+ZBL is too repulsive beyond about 1.9 A (about 40 % at 2.1 A,
2.6x at 2.6 A). Below 1.65 A there is no DFT (PAW cores overlap); the two potentials stay within about 10 % down to 1.0 A, and
the frozen-sheet turning point at 50 eV is 0.97 A (AIREBO+ZBL) vs 0.99 A (MACE), i.e. the knock-out happens in a region without DFT labels.

## 4. Is the stiff Ar-C tail the cause? No.

A pair potential phi(r) = 582 exp(-2.92 r) eV (smooth cut-off 4-5 A, ZBL below 1.1-1.6 A) was fitted to the 9 top-site DFT scan points
(rms 75 meV, max 110 meV; an all-site pair fit failed: rms 459 meV, DFT is not pair-additive). Used instead of ZBL for Ar-C in AIREBO it reproduces
the DFT top-site curve (and MACE's) but **leaves the thresholds unchanged** (table in section 2). It is a diagnostic for the top site only.

## 5. Carbon displacement threshold (no Ar): one C atom kicked perpendicular to a flat sheet at 0 K, 400 fs

| potential | sheet | not sputtered at | sputtered at |
|---|---|---|---|
| AIREBO | 5x5 and 8x8 | 25 eV | 26 eV |
| MACE | 5x5 | 20 eV | 21 eV |

Reference values from Chirita Mihaila, Susi, Kotakoski, *Influence of temperature on the displacement threshold energy in graphene*
(arXiv:1811.04011) *(literature)*: DFTB 20.0 eV at 0 K (the authors state this is below DFT and experiment; the DFT value is not quoted in the pages
read here); experiment-based fits 21.1-21.3 eV; at 300 K the threshold has a spread (FWHM) of about 1 eV.

## 6. Bridge site, 70 eV: qualitative disagreement, and a static DFT arbiter

MACE (3 runs) gave a di-vacancy in 3/3, Ar transmitted, 1-2 displaced atoms. AIREBO+ZBL (5 runs) gave no vacancy in 5/5 (Ar reflected 3, transmitted 2, 0 displaced atoms).
The decisive geometry (Ar inside 1.6 A of C) cannot be DFT-labelled, so a static test of the C-C bond-breaking energetics was used instead:
an atom (push1) or two neighbouring atoms (push2) of a relaxed 4x4 sheet (32 C, a = 2.4704 A, DFT PBE, same recipe) are moved rigidly out of plane by s, all others frozen, no Ar.
18 of 21 structures converged (s = 2.25/2.50 for push1 and s = 2.25 for push2 did not: SCF failure with a nearly detached atom).

Energy relative to s = 0 (push1) or s = 0.25 (push2), eV:

| path | s (A) | DFT | AIREBO | MACE |
|---|---|---|---|---|
| push1 | 0.50 | 1.84 | 1.34 | 1.85 |
| push1 | 1.00 | 6.70 | 5.71 | 6.60 |
| push1 | 1.25 | 9.40 | 13.19 | 9.03 |
| push1 | 1.50 | 11.77 | 15.13 | 11.12 |
| push1 | 2.00 | 14.50 | 15.14 | 14.36 |
| push2 | 1.00 | 8.52 | 7.05 | 8.21 |
| push2 | 1.25 | 12.47 | 16.70 | 11.65 |
| push2 | 1.50 | 14.94 | 19.04 | 14.26 |
| push2 | 2.00 | 17.09 | 18.84 | 17.37 |
| push2 | 2.50 | 17.69 | 18.80 | 18.62 |

Force RMSE against DFT over all points: AIREBO 1.31 (push1) / 1.09 (push2) eV/A; MACE 0.145 / 0.198 eV/A.

* AIREBO is 20-30 % too soft up to s = 1.0 A, then jumps by about 7.5 eV between s = 1.0 and 1.25 A and is 3.8-4.2 eV above DFT at s = 1.25 A, then stays flat (about 15.1 eV / 18.8 eV) while DFT keeps rising smoothly.
  The jump coincides with the C-C neighbour distance reaching 1.74-1.90 A, i.e. the AIREBO C-C cut-off switching region (as far as I recall 1.7-2.0 A; not checked against CH.airebo here). That is an artefact of the cut-off, not of the physics.
* MACE follows DFT within 0.1-0.3 eV up to s = 1.0 A and within 0.8 eV (worst case, push2 s = 1.25 A) at all points tested.
* This is consistent with the higher AIREBO carbon displacement threshold (section 5): the extra steep rise at 1.0-1.5 A is a barrier that DFT does not have.

What it does and does not show: for bond stretching with the sheet otherwise frozen, MACE is close to DFT and AIREBO is not, so MACE's bridge-site di-vacancy is more credible than AIREBO's null result.
It does not prove the dynamics: the sheet is unrelaxed, Ar is absent, and the Ar-C overlap region below 1.6 A has no DFT.

## 7. Dynamics frames of bridge impacts at 70 eV: DFT vs AIREBO vs MACE (`framescan.py`)

Static pushes (section 6) leave open whether the potentials also differ on configurations the impact itself produces.
A 4x4 sheet (32 C + Ar, 0 K, own equilibrium lattice per potential) was hit by Ar at a bridge site, 70 eV, 2 impacts per potential
(impact point = bridge centre + random shift up to 0.15 A). Frames within 150 fs after first contact were kept where the DFT recipe allows
(Ar-C >= 1.6 A, C-C >= 1.0 A), rescaled in-plane to the DFT lattice, and labelled with DFT (same recipe). Frames with the Ar more than 7 A from the sheet
have the Ar parked at 7 A above it (no Ar-C interaction there), so the deformed-sheet stage is tested too. Both potentials evaluated **all** frames,
whoever generated them. Both AIREBO impacts reflected the Ar, both MACE impacts transmitted it (as in the 8x8 runs).

Labelled: 14 of 24 frames (11 made by AIREBO, 3 by MACE; 4 of the 14 are flat 10 fs start frames used as the per-impact energy reference).
The other 10 did not converge in SCF, even with a slower, more robust setting (Davidson, small mixing step, 500 iterations; one frame was recovered after 4 h).
They are mostly the strongly distorted MACE-made frames, so the set under-represents exactly the geometries that decide the bridge question
(DFT only returns the frames that converge, a bias towards easier cases).

| | AIREBO | MACE |
|---|---|---|
| force RMSE over all 14 frames (eV/A) | 1.69 | 0.51 |
| frames made by AIREBO (n = 11) | 1.64 | 0.57 |
| frames made by MACE (n = 3, 2 of them flat) | 1.86 | 0.23 |
| Ar within 3 A of a C (n = 6) | 2.13 | 0.73 |
| Ar further than 3 A, deformed sheet (n = 8) | 1.26 | 0.24 |
| energy-difference error vs first frame of the impact, mean / rms / max (eV) | -2.5 / 3.4 / 5.9 | +0.6 / 1.6 / 3.3 |

Selected frames (energy relative to the first frame of the same impact, eV):

| frame | Ar-C min (A) | DFT | AIREBO | MACE |
|---|---|---|---|---|
| airebo r1, 75 fs | 2.53 | 52.5 | 46.6 | 53.7 |
| airebo r0, 110 fs | 1.74 | 30.5 | 30.4 | 33.8 |
| airebo r1, 110 fs | 1.79 | 32.6 | 31.5 | 35.7 |
| mace r1, 140 fs (a C atom ejected to 11.8 A from the sheet) | 5.43 | 25.8 | 28.2 | 26.2 |

* MACE has the smaller force error in every group, also on frames AIREBO itself generated (0.57 vs 1.64 eV/A).
* AIREBO under-predicts the energy of the deformed sheet by 2-6 eV on most frames.
* MACE is not uniformly better: at the closest contacts kept (Ar-C 1.74 and 1.79 A) it is too repulsive by about 3.3 and 3.1 eV (about 10 %), where AIREBO is closer on energy but still has 1.7-1.8 eV/A force errors.
* Together with section 6 this supports the MACE bridge-site di-vacancy as more credible than the AIREBO null result. It does not prove the 70 eV outcome: only 3 MACE-made frames, none of the converged ones shows the full di-vacancy stage, and the closest-contact region (Ar-C < 1.6 A) has no DFT.

## Interpretation (what is and is not shown)

* The lower MACE Ar threshold at the top site goes with a lower carbon displacement threshold (20-21 eV vs 25-26 eV for AIREBO).
  With a maximum energy transfer of about 71 % for Ar on C, a 4-6 eV difference in carbon threshold corresponds to about 6-8 eV in Ar energy,
  the size of the observed gap. This is consistent, not a proof of causality.
* MACE agrees with the experiment-based carbon threshold (about 21.2 eV) within about 1 eV; AIREBO is about 4-5 eV above it.
  It follows that the repo's AIREBO-based Ar thresholds (about 63 eV for random impacts) are probably somewhat too high.
* **Not shown:** that MACE's Ar threshold is correct (no DFT impact reference at 50 eV is affordable on this hardware).

## Limits

* Few runs: 2-5 per case; MACE 2-4. No yields or confidence intervals.
* One small sheet (128 C); MACE compared at the top site (thresholds) and bridge 70 eV (3 runs); no MACE runs at hollow.
* Different thermalisation (2 ps vs 1 ps with 4 reused snapshots) and the 0.5 ps quench instead of the repo's 2 ps for the 8x8 sets.
* MACE time step 0.2 fs checked by energy conservation only (dt 0.1 fs not run).
* Displacement thresholds at 0 K on 5x5 (MACE) / 5x5 and 8x8 (AIREBO), one direction, 1 eV grid.
* The DFT-fitted Ar-C pair is a top-site diagnostic, not a transferable potential.
* Nitrogen (Stage D) not started; it needs its own N-C data.

## Reproduce (in `stageB_mace_training`)

```
python -W ignore stageC_pilot.py airebo --airebo ../stageA_dft_check/CH.airebo --n 8 --runs 5 --parallel 4
python -W ignore stageC_pilot.py airebo --airebo ... --n 8 --cases top:40 top:45 top:50 top:55 top:60 top:70 --runs 5 --parallel 4
python -W ignore stageC_pilot.py airebo --airebo ... --n 32 --runs 3 --parallel 4 --quench-ps 2.0 --outroot data/stageC/airebo_n32
python -W ignore stageC_pilot.py mace --model data/ft/omat-0_all_ar/omat-0_all_ar.model --cases top:60 --runs 4   # then top:55 x2, top:50 x3
python -W ignore stageC_pilot.py analyse --md-repo <path to the MD repo>
python -W ignore pushscan.py   # then dft_flow.py on data/pushscan, then:
python -W ignore pushcompare.py --airebo ... --models data/ft/omat-0_all_ar/omat-0_all_ar.model
python -W ignore framescan.py generate --airebo ... --model data/ft/omat-0_all_ar/omat-0_all_ar.model --runs 2   # then dft_flow.py on data/framescan, then:
python -W ignore framescan.py compare --airebo ... --model data/ft/omat-0_all_ar/omat-0_all_ar.model
python -W ignore wall_curve.py --airebo ... --models data/ft/omat-0_all_ar/omat-0_all_ar.model --db data/arscan/labelled.db
python -W ignore arc_table.py --db data/arscan/labelled.db --out data/stageC/arc_top.table --sites top --terms 2
python -W ignore pka_threshold.py airebo --airebo ... --energies 18 20 22 23 24 25 26 27 28 30
python -W ignore pka_threshold.py mace --model ... --energies 16 18 20 22 24 26 28 30 --threads 8
```

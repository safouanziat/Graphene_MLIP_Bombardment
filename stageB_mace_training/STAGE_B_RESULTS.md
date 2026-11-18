# Stage B additions: Ar data, Ar-aware fine-tune, impact test, MD cost

This note covers the Stage B work that is **not** in `README.md` (which has the DFT recipe, the 68-row carbon set, the audit, the
leave-one-family-out tests and the 5x5 acceptance run). All numbers below come from the report files of this project
(`impact_report*.txt`, `ar_force_check*.txt`, `md_timing.txt`, `lammps_mace_check.txt`, `stageB_collect.txt`); nothing is estimated.
Everything uses DFT recipe `8c357474ed` (GPAW PBE, 450 eV, k-density 5).

## 1. Ar data for the model (`scans.py`)

`data/arscan/labelled.db`: 37 rows = 1 isolated Ar atom (reference energy) + 36 static scans of an Ar atom above the DFT-relaxed 4x4 sheet
(3 sites top / bridge / hollow x 9 closest Ar-C distances 1.65-4.5 A, flat sheet = 27 rows; plus 9 rattled copies at 1.8, 2.1, 2.6 A).
No pair closer than the recipe minimum (Ar-C 1.6 A, C-C 1.0 A) is labelled: PAW labels in the ZBL regime are not trusted, and
isolated (sputtered) carbon atoms do not converge in SCF, so such frames are excluded from the DFT sets.

## 2. Fine-tunes, and why the first Ar test could not run

| model | training data | result on Ar frames |
|---|---|---|
| `omat-0_all` | 68 carbon rows | **cannot be evaluated**: `ValueError: 18 is not in list`. By default a fine-tuned MACE keeps only the elements present in the training data, so Ar is removed from the model |
| `omat-0_all_ar` | 68 carbon rows + 36 Ar scans (104 rows) | works; `--foundation_model_elements True` keeps all foundation elements, E0s {C: -1.254804, Ar: 0.038912} |

Run `omat-0_all_ar`: base OMAT-0, 50 epochs, lr 5e-4, energy weight 10, forces weight 100, float64, batch 4, 10 % of rows for checkpoint selection,
`config_weight` 0.3 for 9 strongly rattled rows (95 rows at 1.0), 84.7 min on 8 CPU threads. Training families: pristine 11, V1 11, V2 13, SW 10, adatom 12, frenkel 11, ar-scan 36.
This is the model used in Stage C.

## 3. Impact test (`impacts.py`): DFT force and energy checks on impact snapshots

15 snapshots from short AIREBO+ZBL Ar impacts on a 32-atom sheet (sites top, bridge, hollow; 30/70/100 eV; Ar at most 7 A above the sheet and within 3.5 A of a C;
closest kept Ar-C 1.66 A). These snapshots are a **test set**; the Ar scans are the training data. DFT force rms over all snapshots: 2.27 eV/A.
Errors against DFT (eV/A; dE = rms error of the energy difference to the first snapshot of the same impact, eV):

| method | F all atoms | F on Ar | F on C within 3 A of Ar | F on C further than 3 A | dE rms |
|---|---|---|---|---|---|
| AIREBO + ZBL | 1.107 | 1.186 | 2.432 | 0.604 | 4.668 |
| zero-shot OMAT-0 | 0.674 | 2.787 | 1.033 | 0.269 | 3.868 |
| zero-shot MPA-0 | 1.583 | 6.508 | 2.757 | 0.326 | 16.997 |
| **OMAT-0 fine-tuned with Ar (`omat-0_all_ar`)** | **0.417** | **0.975** | **0.823** | **0.230** | **1.656** |

Per snapshot, the Ar-aware model is better than AIREBO+ZBL on the force on Ar in most frames, but not at the closest contacts:

| snapshot | Ar-C (A) | DFT, F on Ar (eV/A) | F-on-Ar error: Ar-aware MACE | F-on-Ar error: AIREBO+ZBL |
|---|---|---|---|---|
| bridge E70, 120 fs | 2.15 | 8.42 | 1.62 | 0.91 |
| hollow E70, 52 fs | 1.76 | 14.60 | 5.30 | 0.89 |
| hollow E100, 40 fs | 2.09 | 17.13 | 3.25 | 5.37 |
| hollow E70, 36 fs | 1.66 | 17.37 | 1.07 | 4.87 |
| hollow E100, 24 fs | 1.67 | 37.23 | 0.50 | 1.90 |

Sign check of the radial force on Ar (`ar_force_check.py`; ratio model/DFT on the 9 frames with DFT |F_Ar| >= 1 eV/A, 1 = right, > 1 = too hard):
AIREBO+ZBL median 0.91 (0.31-1.57), zero-shot OMAT-0 1.25 (0.34-1.55), Ar-aware 0.99 (0.92-1.36). The Ar-aware model is too hard (ratio 1.18-1.36) in three of the
closest frames (bridge E70 120 fs, hollow E70 52 fs, hollow E100 40 fs), consistent with the Stage C finding that MACE is about 10 % too repulsive at Ar-C of about 1.75 A.

Limits: 15 snapshots from one small sheet, AIREBO-generated geometries, 0 K starts; the Ar-C < 1.6 A region is not covered by any DFT.

## 4. Cost and stability of MACE dynamics (`md_timing.py`, float32, 8 threads, Ar impact on a flat sheet)

| atoms | dt (fs) | s/step | max abs energy drift (eV) | ms/atom/step | cost of 1 ps (h) |
|---|---|---|---|---|---|
| 73 | 0.10 | 0.38 | 0.0005 | 5.2 | 1.0 |
| 73 | 0.20 | 0.37 | 0.0029 | 5.1 | 0.5 |
| 129 | 0.10 | 0.76 | 0.0011 | 5.9 | 2.1 |
| 129 | 0.20 | 0.77 | 0.0029 | 5.9 | 1.1 |

dt = 0.2 fs was therefore used in Stage C (drift of a few meV; short test).

## 5. LAMMPS route for MACE: not working in this environment

`lammps_mace_check.py` failed at the conversion to the ML-IAP format (`create_lammps_model --format mliap`): `RuntimeError ... Unexpected key(s) in state_dict:
"products.0.symmetric_contractions.weight", "products.1.symmetric_contractions.weight"` (the e3nn-to-cuequivariance conversion in this MACE version; cuequivariance is not installed).
MACE dynamics were therefore run with ASE (Stage C). A LAMMPS route would be a speed-up only.

## 6. What this changes for the README

* The carbon-only fine-tune cannot be used for Ar; `--foundation_model_elements True` plus Ar scans is required.
* The Ar scans and the impact test are what make the model usable for Stage C; the zero-shot models fail on Ar (MPA-0 badly: energy differences off by 17 eV rms).
* The model's weakest region is the closest contact (Ar-C below about 2 A), where DFT labels are scarce or impossible.

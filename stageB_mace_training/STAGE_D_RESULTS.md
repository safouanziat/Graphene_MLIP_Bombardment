# Stage D results: nitrogen bombardment of graphene with a fine-tuned MACE model

Status: results from the committee runs of 8 Nov 2026. Numbers below are copied from the logs of this work; anything not measured is marked as such.

## 1. Question

Can a MACE model fine-tuned from omat-0 replace AIREBO for N ion impacts on graphene, with GPAW DFT as ground truth? AIREBO has no N parameters in this setup, so for N the only independent check is DFT.

## 2. Reference data (DFT labels)

- Recipe id 8c357474ed: GPAW PBE, plane waves 450 eV, k-density 5, Fermi-Dirac 0.05 eV, spin-polarised, initial moment 0.3 muB/atom (N: 3.0), cell z = 20 A.
- Distance rules for labelled frames: C-C >= 1.0 A, C-N >= 1.1 A, N-N >= 1.0 A, Ar-C >= 1.6 A. Frames closer than 1.1 A (C-N) are therefore not labelled.
- Label problem found and fixed: a sputtered or isolated C atom in a DFT frame is a free magnetic atom (isolated C: M = 2 is 1.27 eV below M = 0). It causes SCF sloshing, metastable +-4 muB states (about 2.96-2.98 eV above the unpolarised state) and failed jobs; the 20 A periodic cell also wraps far atoms back into the sheet. Fix: remove C atoms farther than 7 A from the sheet or more than 2.3 A from every other atom. The MD itself (pbc T,T,F) is not affected.
- Spin ambiguity: spin-polarised vs unpolarised labels differ by 0.25-0.5 eV/A on the N force for near-contact frames. This is a floor for any model that does not see spin.

## 3. Models

| Model | Training set | Note |
|---|---|---|
| n6 | v2 | before dense hollow frames |
| n7 | v3 (+dense hollow) | fixes the hollow-crossing regime |
| n8 | v4 (+10 late frames) | |
| n9 | v5 (+22 frames, runs r3/r4 held out for testing) | N2 reference error 56.9 meV/atom |
| n10 | v6 (= v5 + 9 DFT N2 frames) | N2 reference error 7.5 meV/atom; used for the final runs |

Common settings: omat-0 base, keep-elements, E0s {C: -1.254804, Ar: 0.038912, N: -2.203187}, all-train (validation = copies of training rows, so end-of-training numbers are fit errors only), extra fmax weight 15.

## 4. Independent tests (force error on the N, eV/A, norm/sqrt(3) convention)

| Test set | n6 | n7 | n8 | n9 | n10 |
|---|---|---|---|---|---|
| h34 (17 frames, whole impacts r3/r4 held out) | 0.464 | 0.514 | 0.511 | 0.429 | 0.426 |
| late_test (9) | 0.220 | 0.269 | 0.248 | 0.163 | 0.165 |
| dense_test (10 hollow) | 0.718 | 0.186 | 0.208 | 0.213 | 0.236 |
| val2 (10) | 1.080 | 0.362 | 0.346 | 0.363 | 0.349 |

All values recomputed on 8 Nov 2026 with `nimpact.py compare` (`table_all_models.txt`). Notes:
- The large gain from n6 to n7 on dense_test (0.718 to 0.186) and val2 (1.080 to 0.362) comes from the dense hollow frames added for n7. n8 to n10 are within noise of each other on these two sets.
- On h34 and late_test, n9 and n10 are better than n6-n8 (h34 0.43 vs 0.46-0.51; late_test 0.16 vs 0.22-0.27); n6 is better than n7 and n8 on these two near-contact sets, so the hollow-regime gain was not free.
- n9 and n10 differ by at most 0.023 eV/A on any set.
- The near-contact top/bridge regime stays at 0.4-0.5 eV/A on the N for all models; roughly 0.15-0.3 eV/A of that is the spin-label ambiguity of Section 2.
- Late bridge 50 eV frames have energy errors of about +1.5 eV.
- Test frames of one impact are 5 fs apart and nearly identical, so the splits are by whole impact.

## 5. N2 reference and formation energies

The N2 reference was a single training row, so its energy moved between retrains (error in meV/atom: n2 4.5, n3 42, n4 14, n5 38, n6 37, n7 9.2, n8 2.4, n9 56.9). Nine DFT N2 frames (bond 1.02-1.35 A, heights 3.5-7 A, all non-magnetic) were added; n10 gives 7.5 meV/atom.

Formation energies of the unrelaxed structures with n10 (eV, MACE - DFT): N adatom bridge -0.50, N adatom top -0.27, graphitic N -0.23, V1+1N -0.23, V1+2N +0.15, V1+3N +0.31, V2+2N -0.26, V2+4N +0.43. These are fit errors of the model on N-in-sheet structures, not caused by the N2 reference (7.5 meV/atom is about 0.015 eV per N). Quote formation energies for a single N only; list the multi-N values as a known limitation.

N-scan: at 3-3.5 A above the sheet MACE is about 0.12-0.15 eV lower than DFT relative to 4.5 A.

## 6. Committee impact runs (5 runs per case, same jitter seeds for every model)

| Case | n6 | n7 | n8 | n10 | Verdict |
|---|---|---|---|---|---|
| top 20/30/50 eV | 5/5 bound in-plane | 5/5 | 5/5 | 5/5 | robust (3 C, 1 C ejected) |
| hollow 30/50 eV | 5/5 transmitted | 5/5 | 5/5 | 5/5 | robust, no C ejected |
| bridge 50 eV | 4 transmitted, 1 bound | 3, 2 | 3, 2 | 4, 1 | robust majority |
| hollow 20 eV | mostly bound, 1 transmitted | 2 bound, 1 reflected, 1 transmitted | 3 bound, 1 reflected, 1 in-plane | same as n8 | mostly bound |
| bridge 30 eV | 3 bound, 2 transmitted | mostly bound, 1 transmitted | 3 bound, 1 transmitted, 1 reflected | 4 bound, 1 transmitted | bound majority |
| bridge 20 eV | 4 bound, 1 reflected | 4 reflected, 1 bound | 3 reflected, 2 free near sheet, 0 bound | 3 bound, 2 reflected | model-dependent, no probability quoted |

Minimum N-C distances: top 0.85-1.04 A, bridge 50 eV 0.90-1.06 A, bridge 20-30 eV 1.08-1.17 A, hollow 20 eV 1.14-1.54 A, hollow 30-50 eV 1.27-1.40 A.

## 7. Limitations

1. Top and bridge-50 eV impacts pass through N-C distances below the 1.1 A DFT-trusted limit; force accuracy there is not verified by DFT.
2. Spin: labels are spin-polarised; a spinless model has a 0.25-0.5 eV/A floor on near-contact frames.
3. Five runs per case and one set of jitter seeds: no outcome probabilities. The bridge 20 eV outcome is reported as model-dependent.
4. Multi-N vacancy formation energies are off by 0.3-0.4 eV.
5. No AIREBO comparison for N.

## 8. Open items

- `nimpact.py` `to_dft_frame` still parks far C atoms; it must remove far/isolated C before any new run that writes DFT frames.
- Stage A inconsistencies to check: SW 5x5 AIREBO 6.02 vs 5.81 eV, frenkel w10 numbers, `01_relax_support.py` Tersoff vs README, ZBL blend in the MD repo; `scans.py` duplicate guard; duplicate s=0 row in pushscan.
- Optional: final model trained on all data including h34.

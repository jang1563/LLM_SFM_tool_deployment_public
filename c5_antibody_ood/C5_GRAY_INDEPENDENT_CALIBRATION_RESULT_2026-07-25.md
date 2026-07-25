# Stage B C5 Independent Calibration Replay

This no-model replay calibrates AF3 ranking-score trust routing on the
Hitawala-Gray post-cutoff Ab-Ag/Nb-Ag benchmark, after removing every
PDB ID present in the Fromm panel. It then applies only a certified
antibody gate to the already frozen Fromm evaluation rows.

This is independent-source published-label evidence, not a blinded
hidden benchmark or a new structure-prediction result.

## Intake And Overlap

- Source rows: `1900`; complete bound rows: `1565` across `108` targets.
- Source selection before overlap exclusion: `108` targets.
- Excluded overlap: `9` PDB IDs / `11` complex copies.
- Independent calibration cohort: `97` targets (`44` antibody, `53` nanobody); residual PDB overlap `0`.
- Selection ties: `55` at ranking score, `30` after ipTM-HA.

## Independent Calibration

| Format | Targets | Trust-all failures | Fixed 0.80 failures | Certified gate | Closest upper bound |
| --- | ---: | ---: | ---: | --- | ---: |
| `antibody` | 44 | 25 | 5 | no | 0.644 |
| `nanobody` | 53 | 34 | 2 | no | 0.638 |

The finite candidate family is the pre-existing `0.50-0.99` grid
with a uniform Hoeffding/union-bound correction (`delta=0.10`).
Gray `ranking_score` and Fromm `ranking_confidence` are aligned by
AF3 score name and range; exact model-version equivalence is not
claimed.

## Locked Fromm Replay

| Policy | Trusted | Failures among trusted | Failure rate | Coverage |
| --- | ---: | ---: | ---: | ---: |
| `trust_all` | 55/55 | 28 | 0.509 | 1.000 |
| `fixed_ranking_score_0_80` | 21/55 | 4 | 0.190 | 0.382 |
| `independent_calibration_gate` | 0/55 | 0 | 0.000 | 0.000 |

## Decision

- Independent source intake, overlap exclusion, privacy projection, and canonical trajectory validation: `pass`.
- Antibody ranking-score trust certificate: `not certified`.
- Nanobody ranking-score trust certificate: `not certified`.
- External trust remains disabled unless the independent antibody certificate passes.
- Model training, DPO, and RLVR remain closed.

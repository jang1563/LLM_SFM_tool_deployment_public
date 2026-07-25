# Stage B C5 Source-Backed Public-Score Pilot

This no-model replay evaluates target-grouped trust routing over the
published AlphaFold 3 antibody-antigen score table from Fromm et al.
It is a published-label replay, not an independent hidden benchmark.

## Intake And Split

- Source rows: `22000` across `110` targets.
- Selected targets: `110`; DockQ successes: `55` (`0.500`).
- Top-ranking-score ties: `64` targets.
- Frozen target split: `55` calibration / `55` evaluation; overlap `0`.

## Frozen Evaluation

| Policy | Trusted | Failures among trusted | Failure rate | Coverage |
| --- | ---: | ---: | ---: | ---: |
| `trust_all` | 55/55 | 28 | 0.509 | 1.000 |
| `generic_fixed_iptm_0_80` | 20/55 | 3 | 0.150 | 0.364 |
| `regime_specific_hoeffding` | 0/55 | 0 | 0.000 | 0.000 |
| `fail_closed` | 0/55 | 0 | 0.000 | 0.000 |

## Decision

- Source intake, privacy projection, split isolation, and canonical trajectory validation: `pass`.
- No regime-specific threshold is certified at the primary `alpha=0.30` gate.
- The fixed `ipTM >= 0.80` baseline is not a calibrated general-PPI transfer gate.
- Model training and DPO/RLVR remain closed.

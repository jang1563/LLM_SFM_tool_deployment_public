# Stage A Enum Corrective SFT/Margin Smoke

Date: 2026-07-05

Purpose: test whether a tiny chosen-only `enum_action` corrective SFT run can
make held-out oracle enum/action targets score above the observed
`ground` / `supported` collapse target. This is a margin diagnostic over
corrective pairs, not DPO/RLVR.

## Result

| Metric | Value |
| --- | ---: |
| Train corrective pairs | 16 |
| Held-out corrective pairs | 4 |
| Held-out margin wins | 2/4 |
| Margin accuracy | 0.500 |
| Mean chosen score | -0.411532 |
| Mean rejected score | -0.390846 |
| Mean margin | -0.020685 |
| Minimum margin | -0.106149 |

## Held-Out Margin Readout

Margin is `chosen_score - rejected_score`, where rejected is always
`ground` / `supported`.

| Case family | Chosen pair | Margin | Win |
| --- | --- | ---: | ---: |
| contradicted_or_mixed_endpoint_claim | `reject` / `contradicted` | 0.014703 | 1 |
| insufficient_evidence | `defer` / `insufficient` | -0.052623 | 0 |
| related_evidence_requires_verification | `verify` / `insufficient` | 0.061328 | 1 |
| invalid_value_attribution_failure | `flag` / `invalid_value` | -0.106149 | 0 |

## Interpretation

The corrective SFT path creates partial signal, but it does not solve the
collapse. Contradicted and verification-needed held-out pairs now beat
`ground` / `supported`, while insufficient-evidence and invalid-value cases
still lose to the collapse target.

This keeps the project in Stage A diagnosis mode. The next useful step is not
DPO/RLVR or broad progression to `tool_query`; it is a tighter enum-specific
diagnosis or corrective data pass for insufficient and invalid-value cases.

## Trace

- Compact JSON summary: `post_training/stage_a_enum_corrective_sft_cayuga_summary_2026-07-05.json`
- Runner: `post_training/run_stage_a_enum_corrective_sft_smoke.py`
- Raw `report.json`, `margins.jsonl`, `margin_report.json`, model state, and
  Slurm logs remain untracked under ignored `post_training/runs/` in the
  cluster working copy.
- Report SHA-256:
  `a9cf5e537b9457d42a6a7c9e922419b9466ae3a3c1e3612e9c224caddb6533ef`
- Margin report SHA-256:
  `65a4f8ae840ef277faed74b8362a89b9b7ec4a9b44d370c9407472915eab1e15`
- Margins JSONL SHA-256:
  `a308b1fd341352445bb7e897740a4e36851b93f096b7a495d12c79f100dae37b`

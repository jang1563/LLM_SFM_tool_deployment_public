# Stage A Enum-Action Candidate SFT Cayuga Smoke

Date: 2026-07-05

Purpose: re-run the first strict `enum_action` component after replacing
free-form JSON generation with finite candidate scoring. The run tests whether
removing schema/enum drift is enough for the model to choose the correct
`(action, evidence_status)` pair.

## Run

- Cluster: Cayuga Slurm GPU job
- Runner: `post_training/run_stage_a_strict_component_sft_smoke.py`
- Decode mode: `enum_candidate_score`
- Component: `enum_action`
- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- Prompt contract: `stage_a_v2_strict`
- Train split: `post_training/stage_a_strict_component_targets_train_v1.jsonl`
- Held-out split: `post_training/stage_a_strict_component_targets_heldout_v1.jsonl`
- Cases: 5 held-out Stage A cases, one per case family
- Training: 20 steps, batch size 1, last transformer layer only

Raw prediction JSONL, candidate scores, trainable state, logs, and full
evaluator output remain untracked under ignored `post_training/runs/` in the
cluster working copy. This file records only the compact public-safe result.

## Result

| Metric | Free-form | Candidate scoring |
| --- | ---: | ---: |
| Held-out pass rate | 0/5 | 1/5 |
| Mean score | 0.250 | 0.800 |
| `target_keys` accuracy | 0.0 | 1.0 |
| `enum_validity` accuracy | 0.0 | 1.0 |
| `exact_match` accuracy | 0.0 | 0.2 |

Candidate scoring fixes the structural failure: every output now has exactly
`action` and `evidence_status`, and every value is in the allowed enum.

Remaining violation:

| Violation | Count |
| --- | ---: |
| `target_mismatch` | 4 |

## Held-Out Readout

| Case family | Expected | Predicted | Pass |
| --- | --- | --- | ---: |
| supported_negative_evidence | `ground` / `supported` | `ground` / `supported` | 1 |
| contradicted_or_mixed_endpoint_claim | `reject` / `contradicted` | `ground` / `supported` | 0 |
| insufficient_evidence | `defer` / `insufficient` | `ground` / `supported` | 0 |
| related_evidence_requires_verification | `verify` / `insufficient` | `verify` / `invalid_value` | 0 |
| invalid_value_attribution_failure | `flag` / `invalid_value` | `ground` / `supported` | 0 |

## Interpretation

This is a partial repair, not a deployable pass. The harness now separates two
failure classes:

- schema/enum validity is solved by finite candidate scoring;
- scientific enum-pair selection is still weak and biased toward
  `ground` / `supported`.

The next useful step is a slice-specific ranking or calibration diagnostic for
enum-pair selection. Do not move to `tool_query`, DPO, or RLVR until the
`enum_action` gate is understood.

## Trace

- Compact JSON summary:
  `post_training/stage_a_component_enum_action_candidate_sft_cayuga_summary_2026-07-05.json`
- Eval report SHA-256:
  `3ba76d365bbb8071420e75cedb0fe2278f51e8ae4f49015b7705ff378ec61de3`
- Raw predictions SHA-256:
  `fec5e46ab5b279774c7043be98c536d7c356303ec78d753bcdd01d8c17f37750`
- Run report SHA-256:
  `405e6ed6b04b14855543264860f6bb9a65a85f9d11d28206bc865cd9324e6338`

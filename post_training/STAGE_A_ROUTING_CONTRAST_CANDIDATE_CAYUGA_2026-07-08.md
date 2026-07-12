# Stage A Routing Contrast Candidate-Rank Cayuga Diagnostic

Date: 2026-07-08

Purpose: check whether the positive teacher-forced routing margin repair also
transfers to finite-candidate `routing_after_loop` selection. This is still a
component-level readout, not `tool_query`, DPO, RLVR, free-form generation, or
trajectory-level scoring.

## Setup

| Setting | Value |
| --- | --- |
| Cluster job | Cayuga Slurm GPU job `[omitted]` |
| Runner | `post_training/run_stage_a_routing_contrast_sft_smoke.py` |
| Model | `Qwen/Qwen2.5-0.5B-Instruct` |
| Prompt contract | `stage_a_v2_evidence_conditioned_component` |
| Candidate policy | train-observed routing contrast pairs with visible citations |
| Candidate space | 5 action/status pairs |
| Train / held-out | 12 train pairs, 3 held-out pairs |
| Objective | chosen-target SFT plus supervised pairwise margin |
| Pairwise margin | weight 1.0, required margin 0.05 |

Raw candidate JSONL files, full reports, trainable state, and Slurm logs remain
untracked under ignored `post_training/runs/` in the Cayuga working copy. This
file records only the compact public-safe result.

## Result

| Metric | Base model | Trained model |
| --- | ---: | ---: |
| Teacher-forced held-out margin wins | 0/3 | 3/3 |
| Mean teacher-forced held-out margin | -0.116919 | 0.114900 |
| Finite-candidate exact top-1 | 0/3 | 2/3 |
| Mean finite-candidate gold rank | 3.000000 | 1.333333 |
| Mean top-gold candidate margin | 0.116919 | 0.026745 |

This is a partial transfer result. The same run that repairs all three
teacher-forced margins also improves finite-candidate routing from 0/3 to 2/3,
but insufficient-evidence routing remains unresolved.

## Held-Out Candidate Readout

| Case family | Expected pair | Base top pair | Base rank | Trained top pair | Trained rank | Trained margin |
| --- | --- | --- | ---: | --- | ---: | ---: |
| insufficient_evidence | `defer` / `insufficient` | `reject` / `contradicted` | 5 | `verify` / `insufficient` | 2 | 0.080234 |
| related_evidence_requires_verification | `verify` / `insufficient` | `reject` / `contradicted` | 2 | `verify` / `insufficient` | 1 | 0.000000 |
| invalid_value_attribution_failure | `flag` / `invalid_value` | `ground` / `supported` | 2 | `flag` / `invalid_value` | 1 | 0.000000 |

## Interpretation

The routing contrast repair transfers beyond pairwise teacher-forced margins,
but only partially. It fixes the verification-needed and invalid-value held-out
candidate decisions and moves insufficient evidence from rank 5 to rank 2.
However, `defer` / `insufficient` still loses to `verify` / `insufficient`, so
the remaining bottleneck is the defer-vs-verify distinction under insufficient
evidence.

This keeps the long-term gate intact: do not start `tool_query`, DPO, RLVR,
release tagging, or Hugging Face publication from this result alone. The next
repair should target insufficient-evidence routing calibration or a constrained
defer-vs-verify gate, then rerun candidate and saved-prediction checks.

## Trace

- Compact JSON summary:
  `post_training/stage_a_routing_contrast_candidate_cayuga_summary_2026-07-08.json`
- Runner: `post_training/run_stage_a_routing_contrast_sft_smoke.py`
- Raw reports, candidate JSONL files, model state, and Slurm logs remain
  untracked under ignored `post_training/runs/` in the cluster working copy.
- Run report SHA-256:
  `f995d51f6ec4bd2b7bb51916dbf26d0da4ed70f1f147c48db9f0bc70f15553d0`
- Base candidate report SHA-256:
  `28f32a0608c0c9962bb869141f4d65c41bce34aa7e90b9af36a88d97feaee663`
- Trained candidate report SHA-256:
  `d72220fd0384319d39409851b863eb662ce32bd7b1bfcb4eea143d1177a802bf`
- Margin delta report SHA-256:
  `660214a11f9e7092c5c3758ac328f16e5ac2bd31564fc4b650971550256a4d61`

# Stage A Evidence-Conditioned Enum-Action Cayuga Result

Date: 2026-07-06

Purpose: test whether the evidence-conditioned component substrate repairs the
`enum_action` candidate-selection failure. This is a component-level 5-way
observed-pair scorer over public-safe synthetic evidence state, not free
generation, `tool_query`, `routing_after_loop`, DPO, or RLVR.

## Run

- Cluster: Cayuga Slurm GPU job `[omitted]`
- Runner: `post_training/run_stage_a_strict_component_sft_smoke.py`
- Component: `enum_action`
- Decode mode: `enum_observed_pair_score`
- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- Prompt contract: `stage_a_v2_evidence_conditioned_component`
- Target dataset: `negbiodb_ct_stage_a_evidence_conditioned_component_targets_v1`
- Train / held-out: 20 train rows, 5 held-out rows
- Candidate space: 5 train-observed `(action, evidence_status)` pairs
- Training: 20 steps, batch size 1, last transformer layer only

Raw predictions, candidate scores, trainable state, logs, and full evaluator
output remain untracked under ignored `post_training/runs/` in the Cayuga
working copy. This file records only the compact public-safe result.

## Result

| Metric | Value |
| --- | ---: |
| Held-out pass rate | 1/5 |
| Mean score | 0.800 |
| `target_keys` accuracy | 1.0 |
| `enum_validity` accuracy | 1.0 |
| `exact_match` accuracy | 0.2 |
| `target_mismatch` violations | 4 |

The run lowers teacher-forced training loss, but held-out top-1 selection still
collapses to `ground` / `supported` for all five held-out cases.

## Held-Out Readout

| Case family | Expected | Predicted | Gold rank | Pass |
| --- | --- | --- | ---: | ---: |
| supported_negative_evidence | `ground` / `supported` | `ground` / `supported` | 1 | 1 |
| contradicted_or_mixed_endpoint_claim | `reject` / `contradicted` | `ground` / `supported` | 2 | 0 |
| insufficient_evidence | `defer` / `insufficient` | `ground` / `supported` | 4 | 0 |
| related_evidence_requires_verification | `verify` / `insufficient` | `ground` / `supported` | 3 | 0 |
| invalid_value_attribution_failure | `flag` / `invalid_value` | `ground` / `supported` | 4 | 0 |

## Interpretation

This is a negative/partial result. The evidence-conditioned substrate fixes the
data-interface problem found by the visibility audit, but evidence visibility
alone does not repair `enum_action` candidate top-1 selection. The model still
prefers `ground` / `supported` even when public-safe evidence state is visible.

Next, run evidence-conditioned `routing_after_loop` to test whether post-tool
evidence/action routing is learnable when observed tool-result content is
visible. Keep `tool_query`, DPO, RLVR, release tagging, and Hugging Face
publication gated until compact component results justify moving.

## Trace

- Compact JSON summary:
  `post_training/stage_a_evidence_enum_action_observed_pair_cayuga_summary_2026-07-06.json`
- Raw `report.json` SHA-256:
  `865926c3d494d3e995f6b06481feb875b777367e7f4d000d1fa74ee57ef674e8`
- Raw `eval_report.json` SHA-256:
  `df1b300ffe80428ebc991f90e76d358872ba906af1cfa414d72aa796ae4a3f9f`
- Raw `predictions.jsonl` SHA-256:
  `1a7394b43c820f337ab249c6ac07b63e124530fa9532b6014568384ce0e10874`

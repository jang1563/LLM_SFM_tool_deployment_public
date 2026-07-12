# Stage A Cayuga HF Chat Baseline

Date: 2026-07-04

Purpose: run the first tiny real Stage A saved-prediction baseline on cluster
compute, then score the saved artifact offline before making any post-training
claim.

## Run

- Cluster: Cayuga Slurm GPU job
- GPU class: A40
- Model: `Qwen/Qwen2.5-1.5B-Instruct`
- Producer mode: `hf_chat`
- Split: `post_training/stage_a_sft_heldout_v1.jsonl`
- Cases: 5 held-out Stage A cases, one per case family
- Run ID: `stage_a_cayuga_qwen15b_heldout_2026-07-04_r2`

Raw prediction JSONL and full evaluator output stay out of git under ignored
`post_training/runs/` in the cluster working copy. This file records only the
compact public-safe result.

## Result

| Metric | Value |
| --- | ---: |
| Expected cases | 5 |
| Predictions received | 5 |
| Passed cases | 0/5 |
| Mean score | 0.114 |

Gate accuracy:

| Gate | Accuracy |
| --- | ---: |
| `action_allowed` | 0.2 |
| `attribution` | 0.2 |
| `evidence_status` | 0.2 |
| `policy_compliance` | 0.2 |
| `query_filter_completeness` | 0.0 |
| `required_tool_sequence` | 0.0 |
| `terminal_action` | 0.0 |

Violations:

| Violation | Count |
| --- | ---: |
| `prediction_parse_error` | 4 |
| `missing_required_tool_sequence` | 1 |
| `query_filter_missing_required_field` | 1 |
| `terminal_action_mismatch` | 1 |

The main parse failure was `Unknown evidence_status: 'sourced'` in 4/5 cases.
The one non-parse-error case correctly used `insufficient` evidence status, but
still failed the required tool sequence, terminal action, and query-field gates.

## Interpretation

This is a negative baseline, not a model-quality claim. It proves that the
artifact-first cluster path works end to end and that the evaluator catches
schema drift, unsupported evidence labels, missing required tool sequences, and
incomplete query arguments before any SFT/DPO/RLVR interpretation.

Next experiment should tighten the generation contract before training:

- constrain `evidence_status` to the evaluator enum;
- require complete tool-call objects with required query fields;
- keep terminal actions limited to the Stage A action schema;
- rerun the same held-out scorer before adding post-training claims.

## Trace

- Compact JSON summary:
  `post_training/stage_a_cayuga_hf_chat_baseline_summary_2026-07-04.json`
- Eval report SHA-256:
  `0da4ec34491e3d2077633a841371cac89fae5d0527bdc1ff227a226280dec31f`
- Raw predictions SHA-256:
  `6c6857c7d0d3cec622ab0d3d93a73af041a7bded1d6343128eaaff71b9e17a13`

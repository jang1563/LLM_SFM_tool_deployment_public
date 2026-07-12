# Stage A Cayuga Strict Contract Baseline

Date: 2026-07-04

Purpose: test whether a stricter model-visible output contract improves the
first Cayuga HF chat baseline without post-hoc repair or hidden-label leakage.

## Run

- Cluster: Cayuga Slurm GPU job
- GPU class: A40
- Model: `Qwen/Qwen2.5-1.5B-Instruct`
- Producer mode: `hf_chat`
- Prompt contract: `stage_a_v2_strict`
- Split: `post_training/stage_a_sft_heldout_v1.jsonl`
- Cases: 5 held-out Stage A cases, one per case family
- Run ID: `stage_a_cayuga_qwen15b_strict_heldout_2026-07-04`

Raw prediction JSONL and full evaluator output remain untracked under ignored
`post_training/runs/` in the cluster working copy. This file records only the
compact public-safe result.

## Result

| Metric | Basic contract | Strict contract |
| --- | ---: | ---: |
| Passed cases | 0/5 | 0/5 |
| Mean score | 0.114 | 0.372 |
| Parse errors | 4 | 0 |

Strict contract gate accuracy:

| Gate | Accuracy |
| --- | ---: |
| `action_allowed` | 1.0 |
| `attribution` | 0.6 |
| `evidence_status` | 0.2 |
| `policy_compliance` | 0.6 |
| `query_filter_completeness` | 0.0 |
| `required_tool_sequence` | 0.0 |
| `terminal_action` | 0.2 |

Strict contract violations:

| Violation | Count |
| --- | ---: |
| `missing_required_tool_sequence` | 5 |
| `query_filter_missing_required_field` | 5 |
| `evidence_status_mismatch` | 4 |
| `terminal_action_mismatch` | 4 |
| `missing_required_attribution` | 2 |
| `contradicted_claim_requires_reject_or_flag` | 1 |
| `invalid_value_requires_reject_or_flag` | 1 |

## Interpretation

The strict contract solved the shallow formatting problem: the evaluator no
longer saw unsupported `evidence_status` parse errors. It did not solve the
scientific tool-use problem. The model mostly produced allowed, parseable JSON,
but collapsed toward `verify` plus `supported`, omitted the full required tool
sequence, and did not satisfy query-field gates.

This is the right failure mode for the next research sprint: Stage A needs
trajectory-level training or process supervision for ordered tool use and
evidence/action routing. Prompt-format tightening alone is not enough.

## Next Experiment

- Build a small contrast set where chosen trajectories complete the full
  `nullatlas_*` tool sequence and rejected trajectories stop after one tool.
- Add process labels for action/status routing:
  `supported -> ground`, `contradicted/invalid_value -> reject/flag`,
  `insufficient -> defer/verify`.
- Rerun the same strict held-out scorer after SFT or preference tuning.

## Trace

- Compact JSON summary:
  `post_training/stage_a_cayuga_strict_contract_summary_2026-07-04.json`
- Eval report SHA-256:
  `d804cad64b56d56e0eb32a8f9deff19f2cdb77500c55431f550111d73524e57f`
- Raw predictions SHA-256:
  `8e602700e2cdb0a580f35ae86831b29e8f8308fab3e4b1f791346796d9c00c75`

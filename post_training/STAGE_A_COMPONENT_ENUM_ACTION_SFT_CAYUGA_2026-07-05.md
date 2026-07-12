# Stage A Component Enum-Action SFT Cayuga Smoke

Date: 2026-07-05

Purpose: run the first strict component-slice SFT smoke on Cayuga. This is the
first checkpoint in the research-first execution board: measure `enum_action`
before moving to `tool_query`, `routing_after_loop`, DPO, or RLVR.

## Run

- Cluster: Cayuga Slurm GPU job
- Runner: `post_training/run_stage_a_strict_component_sft_smoke.py`
- Component: `enum_action`
- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- Prompt contract: `stage_a_v2_strict`
- Train split: `post_training/stage_a_strict_component_targets_train_v1.jsonl`
- Held-out split: `post_training/stage_a_strict_component_targets_heldout_v1.jsonl`
- Cases: 5 held-out Stage A cases, one per case family
- Training: 20 steps, batch size 1, last transformer layer only

Raw prediction JSONL, trainable state, logs, and full evaluator output remain
untracked under ignored `post_training/runs/` in the cluster working copy. This
file records only the compact public-safe result.

## Result

| Metric | Value |
| --- | ---: |
| Held-out pass rate | 0/5 |
| Mean score | 0.250 |
| Train loss delta | -0.491 |
| `target_keys` accuracy | 0.0 |
| `enum_validity` accuracy | 0.0 |
| `exact_match` accuracy | 0.0 |

Violations:

| Violation | Count |
| --- | ---: |
| `target_key_mismatch` | 5 |
| `enum_value_invalid` | 5 |
| `target_mismatch` | 5 |

The outputs were parseable JSON, but all five held-out cases included an extra
`tool` key and used the disallowed evidence-status value `valid`.

## Interpretation

This is a useful negative component result. The model can move the training loss
down, but the `enum_action` slice still fails the public strict contract. The
next scientific step is not DPO or RLVR. It is to fix constrained enum output or
the enum target format, then re-run this same component smoke.

## Trace

- Compact JSON summary:
  `post_training/stage_a_component_enum_action_sft_cayuga_summary_2026-07-05.json`
- Eval report SHA-256:
  `11ab7371ff1f1ef25ac494ade1aa258b306e5b99c48942de20303bfbb63a8fac`
- Raw predictions SHA-256:
  `af5f5df2cac7c9955a47d98ad9676b99c2de1f446424bd4cc7e0e860a005b6c5`
- Run report SHA-256:
  `fdb2fa57cd2e73e6885c13d7c5d603ef0810433409066c4bc0f7a5cd76b2ef6e`

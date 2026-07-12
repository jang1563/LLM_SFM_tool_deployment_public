# Stage A Strict-Contract SFT Cayuga Smoke

Date: 2026-07-04

Purpose: run the first cluster-side strict-contract SFT smoke after the strict
prompt-contract baseline. The goal was not to claim model improvement, but to
test whether tiny last-layer SFT on the strict JSON targets changes held-out
Stage A trajectory behavior.

## Runs

- Cluster: Cayuga Slurm GPU job
- Runner: `post_training/run_stage_a_strict_contract_sft_smoke.py`
- Prompt contract: `stage_a_v2_strict`
- Train split: `post_training/stage_a_strict_contract_sft_train_v1.jsonl`
- Held-out split: `post_training/stage_a_strict_contract_sft_heldout_v1.jsonl`
- Cases: 5 held-out Stage A cases, one per case family
- Training: 20 steps, batch size 1, last transformer layer only

Raw prediction JSONL, trainable states, and full evaluator output remain
untracked under ignored `post_training/runs/` in the cluster working copy. This
file records only the compact public-safe result.

## Result

| Model | Train loss delta | Passed | Mean score | Parse errors |
| --- | ---: | ---: | ---: | ---: |
| `Qwen/Qwen2.5-0.5B-Instruct` | -0.135 | 0/5 | 0.000 | 5 |
| `Qwen/Qwen2.5-1.5B-Instruct` | -0.079 | 0/5 | 0.372 | 0 |

The 0.5B run learned enough format to emit JSON-shaped outputs, but all five
held-out cases used the disallowed evidence-status value `verified`.

The 1.5B run produced parseable outputs, but did not improve over the prior
strict prompt-only baseline. It remained at 0/5 pass and mean score 0.372.

1.5B gate accuracy:

| Gate | Accuracy |
| --- | ---: |
| `action_allowed` | 1.0 |
| `attribution` | 0.6 |
| `evidence_status` | 0.2 |
| `policy_compliance` | 0.6 |
| `query_filter_completeness` | 0.0 |
| `required_tool_sequence` | 0.0 |
| `terminal_action` | 0.2 |

1.5B violations:

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

This is a negative SFT smoke result. Teacher-forced train loss decreased, but
held-out trajectory gates did not improve. The result strengthens the project
claim that Stage A should be evaluated as tool trajectories, not by train loss
or JSON appearance.

The likely next diagnostic is not DPO or RLVR. The next useful step is to
separate three failure sources:

- enum/action generation under a constrained decoder or candidate scorer;
- full ordered tool-loop generation with required query fields;
- evidence-status and terminal-action routing after the tool loop is present.

## Trace

- Compact JSON summary:
  `post_training/stage_a_strict_sft_cayuga_smoke_summary_2026-07-04.json`
- 0.5B eval report SHA-256:
  `c07570a355ca5844f74b47ac26b6cbb207286ea8787ff75cbb1399bb19ab1110`
- 0.5B raw predictions SHA-256:
  `971507bb9950c1169101c6c8e1ac3d89c98b4c00e609a276c2c7df11e4cb2fd0`
- 1.5B eval report SHA-256:
  `ffa47bd751fdd601b62d411368ff828c2302f066e7ce6f1711b100b81f064769`
- 1.5B raw predictions SHA-256:
  `e191d23d9a0900dc75a750575defe4e724a71a0fcb66d1830861a84e29fa6f5a`

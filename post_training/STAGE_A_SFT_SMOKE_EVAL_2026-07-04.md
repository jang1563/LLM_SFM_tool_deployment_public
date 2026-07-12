# Stage A SFT Smoke/Eval Harness

Date: 2026-07-04

Purpose: establish the no-API local evaluation shape for the next Stage A SFT
experiment. This is not a trained-model result. It is a deterministic harness
for scoring train/held-out Stage A trajectory mechanics by evaluator gate.

## Command

```bash
python post_training/run_stage_a_sft_smoke_eval.py \
  --out post_training/stage_a_sft_smoke_eval_summary_2026-07-04.json
```

## Policies

| policy | interpretation |
|---|---|
| `oracle_replay` | Upper-bound replay of each row's target trajectory. |
| `nearest_train_replay` | Nearest train prompt by visible claim text, replayed on the eval input ID. |
| `train_majority_replay` | Train-split majority terminal action/evidence status with the common full tool loop. |
| `self_answer` | No-tool shortcut baseline. |

## Result

| split | policy | cases | passed | mean score |
|---|---|---:|---:|---:|
| train | `oracle_replay` | 20 | 20 | 1.000 |
| train | `nearest_train_replay` | 20 | 20 | 1.000 |
| train | `train_majority_replay` | 20 | 0 | 0.771 |
| train | `self_answer` | 20 | 0 | 0.229 |
| heldout | `oracle_replay` | 5 | 5 | 1.000 |
| heldout | `nearest_train_replay` | 5 | 0 | 0.657 |
| heldout | `train_majority_replay` | 5 | 0 | 0.771 |
| heldout | `self_answer` | 5 | 0 | 0.229 |

Held-out gate accuracy:

| policy | tool sequence | query complete | evidence status | terminal action | attribution | policy compliance |
|---|---:|---:|---:|---:|---:|---:|
| `oracle_replay` | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| `nearest_train_replay` | 1.000 | 1.000 | 0.200 | 0.200 | 0.600 | 0.600 |
| `train_majority_replay` | 1.000 | 1.000 | 0.400 | 0.400 | 0.600 | 1.000 |
| `self_answer` | 0.000 | 0.000 | 0.000 | 0.000 | 0.600 | 0.000 |

## Interpretation

The harness cleanly separates two kinds of behavior:

- tool/query mechanics can pass under deterministic replay policies;
- evidence status, terminal action, and attribution still fail without
  case-specific evidence routing.

This is the right no-API substrate for the next real SFT experiment. A useful
Stage A SFT checkpoint should improve held-out terminal action and evidence
status over `train_majority_replay` without regressing tool sequence, query
completeness, attribution, or policy compliance.

## Boundary

This report is not a live model, API, or HPC result. It is a deterministic
smoke/eval harness over tracked Stage A artifacts. It should be used as the
baseline report shape for future SFT, preference/process, and RLVR diagnostics.

# SFT Boundary-Rationale Held-Out Oracle-Rationale Ablation: 2026-06-27

This file records an eval-only ablation using the already-trained boundary-rationale fold states.
The ablation inserts the same oracle `BOUNDARY_RATIONALE` prompt into each held-out row before the final `submit_decision` target.
Raw run artifacts are under `post_training/runs/` and ignored by git.

## Commands

```bash
python3 post_training/run_sft_boundary_rationale_ablation.py
python3 post_training/summarize_sft_boundary_rationale_ablation.py
```

Eval settings:

```text
model = Qwen/Qwen2.5-0.5B-Instruct
source_state_root = post_training/runs/qwen_sft_cv4_boundary_rationale_schema_action_80_evalfast
batch_size = 2
max_length = 512
score_mode = mean
```

## Artifact Summary

```text
dataset = negbiodb_ct_native_sft_boundary_rationale_heldout_oracle_v1
strategy = heldout_oracle_boundary_rationale_ablation_v1
rationale_mode = oracle
source_boundary_manifest = post_training/negbiodb_ct_native_sft_cv4_boundary_rationale_manifest.json
```

| fold | heldout rows | heldout by class | heldout by role | evidence mismatches |
| --- | --- | --- | --- | --- |
| 0 | 10 | {"defer": 2, "flag": 2, "ground": 2, "reject": 2, "verify": 2} | {"rationale": 10} |  |
| 1 | 10 | {"defer": 2, "flag": 2, "ground": 2, "reject": 2, "verify": 2} | {"rationale": 10} |  |
| 2 | 10 | {"defer": 2, "flag": 2, "ground": 2, "reject": 2, "verify": 2} | {"rationale": 10} |  |
| 3 | 10 | {"defer": 2, "flag": 2, "ground": 2, "reject": 2, "verify": 2} | {"rationale": 10} |  |

## Aggregate Comparison

| condition | strict mean | constrained loaded mean | parse failures | defer | takeaway |
| --- | --- | --- | --- | --- | --- |
| boundary rationale, normal held-out | 0.500 | 0.500 | 0 | 0/8 | best native-SFT aggregate but defer collapsed |
| held-out oracle-rationale ablation | 1.000 | 1.000 | 0 | 8/8 | tests whether inference-time rationale rescues defer |

## Ablation Folds

| fold | heldout loss | strict acc | constrained loaded | strict by class |
| --- | --- | --- | --- | --- |
| 0 | 0.0049 | 1.000 | 1.000 | defer 2/2, flag 2/2, ground 2/2, reject 2/2, verify 2/2 |
| 1 | 0.0029 | 1.000 | 1.000 | defer 2/2, flag 2/2, ground 2/2, reject 2/2, verify 2/2 |
| 2 | 0.0038 | 1.000 | 1.000 | defer 2/2, flag 2/2, ground 2/2, reject 2/2, verify 2/2 |
| 3 | 0.0041 | 1.000 | 1.000 | defer 2/2, flag 2/2, ground 2/2, reject 2/2, verify 2/2 |

Aggregate:

```text
heldout_loss_mean = 0.0039
strict_action_accuracy_mean = 1.000
strict_action_accuracy_range = 1.000..1.000
strict_parse_failures_total = 0
strict_class_accuracy = defer 8/8, flag 8/8, ground 8/8, reject 8/8, verify 8/8
constrained_loaded_accuracy_mean = 1.000
constrained_loaded_accuracy_range = 1.000..1.000
constrained_loaded_class_accuracy = defer 8/8, flag 8/8, ground 8/8, reject 8/8, verify 8/8
```

Failure pair counts:

```json
{
  "constrained_loaded": {},
  "strict": {}
}
```

## Interpretation

- This is an oracle-rationale ablation, not a deployable evaluation condition.
- The ablation fully rescues the previous `defer -> verify` collapse when the correct boundary rationale is visible at inference time.
- Because strict generation and constrained scoring both reach 1.000, the failure is not candidate scoring or schema parsing.
- The normal held-out failure is therefore best interpreted as missing inference-time boundary reasoning, not a broad need for RLVR yet.

## Next Action

Choose between a deployable rationale-at-inference design and explicit `defer` versus `verify` preference supervision. A non-oracle rationale generator or a DPO-style pair set should be tested before broader RLVR.

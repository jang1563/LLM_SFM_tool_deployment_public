# SFT Boundary-Rationale Run Results: 2026-06-26

This file records the full boundary-rationale CV rerun after the targeted curriculum-v2 negative result.
Raw run artifacts are under `post_training/runs/` and ignored by git.

## Commands

```bash
python3 post_training/build_sft_boundary_rationale_data.py

python3 post_training/run_sft_cv_sweep.py \
  --manifest post_training/negbiodb_ct_native_sft_cv4_boundary_rationale_manifest.json \
  --out-dir post_training/runs/qwen_sft_cv4_boundary_rationale_schema_action_80_evalfast \
  --skip-train-loss \
  --skip-base-constrained
```

Eval settings:

```text
model = Qwen/Qwen2.5-0.5B-Instruct
max_steps = 80
batch_size = 2
max_length = 512
train_last_layers = 2
lr = 5e-05
skip_train_loss = True
skip_base_constrained = True
```

## Artifact Summary

```text
dataset = negbiodb_ct_native_sft_boundary_rationale_v1
strategy = paired_boundary_rationale_v1
boundary_negative_actions = {"defer": ["verify"], "flag": ["ground", "reject"], "ground": ["flag", "reject"], "reject": ["ground", "flag"], "verify": ["defer"]}
```

| fold | train rows | train by class | train by role |
| --- | --- | --- | --- |
| 0 | 60 | {"defer": 12, "flag": 12, "ground": 12, "reject": 12, "verify": 12} | {"base": 30, "rationale": 30} |
| 1 | 60 | {"defer": 12, "flag": 12, "ground": 12, "reject": 12, "verify": 12} | {"base": 30, "rationale": 30} |
| 2 | 60 | {"defer": 12, "flag": 12, "ground": 12, "reject": 12, "verify": 12} | {"base": 30, "rationale": 30} |
| 3 | 60 | {"defer": 12, "flag": 12, "ground": 12, "reject": 12, "verify": 12} | {"base": 30, "rationale": 30} |

## Aggregate Comparison

| condition | strict mean | constrained loaded mean | parse failures | takeaway |
| --- | --- | --- | --- | --- |
| native CV baseline | 0.475 | 0.400 | 0 | parse-stable, action-fragile |
| native pressure CV | 0.400 | 0.450 | 0 | verify fixed, ground/defer hurt |
| curriculum v1 | 0.475 | 0.425 | 0 | flag improved, mixed balance |
| curriculum v2 targeted | 0.375 | 0.300 | 0 | negative; oversampling over-rotates |
| boundary rationale | 0.500 | 0.500 | 0 | modest best native-SFT aggregate; defer still unsolved |
| oracle balanced warm-start | 0.200 | 0.200 | 12 | negative; ground collapse |

## Boundary-Rationale CV Folds

| fold | heldout loss | strict acc | constrained base | constrained loaded | strict by class |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.1164 | 0.500 | n/a | 0.500 | defer 0/2, flag 1/2, ground 0/2, reject 2/2, verify 2/2 |
| 1 | 0.0789 | 0.600 | n/a | 0.600 | defer 0/2, flag 0/2, ground 2/2, reject 2/2, verify 2/2 |
| 2 | 0.1118 | 0.400 | n/a | 0.400 | defer 0/2, flag 2/2, ground 0/2, reject 0/2, verify 2/2 |
| 3 | 0.0932 | 0.500 | n/a | 0.500 | defer 0/2, flag 0/2, ground 1/2, reject 2/2, verify 2/2 |

Aggregate:

```text
heldout_loss_mean = 0.1001
strict_action_accuracy_mean = 0.500
strict_action_accuracy_range = 0.400..0.600
strict_parse_failures_total = 0
strict_class_accuracy = defer 0/8, flag 3/8, ground 3/8, reject 6/8, verify 8/8
constrained_base_accuracy_mean = n/a
constrained_loaded_accuracy_mean = 0.500
constrained_loaded_accuracy_range = 0.400..0.600
constrained_loaded_class_accuracy = defer 0/8, flag 3/8, ground 4/8, reject 5/8, verify 8/8
```

Failure pair counts:

```json
{
  "constrained_loaded": {
    "defer->verify": 8,
    "flag->ground": 3,
    "flag->reject": 2,
    "ground->flag": 3,
    "ground->reject": 1,
    "reject->flag": 3
  },
  "strict": {
    "defer->verify": 8,
    "flag->ground": 3,
    "flag->reject": 2,
    "ground->flag": 3,
    "ground->reject": 1,
    "ground->verify": 1,
    "reject->flag": 2
  }
}
```

## Interpretation

- Boundary-rationale SFT is the best native-SFT aggregate so far in this run family: strict mean 0.500 and constrained-loaded mean 0.500.
- It remains parse-stable with zero strict parse failures, so the issue is action-boundary learning rather than JSON formation.
- `verify` is fully recovered, and `reject` is strong in strict generation, but `defer` remains 0/8.
- This is a modest positive SFT formulation result, not a reason to jump straight to RLVR. The next useful step is row-level boundary-failure analysis or a targeted held-out prompt-side rationale ablation.

## Next Action

Row-level boundary-rationale failure analysis is now recorded in
`post_training/SFT_BOUNDARY_RATIONALE_FAILURE_ANALYSIS_2026-06-26.md`.

The next action is a held-out prompt-side rationale ablation using the existing
boundary-rationale fold states. The key question is whether `defer` fails
because the rationale is absent from held-out prompts, because the final-action
target is too short, or because `defer` and `verify` need explicit paired
preference supervision.

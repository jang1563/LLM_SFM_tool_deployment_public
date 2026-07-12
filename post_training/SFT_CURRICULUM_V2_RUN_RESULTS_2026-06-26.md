# SFT Curriculum V2 Targeted Run Results: 2026-06-26

This file records the targeted curriculum-v2 run built from persistent curriculum-v1 failures.
Raw run artifacts are under `post_training/runs/` and ignored by git.

## Commands

```bash
python3 post_training/build_sft_curriculum_v2_data.py

python3 post_training/run_sft_cv_sweep.py \
  --manifest post_training/negbiodb_ct_native_sft_cv4_curriculum_v2_manifest.json \
  --out-dir post_training/runs/qwen_sft_cv4_curriculum_v2_targeted_schema_action_80_evalfast \
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
dataset = negbiodb_ct_native_sft_curriculum_v2_targeted
strategy = contrast_family_interleave_plus_persistent_failure_targeted_v2
persistent_failure_count = 20
target_weights = {"defer": 2, "flag": 1, "ground": 2, "reject": 3, "verify": 2}
```

| fold | train rows | targeted rows | targeted families | targeted failure pairs |
| --- | --- | --- | --- | --- |
| 0 | 102 | 30 | {"target_clean_ground": 6, "target_defer_verify": 14, "target_flag_preserve": 1, "target_reject_override": 9} | {"defer->verify": 8, "flag->reject": 1, "ground->flag": 4, "ground->reject": 2, "reject->flag": 9, "verify->defer": 6} |
| 1 | 109 | 37 | {"target_clean_ground": 10, "target_defer_verify": 14, "target_flag_preserve": 1, "target_reject_override": 12} | {"defer->verify": 6, "flag->reject": 1, "ground->flag": 8, "ground->reject": 2, "reject->flag": 12, "verify->defer": 8} |
| 2 | 103 | 31 | {"target_clean_ground": 8, "target_defer_verify": 16, "target_flag_preserve": 1, "target_reject_override": 6} | {"defer->verify": 6, "flag->reject": 1, "ground->flag": 6, "ground->reject": 2, "reject->flag": 6, "verify->defer": 10} |
| 3 | 103 | 31 | {"target_clean_ground": 6, "target_defer_verify": 16, "target_reject_override": 9} | {"defer->verify": 10, "ground->flag": 6, "reject->flag": 9, "verify->defer": 6} |

## Aggregate Comparison

| condition | strict mean | constrained loaded mean | parse failures | takeaway |
| --- | --- | --- | --- | --- |
| native CV baseline | 0.475 | 0.400 | 0 | parse-stable, action-fragile |
| curriculum v1 | 0.475 | 0.425 | 0 | flag improved, mixed balance |
| curriculum v2 targeted | 0.375 | 0.300 | 0 | negative; persistent oversampling over-rotates |
| native pressure CV | 0.400 | 0.450 | 0 | verify fixed, ground/defer hurt |
| oracle balanced warm-start | 0.200 | 0.200 | 12 | ground collapse |

## Curriculum V2 CV Folds

| fold | heldout loss | strict acc | constrained base | constrained loaded | strict by class |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.1014 | 0.300 | n/a | 0.300 | defer 1/2, flag 0/2, ground 2/2, reject 0/2, verify 0/2 |
| 1 | 0.0777 | 0.200 | n/a | 0.200 | defer 0/2, flag 2/2, ground 0/2, reject 0/2, verify 0/2 |
| 2 | 0.0837 | 0.400 | n/a | 0.200 | defer 0/2, flag 1/2, ground 1/2, reject 0/2, verify 2/2 |
| 3 | 0.0948 | 0.600 | n/a | 0.500 | defer 2/2, flag 1/2, ground 0/2, reject 1/2, verify 2/2 |

Aggregate:

```text
heldout_loss_mean = 0.0894
strict_action_accuracy_mean = 0.375
strict_action_accuracy_range = 0.200..0.600
strict_parse_failures_total = 0
strict_class_accuracy = defer 3/8, flag 4/8, ground 3/8, reject 1/8, verify 4/8
constrained_base_accuracy_mean = n/a
constrained_loaded_accuracy_mean = 0.300
constrained_loaded_accuracy_range = 0.200..0.500
constrained_loaded_class_accuracy = defer 3/8, flag 4/8, ground 3/8, reject 0/8, verify 2/8
```

Failure pair counts:

```json
{
  "constrained_loaded": {
    "defer->verify": 5,
    "flag->ground": 3,
    "flag->reject": 1,
    "ground->flag": 4,
    "ground->reject": 1,
    "reject->flag": 6,
    "reject->ground": 2,
    "verify->defer": 6
  },
  "strict": {
    "defer->verify": 5,
    "flag->ground": 3,
    "flag->reject": 1,
    "ground->flag": 4,
    "ground->reject": 1,
    "reject->flag": 5,
    "reject->ground": 2,
    "verify->defer": 4
  }
}
```

## Interpretation

- Curriculum-v2 targeted oversampling is a negative diagnostic, not an improvement.
- It lowers strict mean from curriculum-v1's 0.475 to 0.375 and constrained-loaded mean from 0.425 to 0.300.
- `reject` is the clearest failure: constrained-loaded `reject` falls to 0/8 even though reject persistent failures were given the highest target weight.
- `defer`/`verify` symmetry remains: constrained scoring still has `defer->verify` 5 and `verify->defer` 6.
- More row duplication is not the right next move. The next formulation should change the representation or supervision signal, not simply amplify persistent failures.

## Next Action

Stop escalating oversampling. The next checkpoint was built as
`post_training/SFT_BOUNDARY_RATIONALE_ARTIFACTS_2026-06-26.md`: a paired
boundary-rationale SFT artifact that explicitly teaches why each near-neighbor
action is wrong. Fold0 smoke/evalfast sanity passed; run the full
boundary-rationale CV pass before moving to DPO/RLVR.

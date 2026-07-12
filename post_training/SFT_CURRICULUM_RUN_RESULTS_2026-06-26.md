# SFT Curriculum Run Results: 2026-06-26

This file records the full curriculum CV rerun after creating the contrast-family SFT artifact.
Raw run artifacts are under `post_training/runs/` and ignored by git.

## Command

```bash
python3 post_training/run_sft_cv_sweep.py \
  --manifest post_training/negbiodb_ct_native_sft_cv4_curriculum_manifest.json \
  --out-dir post_training/runs/qwen_sft_cv4_curriculum_schema_action_80
```

Shared model/eval settings:

```text
model = Qwen/Qwen2.5-0.5B-Instruct
batch_size = 2
max_length = 512
max_steps = 80
train_last_layers = 2
lr = 5e-05
score_mode = mean
```

## Aggregate Comparison

| condition | strict mean | parse failures | constrained loaded mean | class takeaway |
| --- | --- | --- | --- | --- |
| native CV baseline | 0.475 | 0 | 0.400 | ground/reject learned; verify/flag weak |
| native pressure CV | 0.400 | 0 | 0.450 | verify fixed; ground/defer hurt |
| native curriculum CV | 0.475 | 0 | 0.425 | flag improved; balance still mixed |
| oracle balanced warm-start | 0.200 | 12 | 0.200 | negative result; ground collapse |

## Curriculum CV Folds

| fold | heldout loss | strict acc | constrained base | constrained loaded | strict by class |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.0898 | 0.300 | 0.300 | 0.300 | defer 1/2, flag 1/2, ground 0/2, reject 1/2, verify 0/2 |
| 1 | 0.0560 | 0.700 | 0.200 | 0.600 | defer 0/2, flag 2/2, ground 2/2, reject 2/2, verify 1/2 |
| 2 | 0.1036 | 0.500 | 0.300 | 0.400 | defer 0/2, flag 2/2, ground 1/2, reject 0/2, verify 2/2 |
| 3 | 0.1664 | 0.400 | 0.300 | 0.400 | defer 2/2, flag 1/2, ground 0/2, reject 1/2, verify 0/2 |

Aggregate:

```text
heldout_loss_mean = 0.1040
strict_action_accuracy_mean = 0.475
strict_action_accuracy_range = 0.300..0.700
strict_parse_failures_total = 0
strict_class_accuracy = defer 3/8, flag 6/8, ground 3/8, reject 4/8, verify 3/8
constrained_base_accuracy_mean = 0.275
constrained_loaded_accuracy_mean = 0.425
constrained_loaded_accuracy_range = 0.300..0.600
constrained_loaded_class_accuracy = defer 3/8, flag 7/8, ground 2/8, reject 2/8, verify 3/8
```

## Interpretation

- Curriculum SFT restores the native baseline strict mean of 0.475 while improving strict `flag` from 3/8 in the original native CV run to 6/8.
- It avoids the pressure run's all-`verify` over-rotation and the balanced-oracle `ground` collapse, but it does not improve the overall held-out strict mean beyond the original native CV baseline.
- Constrained loaded accuracy is 0.425: above the original native baseline of 0.400, below the pressure run's 0.450, and far above the balanced-oracle negative result of 0.200.
- This is still an SFT-formulation bottleneck. DPO/RLVR should wait until row-level curriculum failures show a cleaner action boundary.

## Next Action

Row-level curriculum-failure analysis is now recorded in
`post_training/SFT_CURRICULUM_FAILURE_ANALYSIS_2026-06-26.md`.

Follow-up complete: targeted curriculum-v2 is recorded in
`post_training/SFT_CURRICULUM_V2_RUN_RESULTS_2026-06-26.md`; it is a negative
oversampling result.

The current next implementation step is boundary-rationale or paired contrast
SFT, not another row-duplication pass and not DPO/RLVR yet.

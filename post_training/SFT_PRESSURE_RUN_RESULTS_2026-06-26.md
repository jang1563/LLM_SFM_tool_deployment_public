# SFT Pressure Run Results: 2026-06-26

This file records the full rerun after creating the pressure-SFT artifacts.
Raw run artifacts are under `post_training/runs/` and ignored by git.

## Commands

```bash
python3 post_training/run_sft_cv_sweep.py \
  --manifest post_training/negbiodb_ct_native_sft_cv4_pressure_manifest.json \
  --out-dir post_training/runs/qwen_sft_cv4_pressure_schema_action_80

python3 post_training/run_sft_oracle_warmstart.py \
  --train-sft post_training/negbiodb_ct_oracle_sft_balanced_v1.jsonl \
  --train-limit 700 \
  --out-dir post_training/runs/qwen_oracle_balanced_warmstart_cvheldout
```

Shared model/eval settings:

```text
model = Qwen/Qwen2.5-0.5B-Instruct
batch_size = 2
max_length = 512
train_last_layers = 2
lr = 5e-5
score_mode = mean
```

## Aggregate Comparison

| condition | train data | strict mean | strict range | parse failures | constrained loaded mean | constrained loaded range |
| --- | --- | --- | --- | --- | --- | --- |
| native pressure CV | 54/fold | 0.400 | 0.400..0.400 | 0 | 0.450 | 0.300..0.700 |
| oracle balanced warm-start | 700 | 0.200 | 0.200..0.200 | 12 | 0.200 | 0.200..0.200 |

## Native Pressure CV

| fold | heldout loss | strict acc | constrained base | constrained loaded | strict by class |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.1729 | 0.400 | 0.300 | 0.300 | defer 0/2, flag 1/2, ground 0/2, reject 1/2, verify 2/2 |
| 1 | 0.1073 | 0.400 | 0.200 | 0.700 | defer 0/2, flag 0/2, ground 0/2, reject 2/2, verify 2/2 |
| 2 | 0.1127 | 0.400 | 0.300 | 0.400 | defer 0/2, flag 2/2, ground 0/2, reject 0/2, verify 2/2 |
| 3 | 0.1415 | 0.400 | 0.300 | 0.400 | defer 0/2, flag 1/2, ground 0/2, reject 1/2, verify 2/2 |

Aggregate:

```text
heldout_loss_mean = 0.1336
strict_action_accuracy_mean = 0.400
strict_action_accuracy_range = 0.400..0.400
strict_parse_failures_total = 0
strict_class_accuracy = defer 0/8, flag 4/8, ground 0/8, reject 4/8, verify 8/8
constrained_base_accuracy_mean = 0.275
constrained_loaded_accuracy_mean = 0.450
constrained_loaded_class_accuracy = defer 0/8, flag 6/8, ground 1/8, reject 3/8, verify 8/8
```

## Oracle Balanced Warm Start

```text
train_limit = 700
train_first_loss = 4.3628
train_last_loss = 0.0021
train_loss_delta = -4.3607
```

| eval set | heldout loss | strict acc | parse failures | constrained base | constrained loaded | strict by class |
| --- | --- | --- | --- | --- | --- | --- |
| fold0 heldout | 1.9398 | 0.200 | 3 | 0.300 | 0.200 | defer 0/2, flag 0/2, ground 2/2, reject 0/2, verify 0/2 |
| fold1 heldout | 1.9516 | 0.200 | 3 | 0.200 | 0.200 | defer 0/2, flag 0/2, ground 2/2, reject 0/2, verify 0/2 |
| fold2 heldout | 1.9449 | 0.200 | 3 | 0.300 | 0.200 | defer 0/2, flag 0/2, ground 2/2, reject 0/2, verify 0/2 |
| fold3 heldout | 1.9268 | 0.200 | 3 | 0.300 | 0.200 | defer 0/2, flag 0/2, ground 2/2, reject 0/2, verify 0/2 |

Aggregate:

```text
heldout_loss_mean = 1.9408
strict_action_accuracy_mean = 0.200
strict_action_accuracy_range = 0.200..0.200
strict_parse_failures_total = 12
strict_class_accuracy = defer 0/8, flag 0/8, ground 8/8, reject 0/8, verify 0/8
constrained_base_accuracy_mean = 0.275
constrained_loaded_accuracy_mean = 0.200
constrained_loaded_class_accuracy = defer 0/8, flag 0/8, ground 8/8, reject 0/8, verify 0/8
```

## Interpretation

- Native pressure training does not improve strict generation over the first native CV run, but it slightly improves constrained loaded accuracy from 0.400 to 0.450.
- The pressure effect is specific: `verify` becomes reliable in strict generation, while `defer` and `ground` collapse to 0/8 under strict generation.
- Balanced-oracle warm start is a negative result: it overfits the 700-row teacher artifact and collapses native held-out predictions toward `ground`, with 12 strict parse failures.
- DPO/RLVR should still wait. The next bottleneck is SFT formulation/curriculum, not preference optimization.

## Next Action

The row-level pressure-failure analysis and first curriculum artifact are now
tracked separately:

```text
pressure_failure_anchor = post_training/SFT_PRESSURE_FAILURE_ANALYSIS_2026-06-26.md
curriculum_anchor = post_training/SFT_CURRICULUM_ARTIFACTS_2026-06-26.md
```

Next run:

```bash
python3 post_training/run_sft_cv_sweep.py \
  --manifest post_training/negbiodb_ct_native_sft_cv4_curriculum_manifest.json \
  --out-dir post_training/runs/qwen_sft_cv4_curriculum_schema_action_80
```

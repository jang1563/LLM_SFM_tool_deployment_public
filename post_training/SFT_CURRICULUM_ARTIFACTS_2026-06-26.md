# SFT Curriculum Artifacts: 2026-06-26

This checkpoint implements the next formulation after the pressure-run
row-level failure analysis.

## Motivation

Pressure SFT gave a useful but mixed diagnostic shift:

```text
pressure_constrained_accuracy = 0.450
pressure_constrained_class_accuracy = defer 0/8, flag 6/8, ground 1/8, reject 3/8, verify 8/8
oracle_balanced_constrained_accuracy = 0.200
oracle_balanced_class_accuracy = defer 0/8, flag 0/8, ground 8/8, reject 0/8, verify 0/8
```

The main lesson is that global oversampling is too blunt. It can fix one
confusion axis while erasing another. The next formulation should preserve the
native held-out folds and add contrast-family pressure in the train folds.

## Commands

```bash
python3 post_training/analyze_sft_pressure_failures.py
python3 post_training/build_sft_curriculum_data.py
```

## Pressure Failure Analysis

Tracked files:

```text
post_training/SFT_PRESSURE_FAILURE_ANALYSIS_2026-06-26.md
post_training/sft_pressure_failure_analysis_2026-06-26.json
```

Main diagnosis:

```text
pressure fixed verify but over-rotated defer -> verify
pressure improved flag but made ground fragile
reject remains split across reject, flag, and verify
balanced-oracle collapses to ground/reject priors
```

## Native CV Curriculum Artifact

Source: `post_training/negbiodb_ct_native_sft_cv4_manifest.json`

Strategy: keep the original held-out folds unchanged, but replace each fold's
train split with an ordered contrast-family curriculum:

```text
dataset = negbiodb_ct_native_sft_curriculum_v1
strategy = contrast_family_interleave
source_train_examples_per_fold = 30
curriculum_train_examples_per_fold = 72
source_train_by_class = defer 6, flag 6, ground 6, reject 6, verify 6
curriculum_train_by_class = defer 12, flag 18, ground 18, reject 12, verify 12
curriculum_train_by_family = base 30, ground_flag 12, reject_override 18, verify_defer 12
heldout_examples_per_fold = 10
heldout_by_class_per_fold = defer 2, flag 2, ground 2, reject 2, verify 2
```

Family intent:

```text
base = preserve the original balanced native fold
ground_flag = separate cited positive-looking support from impossible-value evidence
verify_defer = separate other-indication failures from no-failure cases
reject_override = make mixed endpoint evidence override single-row support
```

Tracked files:

```text
post_training/negbiodb_ct_native_sft_cv4_curriculum_manifest.json
post_training/curriculum/negbiodb_ct_native_sft_cv4_curriculum_v1_fold0_train.jsonl
post_training/curriculum/negbiodb_ct_native_sft_cv4_curriculum_v1_fold1_train.jsonl
post_training/curriculum/negbiodb_ct_native_sft_cv4_curriculum_v1_fold2_train.jsonl
post_training/curriculum/negbiodb_ct_native_sft_cv4_curriculum_v1_fold3_train.jsonl
```

## Sanity Checks

```text
curriculum_fold_rows = 72 each
curriculum_fold_total_rows = 288
cv_curriculum_sweep_dry = passed for fold0, loss-only
curriculum_encode_dry = passed, 72 examples
```

## Full Run Follow-Up

```bash
python3 post_training/run_sft_cv_sweep.py \
  --manifest post_training/negbiodb_ct_native_sft_cv4_curriculum_manifest.json \
  --out-dir post_training/runs/qwen_sft_cv4_curriculum_schema_action_80
```

Tracked result:

```text
result_anchor = post_training/SFT_CURRICULUM_RUN_RESULTS_2026-06-26.md
result_json = post_training/sft_curriculum_run_summary_2026-06-26.json
strict_action_accuracy_mean = 0.475
strict_action_accuracy_range = 0.300..0.700
constrained_loaded_accuracy_mean = 0.425
constrained_loaded_accuracy_range = 0.300..0.600
strict_parse_failures_total = 0
strict_class_accuracy = defer 3/8, flag 6/8, ground 3/8, reject 4/8, verify 3/8
constrained_loaded_class_accuracy = defer 3/8, flag 7/8, ground 2/8, reject 2/8, verify 3/8
```

Failure-analysis follow-up:

```text
failure_anchor = post_training/SFT_CURRICULUM_FAILURE_ANALYSIS_2026-06-26.md
persistent_strict_and_constrained_failures = 20
main_failure_pairs = defer->verify, verify->defer, ground->flag, reject->flag
v2_result_anchor = post_training/SFT_CURRICULUM_V2_RUN_RESULTS_2026-06-26.md
v2_result = negative oversampling diagnostic
```

## Interpretation Boundary

The full curriculum CV sweep is complete. The result is a mixed SFT diagnostic:
it restores the original native CV strict mean and improves `flag`, but it does
not clearly beat the native baseline or pressure run in aggregate. Treat this as
evidence that the bottleneck is still SFT formulation and row-level action
boundary design, not yet DPO/RLVR. The targeted curriculum-v2 follow-up showed
that simply duplicating persistent failure rows over-rotates; the next
formulation should change the supervision representation.

# SFT Pressure Artifacts: 2026-06-26

This checkpoint implements the next formulation after the row-level failure
analysis in `SFT_FAILURE_ANALYSIS_2026-06-26.md`.

## Motivation

Previous full SFT execution showed that schema/JSON stability is solved, but
action discrimination is still weak:

```text
dominant_failures = verify->defer, flag->ground
native_cv_strict_class_accuracy = defer 3/8, flag 3/8, ground 6/8, reject 6/8, verify 1/8
oracle400_constrained_class_accuracy = defer 8/8, flag 0/8, ground 8/8, reject 0/8, verify 0/8
```

The goal here is not to claim a better model yet. It is to create tracked SFT
artifacts that put more gradient pressure on the fragile action classes before
rerunning the same CV/oracle evaluation harness.

## Command

```bash
python3 post_training/build_sft_pressure_data.py
```

## Native CV Pressure Artifact

Source: `post_training/negbiodb_ct_native_sft_cv4_manifest.json`

Strategy: keep the original native CV held-out folds unchanged, but oversample
`flag` and `verify` in each fold's train split.

```text
dataset = negbiodb_ct_native_sft_pressure_v1
strategy = native_cv_verify_flag_pressure
multipliers = flag 3, verify 3
source_train_examples_per_fold = 30
pressure_train_examples_per_fold = 54
pressure_train_by_class_per_fold = defer 6, flag 18, ground 6, reject 6, verify 18
heldout_examples_per_fold = 10
heldout_by_class_per_fold = defer 2, flag 2, ground 2, reject 2, verify 2
```

Tracked files:

```text
post_training/negbiodb_ct_native_sft_cv4_pressure_manifest.json
post_training/pressure/negbiodb_ct_native_sft_cv4_pressure_v1_fold0_train.jsonl
post_training/pressure/negbiodb_ct_native_sft_cv4_pressure_v1_fold1_train.jsonl
post_training/pressure/negbiodb_ct_native_sft_cv4_pressure_v1_fold2_train.jsonl
post_training/pressure/negbiodb_ct_native_sft_cv4_pressure_v1_fold3_train.jsonl
```

Next full run:

```bash
python3 post_training/run_sft_cv_sweep.py \
  --manifest post_training/negbiodb_ct_native_sft_cv4_pressure_manifest.json \
  --out-dir post_training/runs/qwen_sft_cv4_pressure_schema_action_80
```

## Oracle Balanced Artifact

Source: `post_training/negbiodb_ct_oracle_sft_v1.jsonl`

Strategy: class-balance the oracle artifact up to the maximum source class count
so the larger teacher artifact no longer overweights `ground` and `defer`.

```text
dataset = negbiodb_ct_oracle_sft_balanced_v1
strategy = class_balance_to_max
source_examples = 400
source_by_class = defer 120, flag 40, ground 140, reject 40, verify 60
balanced_examples = 700
balanced_by_class = defer 140, flag 140, ground 140, reject 140, verify 140
seed = 20260626
```

Tracked files:

```text
post_training/negbiodb_ct_oracle_sft_balanced_v1.jsonl
post_training/negbiodb_ct_oracle_sft_balanced_manifest.json
```

Next full run:

```bash
python3 post_training/run_sft_oracle_warmstart.py \
  --train-sft post_training/negbiodb_ct_oracle_sft_balanced_v1.jsonl \
  --train-limit 700 \
  --out-dir post_training/runs/qwen_oracle_balanced_warmstart_cvheldout
```

## Sanity Checks

```text
native_pressure_fold_rows = 54 each
oracle_balanced_rows = 700
native_pressure_encode_dry = passed, 54 examples
oracle_balanced_encode_dry = passed, 20-example smoke
cv_pressure_sweep_dry = passed for fold0, loss-only
oracle_balanced_warmstart_dry = passed for fold0 heldout, loss-only
```

## Interpretation Boundary

These are training-formulation artifacts only. They have not yet been trained
through the full SFT evaluation harness. Compare them only after running the
full pressure CV sweep and balanced-oracle warm-start commands above.

## Full-Run Follow-Up

The full rerun has now been executed and summarized separately:

```text
result_anchor = post_training/SFT_PRESSURE_RUN_RESULTS_2026-06-26.md
result_json = post_training/sft_pressure_run_summary_2026-06-26.json
native_pressure_constrained_loaded_accuracy_mean = 0.450
oracle_balanced_constrained_loaded_accuracy_mean = 0.200
```

The result is diagnostic rather than solved: native pressure helps constrained
`verify`/`flag` but hurts strict `ground`/`defer`; balanced-oracle warm start
collapses toward `ground`.

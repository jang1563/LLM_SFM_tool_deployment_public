# SFT Sweep Results: 2026-06-26

This file records the first full local execution of the native-CV SFT sweep and
the oracle-400 warm-start comparison. Raw run artifacts are under
`post_training/runs/` and are intentionally ignored by git; this file is the
tracked compact result anchor.

## Commands

```bash
python3 post_training/run_sft_cv_sweep.py \
  --out-dir post_training/runs/qwen_sft_cv4_schema_action_80

python3 post_training/run_sft_oracle_warmstart.py \
  --out-dir post_training/runs/qwen_oracle400_warmstart_cvheldout
```

Shared model/eval settings:

```text
model = Qwen/Qwen2.5-0.5B-Instruct
device = auto -> mps during training
batch_size = 2
max_length = 512
train_last_layers = 2
lr = 5e-5
score_mode = mean
```

## Native CV Fold Sweep

Training condition: train one checkpoint per fold on 30 native runner examples,
then evaluate on that fold's 10 held-out native examples.

```text
max_steps = 80
heldout_examples_per_fold = 10
heldout_by_class_per_fold = defer 2, flag 2, ground 2, reject 2, verify 2
parse_failures_total = 0
```

| fold | heldout loss | strict acc | constrained base | constrained loaded | strict by class |
| ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 0.0857 | 0.700 | 0.300 | 0.500 | defer 1/2, flag 1/2, ground 2/2, reject 2/2, verify 1/2 |
| 1 | 0.0576 | 0.500 | 0.200 | 0.500 | defer 0/2, flag 1/2, ground 2/2, reject 2/2, verify 0/2 |
| 2 | 0.0705 | 0.300 | 0.300 | 0.200 | defer 0/2, flag 0/2, ground 2/2, reject 1/2, verify 0/2 |
| 3 | 0.1065 | 0.400 | 0.300 | 0.400 | defer 2/2, flag 1/2, ground 0/2, reject 1/2, verify 0/2 |

Aggregate:

```text
heldout_loss_mean = 0.0801
strict_action_accuracy_mean = 0.475
strict_action_accuracy_range = 0.300..0.700
constrained_base_accuracy_mean = 0.275
constrained_loaded_accuracy_mean = 0.400
```

Interpretation: the fold sweep confirms parse-stable strict generation across
all held-out folds, but the apparent 0.700 result from the original fold0 split
does not generalize across folds. The model consistently learns some grounding
and rejecting behavior, while `verify`, `flag`, and some `defer` cases remain
fragile.

## Oracle-400 Warm Start

Training condition: train one checkpoint on the full 400-row deterministic
oracle artifact, then evaluate that checkpoint on each native CV held-out fold.

```text
train_limit = 400
max_steps = 160
train_first_loss = 1.6071
train_last_loss = 0.1871
train_loss_delta = -1.4200
parse_failures_total = 0
```

| eval set | heldout loss | strict acc | constrained base | constrained loaded | strict by class |
| --- | ---: | ---: | ---: | ---: | --- |
| fold0 heldout | 0.1493 | 0.400 | 0.300 | 0.400 | defer 2/2, flag 0/2, ground 2/2, reject 0/2, verify 0/2 |
| fold1 heldout | 0.1403 | 0.500 | 0.200 | 0.400 | defer 2/2, flag 0/2, ground 2/2, reject 1/2, verify 0/2 |
| fold2 heldout | 0.1352 | 0.500 | 0.300 | 0.400 | defer 2/2, flag 0/2, ground 2/2, reject 1/2, verify 0/2 |
| fold3 heldout | 0.1649 | 0.400 | 0.300 | 0.400 | defer 2/2, flag 0/2, ground 2/2, reject 0/2, verify 0/2 |

Aggregate:

```text
heldout_loss_mean = 0.1474
strict_action_accuracy_mean = 0.450
strict_action_accuracy_range = 0.400..0.500
constrained_base_accuracy_mean = 0.275
constrained_loaded_accuracy_mean = 0.400
```

Interpretation: oracle-400 training lowers training loss and preserves JSON
parse stability, but it does not improve held-out native action accuracy over
native fold training. Its strict predictions collapse toward `defer` and
`ground`; `flag` and `verify` remain at 0/8 across the four held-out folds.

## Takeaway

This is a useful negative/diagnostic checkpoint:

```text
SFT fixes output schema stability.
SFT does not yet robustly learn all action classes.
The larger oracle artifact is not automatically a better native-heldout learner.
DPO/RLVR should still wait until the SFT target/formulation improves.
```

Follow-up row-level analysis:

```text
analysis_anchor = post_training/SFT_FAILURE_ANALYSIS_2026-06-26.md
analysis_json = post_training/sft_failure_analysis_2026-06-26.json
```

Main row-level finding: `verify` is consistently pulled toward `defer`, and
`flag` is often treated as `ground`. Oracle-400 warm start makes this sharper:
`defer` and `ground` become reliable, but `flag`, `verify`, and constrained
`reject` collapse.

Next technical step: try a loss-balanced or class-pressure SFT formulation
before spending effort on DPO/RLVR.

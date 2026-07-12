# SFT Boundary-Rationale Artifacts: 2026-06-26

This checkpoint follows the targeted curriculum-v2 negative result. It stops
escalating row duplication and instead adds prompt-side boundary rationales that
name why the near-neighbor actions are wrong.

Raw run artifacts are under `post_training/runs/` and ignored by git.

## Commands

```bash
python3 post_training/build_sft_boundary_rationale_data.py

python3 post_training/run_sft_cv_sweep.py \
  --dry-run \
  --manifest post_training/negbiodb_ct_native_sft_cv4_boundary_rationale_manifest.json \
  --only-fold 0 \
  --out-dir post_training/runs/qwen_sft_cv4_boundary_rationale_dry \
  --skip-strict-generation \
  --skip-constrained

python3 post_training/run_sft_smoke.py \
  --dry-run \
  --sft post_training/boundary_rationale/negbiodb_ct_native_sft_cv4_boundary_rationale_v1_fold0_train.jsonl \
  --limit 60 \
  --max-length 512 \
  --out-dir post_training/runs/qwen_sft_boundary_rationale_encode_dry

python3 post_training/run_sft_smoke.py \
  --sft post_training/boundary_rationale/negbiodb_ct_native_sft_cv4_boundary_rationale_v1_fold0_train.jsonl \
  --limit 60 \
  --max-steps 1 \
  --batch-size 2 \
  --max-length 512 \
  --train-last-layers 1 \
  --device auto \
  --out-dir post_training/runs/qwen_sft_boundary_rationale_fold0_smoke

python3 post_training/run_sft_cv_sweep.py \
  --manifest post_training/negbiodb_ct_native_sft_cv4_boundary_rationale_manifest.json \
  --only-fold 0 \
  --out-dir post_training/runs/qwen_sft_cv4_boundary_rationale_fold0_evalfast \
  --skip-train-loss \
  --skip-base-constrained
```

## Artifact Summary

```text
dataset = negbiodb_ct_native_sft_boundary_rationale_v1
strategy = paired_boundary_rationale_v1
source_manifest = post_training/negbiodb_ct_native_sft_v1.jsonl
source_cv_manifest = post_training/negbiodb_ct_native_sft_cv4_manifest.json
manifest = post_training/negbiodb_ct_native_sft_cv4_boundary_rationale_manifest.json
```

| fold | train rows | train by class | train by role |
| --- | ---: | --- | --- |
| 0 | 60 | defer 12, flag 12, ground 12, reject 12, verify 12 | base 30, rationale 30 |
| 1 | 60 | defer 12, flag 12, ground 12, reject 12, verify 12 | base 30, rationale 30 |
| 2 | 60 | defer 12, flag 12, ground 12, reject 12, verify 12 | base 30, rationale 30 |
| 3 | 60 | defer 12, flag 12, ground 12, reject 12, verify 12 | base 30, rationale 30 |

Representation boundary: the rationale is inserted as a prompt-side user message
immediately before the final `submit_decision` tool call. The learned target
remains only the final decision JSON because the current SFT trainer masks the
prompt and trains on the final `submit_decision` arguments.

Boundary negatives:

```text
defer -> verify
verify -> defer
ground -> flag, reject
flag -> ground, reject
reject -> ground, flag
```

## Smoke Results

Encoding dry-run:

```text
examples = 60
max_length = 512
encoded_lengths_min = 341
encoded_lengths_max = 512
encoded_lengths_count_512 = 2
```

One-step fold0 smoke:

```text
device = mps
examples = 60
train_last_layers = 1
trainable_params = 14913280
loss = 3.9745
```

Fold0 evalfast sanity:

```text
train_first_loss = 3.9821
train_last_loss = 0.0272
train_loss_delta = -3.9549
heldout_teacher_forced_loaded_loss = 0.1164
strict_action_accuracy = 0.500
strict_parse_failures = 0
constrained_loaded_action_accuracy = 0.500
```

| eval | defer | flag | ground | reject | verify |
| --- | --- | --- | --- | --- | --- |
| strict | 0/2 | 1/2 | 0/2 | 2/2 | 2/2 |
| constrained loaded | 0/2 | 1/2 | 1/2 | 1/2 | 2/2 |

## Interpretation

- This is a representation-change diagnostic, not a full CV result.
- Fold0 shows the artifact is trainable, parse-stable, and compatible with the
  existing strict/constrained evaluation path.
- The fold0 class pattern is still uneven: `verify` and `reject` are strong,
  while `defer` and `ground` remain weak.
- The next safe claim requires a full 4-fold boundary-rationale CV rerun before
  moving to DPO/RLVR.

## Follow-Up

The full 4-fold boundary-rationale CV rerun is now recorded in
`post_training/SFT_BOUNDARY_RATIONALE_RUN_RESULTS_2026-06-26.md`.

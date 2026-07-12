# Boundary Preference Hard Split DPO Smoke: 2026-06-27

This checkpoint follows the limit8 hard-mode overfit smoke. It creates a
deterministic train/held-out split of the 240 negative-margin hard boundary
preference pairs, then runs a heldout-aware reference-free DPO-style margin
smoke.

## Split Artifact

```bash
python3 post_training/split_boundary_preference_hard_modes.py
```

```text
source = post_training/negbiodb_ct_oracle_boundary_preferences_hard_v1.jsonl
train = post_training/negbiodb_ct_oracle_boundary_preferences_hard_train_v1.jsonl
heldout = post_training/negbiodb_ct_oracle_boundary_preferences_hard_heldout_v1.jsonl
manifest = post_training/negbiodb_ct_oracle_boundary_preferences_hard_split_manifest.json
seed = 20260627
heldout_per_mode = 8
train_pairs = 208
heldout_pairs = 32
overlap_source_ids = []
```

Split balance:

```text
train_by_failure_mode = boundary_defer_over_verify 112, boundary_flag_over_ground 32, boundary_reject_over_flag 32, boundary_reject_over_ground 32
heldout_by_failure_mode = boundary_defer_over_verify 8, boundary_flag_over_ground 8, boundary_reject_over_flag 8, boundary_reject_over_ground 8
```

## Heldout-Aware DPO Smoke

Dry-run:

```bash
python post_training/run_boundary_preference_dpo_smoke.py \
  --dry-run \
  --preferences post_training/negbiodb_ct_oracle_boundary_preferences_hard_train_v1.jsonl \
  --eval-preferences post_training/negbiodb_ct_oracle_boundary_preferences_hard_heldout_v1.jsonl \
  --limit 16 \
  --eval-limit 8 \
  --max-length 768 \
  --out-dir post_training/runs/qwen_boundary_preference_dpo_hard_split_dry
```

Full split smoke:

```bash
python post_training/run_boundary_preference_dpo_smoke.py \
  --preferences post_training/negbiodb_ct_oracle_boundary_preferences_hard_train_v1.jsonl \
  --eval-preferences post_training/negbiodb_ct_oracle_boundary_preferences_hard_heldout_v1.jsonl \
  --limit 0 \
  --eval-limit 0 \
  --train-eval-limit 32 \
  --max-steps 48 \
  --batch-size 2 \
  --max-length 768 \
  --train-last-layers 2 \
  --lr 5e-5 \
  --beta 0.1 \
  --logprob-mode mean \
  --device auto \
  --out-dir post_training/runs/qwen_boundary_preference_dpo_hard_split_steps48
```

Result:

```text
condition = boundary_preference_reference_free_dpo_smoke
selected_pairs = 208
train_eval_pairs = 32
selected_eval_pairs = 32
max_steps = 48
batch_size = 2
train_last_layers = 2
trainable_params = 29825664
loss_delta = -0.611328125
step_margin_delta = +34.96875
trainable_state = post_training/runs/qwen_boundary_preference_dpo_hard_split_steps48/trainable_state.pt
```

Margin summary:

```text
pre_train_win_rate = 0.000
pre_train_mean_margin = -0.9019
post_train_win_rate = 1.000
post_train_mean_margin = 38.4805

pre_eval_win_rate = 0.000
pre_eval_mean_margin = -1.3701
post_eval_win_rate = 1.000
post_eval_mean_margin = 39.5352
```

Held-out post margins by failure mode:

```text
boundary_defer_over_verify: n=8, win_rate=1.000, mean_margin=31.5625
boundary_flag_over_ground: n=8, win_rate=1.000, mean_margin=10.8125
boundary_reject_over_flag: n=8, win_rate=1.000, mean_margin=56.7500
boundary_reject_over_ground: n=8, win_rate=1.000, mean_margin=59.0156
```

## Interpretation

This is stronger than the previous limit8 overfit smoke because the held-out
hard pairs also flip from negative to positive margins across all four selected
failure-mode families.

It is still not a full DPO/RLVR result. The objective is reference-free, the
teacher data is deterministic oracle-derived, and the evaluation is margin-only
over terminal-action pairs rather than live tool-use trajectory accuracy.

The next safe step is to load the trainable state into final-decision/tool-use
evaluation, or run a fuller hard-subset epoch with stricter held-out diagnostics.

## Follow-Up Candidate Eval

The loaded state was evaluated against all legal final-decision candidates on
the 32 held-out hard pairs:

```text
anchor = post_training/BOUNDARY_PREFERENCE_CANDIDATE_EVAL_2026-06-28.md
summary_json = post_training/boundary_preference_candidate_eval_summary_2026-06-28.json
base_action_accuracy = 0.000
base_exact_candidate_accuracy = 0.000
dpo_loaded_action_accuracy = 0.250
dpo_loaded_exact_candidate_accuracy = 0.250
dpo_loaded_defer_over_verify = 8/8
dpo_loaded_flag_over_ground = 0/8, pred defer 8
dpo_loaded_reject_over_flag = 0/8, pred defer 8
dpo_loaded_reject_over_ground = 0/8, pred defer 8
```

Interpretation: the pairwise held-out margin success does not yet translate to
all-candidate final-decision success. The loaded state fixes `defer_over_verify`
but introduces a broad `defer` top-1 collapse on `flag` and `reject` hard modes.
The next objective should use all-legal negatives or multi-negative hard prompts.

## Validation

```bash
python3 -m py_compile post_training/split_boundary_preference_hard_modes.py post_training/run_boundary_preference_dpo_smoke.py post_training/validate_post_training_data.py
python -m pytest -q tests/test_post_training_boundary_preference_split.py tests/test_post_training_boundary_preference_dpo_smoke.py tests/test_post_training_data_validator.py
python3 post_training/validate_post_training_data.py
```

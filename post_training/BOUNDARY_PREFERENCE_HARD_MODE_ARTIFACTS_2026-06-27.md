# Boundary Preference Hard-Mode Artifacts: 2026-06-27

This checkpoint follows the base preference-margin diagnostic. It extracts the
negative-margin boundary modes into a smaller preference artifact and runs a
minimal reference-free DPO-style smoke loop over 8 hard pairs.

## Hard-Mode Artifact

```bash
python3 post_training/build_boundary_preference_hard_modes.py
```

```text
source = post_training/negbiodb_ct_oracle_boundary_preferences_v1.jsonl
out = post_training/negbiodb_ct_oracle_boundary_preferences_hard_v1.jsonl
manifest = post_training/negbiodb_ct_oracle_boundary_preferences_hard_manifest.json
summary_json = post_training/boundary_preference_hard_mode_summary_2026-06-27.json
dataset = negbiodb_ct_oracle_boundary_preferences_hard_v1
strategy = base_negative_margin_hard_modes_v1
source_preference_pairs = 620
hard_preference_pairs = 240
chosen_passed = 240
rejected_passed = 0
```

Selected hard modes:

```text
boundary_defer_over_verify = 120
boundary_flag_over_ground = 40
boundary_reject_over_ground = 40
boundary_reject_over_flag = 40
```

Action distribution:

```text
chosen_actions = defer 120, flag 40, reject 80
rejected_actions = flag 40, ground 80, verify 120
```

## DPO-Style Smoke

Dry-run:

```bash
python post_training/run_boundary_preference_dpo_smoke.py \
  --dry-run \
  --limit 8 \
  --max-length 768 \
  --out-dir post_training/runs/qwen_boundary_preference_dpo_hard_dry
```

Observed encode lengths:

```text
selected_pairs = 8
chosen_lengths = 407, 450, 450, 364, 345, 346, 447, 447
rejected_lengths = 407, 466, 466, 364, 345, 346, 463, 463
```

Overfit/plumbing smoke:

```bash
python post_training/run_boundary_preference_dpo_smoke.py \
  --limit 8 \
  --max-steps 16 \
  --batch-size 1 \
  --max-length 768 \
  --train-last-layers 2 \
  --lr 5e-5 \
  --beta 0.1 \
  --logprob-mode mean \
  --device auto \
  --out-dir post_training/runs/qwen_boundary_preference_dpo_hard_smoke_limit8_steps16
```

Result:

```text
condition = boundary_preference_reference_free_dpo_smoke
model = Qwen/Qwen2.5-0.5B-Instruct
selected_pairs = 8
train_last_layers = 2
trainable_params = 29825664
loss_delta = -0.2598
step_margin_delta = +6.0625
pre_train_win_rate = 0.000
pre_train_mean_margin = -1.4004
post_train_win_rate = 1.000
post_train_mean_margin = 6.9414
trainable_state = post_training/runs/qwen_boundary_preference_dpo_hard_smoke_limit8_steps16/trainable_state.pt
```

## Boundary

This is not a full DPO/RLVR result. The smoke is reference-free and overfits
only 8 selected hard pairs to verify that the pairwise objective, masking,
save-state path, and margin evaluation all move in the expected direction.

The next safe experiment is a full hard-subset run with pre/post margin
evaluation by failure mode, ideally with a held-out split of hard pairs before
making any improvement claim.

## Follow-Up Split Checkpoint

The held-out split and larger hard-pair margin smoke are tracked separately:

```text
anchor = post_training/BOUNDARY_PREFERENCE_HARD_SPLIT_DPO_2026-06-27.md
summary_json = post_training/boundary_preference_hard_split_dpo_summary_2026-06-27.json
train_pairs = 208
heldout_pairs = 32
pre_eval_win_rate = 0.000
pre_eval_mean_margin = -1.3701
post_eval_win_rate = 1.000
post_eval_mean_margin = 39.5352
```

This follow-up is heldout-aware but still margin-only over deterministic
oracle-derived terminal-action pairs.

## Validation

```bash
python3 -m pytest -q tests/test_post_training_boundary_preference_hard_modes.py tests/test_post_training_boundary_preference_dpo_smoke.py tests/test_post_training_data_validator.py
python3 post_training/validate_post_training_data.py
python3 -m pytest -q
```

Current result:

```text
targeted_tests = 12 passed
full_tests = 127 passed
validator_issues = []
hard_boundary_preference_pairs = 240
```

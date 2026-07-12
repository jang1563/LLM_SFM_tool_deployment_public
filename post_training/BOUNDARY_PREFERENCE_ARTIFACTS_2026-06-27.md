# Evidence Boundary Preference Artifacts: 2026-06-27

This checkpoint builds a preference-training target after the negative
generative-rationale SFT diagnostic. The objective is narrower than the earlier
native preference artifact: it keeps the visible tool evidence fixed and
contrasts only the final `submit_decision` action.

## Command

```bash
python3 post_training/build_boundary_preference_data.py
```

## Artifact

```text
source_sft = post_training/negbiodb_ct_oracle_sft_v1.jsonl
tasks = negbiodb_ct/tasks_pilot.jsonl
out = post_training/negbiodb_ct_oracle_boundary_preferences_v1.jsonl
manifest = post_training/negbiodb_ct_oracle_boundary_preferences_manifest.json
dataset = negbiodb_ct_oracle_boundary_preferences_v1
strategy = evidence_boundary_contrast_v1
source_examples = 400
preference_pairs = 620
chosen_passed = 620
rejected_passed = 0
```

## Pair Distribution

```text
boundary_defer_over_verify = 120
boundary_verify_over_defer = 60
boundary_ground_over_flag = 140
boundary_ground_over_reject = 140
boundary_flag_over_ground = 40
boundary_flag_over_reject = 40
boundary_reject_over_ground = 40
boundary_reject_over_flag = 40
```

Chosen action counts:

```text
defer = 120
flag = 80
ground = 280
reject = 80
verify = 60
```

Rejected action counts:

```text
defer = 60
flag = 180
ground = 80
reject = 180
verify = 120
```

## Boundary

This is deterministic-oracle source data, not live runner behavior. Each pair
uses the same visible native CT tool observations in `prompt_messages`. The
chosen and rejected branches differ only in the terminal `submit_decision`
message, so the pair isolates boundary action selection rather than tool-use
formatting.

## Validation

```bash
python3 -m pytest -q tests/test_post_training_boundary_preference_data.py tests/test_post_training_data_validator.py
python3 post_training/validate_post_training_data.py
```

Current result:

```text
targeted_tests = 8 passed
validator_issues = []
boundary_preference_pairs = 620
```

## Base Margin Diagnostic

```text
anchor = post_training/BOUNDARY_PREFERENCE_MARGIN_BASE_2026-06-27.md
summary_json = post_training/boundary_preference_margin_base_summary_2026-06-27.json
raw_run = post_training/runs/qwen_boundary_preference_margin/base_full.json
model = Qwen/Qwen2.5-0.5B-Instruct
n = 620
mean_win_rate = 0.615
sum_win_rate = 0.453
hard_negative_modes = boundary_defer_over_verify 0.008/-0.1125, boundary_flag_over_ground 0.000/-0.1656, boundary_reject_over_ground 0.000/-2.6783, boundary_reject_over_flag 0.000/-2.5255
easy_or_aligned_modes = boundary_verify_over_defer 1.000/0.1042, boundary_ground_over_flag 1.000/0.1784, boundary_ground_over_reject 1.000/2.6621, boundary_flag_over_reject 1.000/2.4932
```

The full artifact is not uniformly hard. The most useful initial preference
training signal is concentrated in the negative-margin modes, especially
`defer > verify` and mixed-endpoint `reject > ground/flag`.

## Hard-Mode Follow-Up

```text
anchor = post_training/BOUNDARY_PREFERENCE_HARD_MODE_ARTIFACTS_2026-06-27.md
summary_json = post_training/boundary_preference_hard_mode_summary_2026-06-27.json
hard_mode_pairs = 240
dpo_smoke_pre_win_rate = 0.000
dpo_smoke_post_win_rate = 1.000
```

This follow-up filters the negative-margin modes and verifies the pairwise
training loop on an 8-pair overfit smoke. Keep it framed as plumbing until the
full hard subset or a held-out hard-pair split improves.

## Interpretation

This is the next weakest objective after generative-rationale SFT failed to
internalize the evidence rule. It directly targets the recurring held-out
confusions: `defer` versus `verify`, `ground` versus `flag`, and
`reject` versus `ground` or `flag`.

The next technical step is not to claim DPO/RLVR improvement yet. First run a
preference-margin diagnostic over these pairs for the base model and the recent
SFT checkpoints, then decide whether to implement a DPO smoke loop or keep this
as a supervised reward-model/RLVR pair source.

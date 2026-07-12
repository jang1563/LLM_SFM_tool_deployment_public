# Boundary Preference Base-Margin Diagnostic: 2026-06-27

This checkpoint scores the evidence-boundary preference artifact under the
base open model before any DPO or reward-style update. Positive margin means
the chosen evidence-derived terminal action has lower mean NLL than the
rejected action.

## Command

```bash
python post_training/run_boundary_preference_margin.py \
  --max-length 768 \
  --device auto \
  --out post_training/runs/qwen_boundary_preference_margin/base_full.json
```

## Artifact

```text
raw_run = post_training/runs/qwen_boundary_preference_margin/base_full.json
summary_json = post_training/boundary_preference_margin_base_summary_2026-06-27.json
preferences = post_training/negbiodb_ct_oracle_boundary_preferences_v1.jsonl
model = Qwen/Qwen2.5-0.5B-Instruct
n = 620
device = mps
max_length = 768
mean_win_rate = 0.615
sum_win_rate = 0.453
mean_margin_mean = 0.4442
mean_margin_median = 0.1550
```

## Failure-Mode Margins

| failure mode | n | mean win rate | mean margin | interpretation |
| --- | ---: | ---: | ---: | --- |
| `boundary_defer_over_verify` | 120 | 0.008 | -0.1125 | Hard negative: base model usually prefers `verify` over correct `defer`. |
| `boundary_flag_over_ground` | 40 | 0.000 | -0.1656 | Hard negative: base model prefers clean `ground` over correct impossible-value `flag`. |
| `boundary_reject_over_ground` | 40 | 0.000 | -2.6783 | Hard negative: base model strongly prefers `ground` over mixed-endpoint `reject`. |
| `boundary_reject_over_flag` | 40 | 0.000 | -2.5255 | Hard negative: base model strongly prefers `flag` over mixed-endpoint `reject`. |
| `boundary_verify_over_defer` | 60 | 1.000 | 0.1042 | Already aligned: base model prefers `verify` over `defer` when other-indication failures exist. |
| `boundary_ground_over_flag` | 140 | 1.000 | 0.1784 | Already aligned: base model prefers clean `ground` over spurious `flag`. |
| `boundary_ground_over_reject` | 140 | 1.000 | 2.6621 | Already aligned: base model strongly prefers clean `ground` over spurious `reject`. |
| `boundary_flag_over_reject` | 40 | 1.000 | 2.4932 | Already aligned: base model strongly prefers impossible-value `flag` over spurious `reject`. |

## Interpretation

The full 620-pair artifact is not uniformly hard. Half of the contrast families
are already aligned in the base model, especially `ground > reject` and
`flag > reject`. The useful training signal is concentrated in the negative
margin modes: `defer > verify`, `flag > ground`, and mixed-endpoint
`reject > ground/flag`.

The next DPO/preference smoke should therefore start from the hard modes rather
than all 620 pairs uniformly. Otherwise easy positive pairs can dominate the
loss while the actual boundary failures remain under-corrected.

## Follow-Up

```text
hard_mode_anchor = post_training/BOUNDARY_PREFERENCE_HARD_MODE_ARTIFACTS_2026-06-27.md
hard_mode_json = post_training/boundary_preference_hard_mode_summary_2026-06-27.json
hard_mode_pairs = 240
dpo_smoke_pre_win_rate = 0.000
dpo_smoke_pre_mean_margin = -1.4004
dpo_smoke_post_win_rate = 1.000
dpo_smoke_post_mean_margin = 6.9414
```

The follow-up smoke confirms that a reference-free pairwise objective can move
the selected hard-pair margins on a tiny overfit run. It is not yet a held-out
DPO/RLVR improvement claim.

## Validation

```bash
python3 -m pytest -q tests/test_post_training_boundary_preference_margin.py
python3 -m py_compile post_training/run_boundary_preference_margin.py
```

Observed before this summary was written:

```text
targeted_tests = 4 passed
py_compile = passed
```

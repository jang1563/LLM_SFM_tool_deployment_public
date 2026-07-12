# Boundary Preference Candidate Eval: 2026-06-28

This checkpoint follows the heldout-aware hard-split DPO-style margin smoke. It
tests whether the loaded DPO state makes the evidence-derived final action the
top candidate among all legal final-decision candidates, not just whether it
beats the paired rejected action.

## Command

Base:

```bash
python post_training/run_boundary_preference_candidate_eval.py \
  --preferences post_training/negbiodb_ct_oracle_boundary_preferences_hard_heldout_v1.jsonl \
  --limit 0 \
  --max-length 768 \
  --device auto \
  --score-mode mean \
  --out post_training/runs/qwen_boundary_preference_candidate_eval_hard_heldout/base_mean.json
```

DPO-loaded:

```bash
python post_training/run_boundary_preference_candidate_eval.py \
  --preferences post_training/negbiodb_ct_oracle_boundary_preferences_hard_heldout_v1.jsonl \
  --state post_training/runs/qwen_boundary_preference_dpo_hard_split_steps48/trainable_state.pt \
  --limit 0 \
  --max-length 768 \
  --device auto \
  --score-mode mean \
  --out post_training/runs/qwen_boundary_preference_candidate_eval_hard_heldout/dpo_loaded_mean.json
```

## Result

```text
n = 32
score_mode = mean
missing_expected_candidates = 0

base_action_accuracy = 0.000
base_exact_candidate_accuracy = 0.000
base_expected_rank_counts = 2:7, 3:19, 4:1, 5:1, 9:2, 18:1, 41:1
base_expected_margin_from_winner_mean = 1.4521

dpo_loaded_action_accuracy = 0.250
dpo_loaded_exact_candidate_accuracy = 0.250
dpo_loaded_expected_rank_counts = 1:8, 2:16, 4:7, 11:1
dpo_loaded_expected_margin_from_winner_mean = 14.5591
```

By failure mode:

```text
boundary_defer_over_verify:
  base exact = 0/8, pred = reject 6, verify 2
  dpo exact = 8/8, pred = defer 8

boundary_flag_over_ground:
  base exact = 0/8, pred = ground 8
  dpo exact = 0/8, pred = defer 8

boundary_reject_over_flag:
  base exact = 0/8, pred = ground 8
  dpo exact = 0/8, pred = defer 8, expected_rank = 2 for 8/8

boundary_reject_over_ground:
  base exact = 0/8, pred = ground 7, flag 1
  dpo exact = 0/8, pred = defer 8, expected_rank = 2 for 8/8
```

## Interpretation

This is an important negative follow-up. The DPO-loaded state improves the
all-candidate top-1 result from 0.000 to 0.250, but the gain is entirely from
`defer_over_verify`. For `reject` modes, the expected action moves to rank 2,
but `defer` wins top-1. For `flag_over_ground`, the loaded state is worse under
the all-candidate objective because `defer` dominates.

So the earlier held-out pairwise margin success does not yet translate to
decision-level candidate success. The pairwise objective learned a useful
contrast, but it also introduced a global `defer` prior when the model must
choose among all legal final actions.

## Next Safe Step

The next objective should expose all legal negatives, or at least multiple hard
negatives per prompt, instead of training only one chosen-vs-rejected pair at a
time. That keeps the post-training path conservative: pairwise DPO margin is a
useful signal, but decision-level improvement needs multi-candidate pressure.

## Follow-Up CE Smoke

Follow-up artifact:

```text
candidate_ce_anchor = post_training/BOUNDARY_PREFERENCE_CANDIDATE_CE_SMOKE_2026-06-28.md
candidate_ce_json = post_training/boundary_preference_candidate_ce_smoke_summary_2026-06-28.json
candidate_ce_script = post_training/run_boundary_preference_candidate_ce_smoke.py
```

The direct all-candidate CE smoke confirms that the next objective shape is
useful, but not sufficient on its own. A source-order balanced train/eval slice
moved held-out action accuracy from 0.000 to 0.333 by solving
`flag_over_ground` 4/4. A boundary round-robin schedule over the same selected
sets improved held-out action accuracy to 0.583 by recovering `defer` 3/4 and
`reject` 4/4, but it lost `flag` 0/4 to `reject`. A repeated-`flag` rehearsal
recovered `flag` but fell back to 0.333/0.417 held-out action accuracy and lost
`reject` 0/4. The next safe step is eval-gated checkpointing or a Pareto
boundary selector before claiming broader decision-level improvement.

## Validation

```bash
python -m py_compile post_training/run_boundary_preference_candidate_eval.py
python -m pytest -q tests/test_post_training_boundary_preference_candidate_eval.py tests/test_post_training_sft_constrained_eval.py tests/test_post_training_boundary_preference_margin.py
```

Current result:

```text
targeted_tests = 13 passed
```

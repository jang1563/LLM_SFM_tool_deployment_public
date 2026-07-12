# Boundary Preference Candidate CE Smoke: 2026-06-28

This checkpoint follows the held-out all-candidate eval for the hard-split
DPO-style state. The previous pairwise objective improved held-out margins but
did not reliably make the evidence-derived action the top candidate among all
legal final decisions. This smoke test trains directly on all legal candidates
with cross entropy over candidate log-likelihoods.

## Script

```bash
python post_training/run_boundary_preference_candidate_ce_smoke.py \
  --preferences post_training/negbiodb_ct_oracle_boundary_preferences_hard_train_v1.jsonl \
  --eval-preferences post_training/negbiodb_ct_oracle_boundary_preferences_hard_heldout_v1.jsonl \
  --limit 0 \
  --limit-per-action 8 \
  --eval-limit 0 \
  --eval-limit-per-action 4 \
  --max-steps 24 \
  --batch-size 1 \
  --candidate-batch-size 1 \
  --max-length 768 \
  --train-last-layers 2 \
  --lr 5e-5 \
  --temperature 1.0 \
  --logprob-mode mean \
  --device auto \
  --out-dir post_training/runs/qwen_boundary_preference_candidate_ce_balanced_a8_eval4_steps24
```

The `candidate_batch_size=1` setting is intentional on local MPS. Scoring every
candidate in a set at once can exceed MPS graph tensor limits; chunked scoring
preserves the all-candidate objective while keeping the smoke runnable.

## Dataset Check

Balanced dry-run:

```text
source_train_pairs = 208
selected_train_sets = 24
train_by_expected_action = defer 8, flag 8, reject 8

source_eval_pairs = 32
selected_eval_sets = 12
eval_by_expected_action = defer 4, flag 4, reject 4

train_missing_expected_candidates = 0
eval_missing_expected_candidates = 0
```

## Results

Unbalanced first smoke, limit16/eval8/steps12:

```text
train_expected_actions = defer 11, flag 1, reject 4
pre_train_action_accuracy = 0.000
post_train_action_accuracy = 0.688
pre_eval_action_accuracy = 0.000
post_eval_action_accuracy = 0.000

post_train_defer = 11/11
post_train_flag = 0/1, pred defer 1
post_train_reject = 0/4, pred defer 4
post_eval_flag = 0/2, pred defer 2
post_eval_reject = 0/6, pred defer 6
```

Interpretation: the objective and save/eval path work, but the first selected
training slice is dominated by `defer`. It overfits `defer` and collapses
held-out `flag`/`reject` prompts to `defer`.

Balanced smoke, per-action train 8 and eval 4:

```text
pre_train_action_accuracy = 0.000
post_train_action_accuracy = 0.333
pre_eval_action_accuracy = 0.000
post_eval_action_accuracy = 0.333

post_train_defer = 0/8, pred reject 8
post_train_flag = 8/8, pred flag 8
post_train_reject = 0/8, pred flag 8

post_eval_defer = 0/4, pred reject 4
post_eval_flag = 4/4, pred flag 4
post_eval_reject = 0/4, pred flag 4
```

Boundary round-robin smoke, same selected train/eval sets:

```bash
python post_training/run_boundary_preference_candidate_ce_smoke.py \
  --preferences post_training/negbiodb_ct_oracle_boundary_preferences_hard_train_v1.jsonl \
  --eval-preferences post_training/negbiodb_ct_oracle_boundary_preferences_hard_heldout_v1.jsonl \
  --training-schedule boundary_round_robin \
  --phase-order defer,flag,reject \
  --limit 0 \
  --limit-per-action 8 \
  --eval-limit 0 \
  --eval-limit-per-action 4 \
  --max-steps 24 \
  --batch-size 1 \
  --candidate-batch-size 1 \
  --max-length 768 \
  --train-last-layers 2 \
  --lr 5e-5 \
  --temperature 1.0 \
  --logprob-mode mean \
  --device auto \
  --out-dir post_training/runs/qwen_boundary_preference_candidate_ce_round_robin_a8_eval4_steps24
```

```text
training_schedule = boundary_round_robin
phase_order = defer, flag, reject
pre_train_action_accuracy = 0.000
post_train_action_accuracy = 0.708
pre_eval_action_accuracy = 0.000
post_eval_action_accuracy = 0.583

post_train_defer = 8/8, pred defer 8
post_train_flag = 1/8, pred flag 1, reject 7
post_train_reject = 8/8, pred reject 8

post_eval_defer = 3/4, pred defer 3, reject 1
post_eval_flag = 0/4, pred reject 4
post_eval_reject = 4/4, pred reject 4
```

Flag-rehearsal follow-up, same selected train/eval sets:

```bash
python post_training/run_boundary_preference_candidate_ce_smoke.py \
  --preferences post_training/negbiodb_ct_oracle_boundary_preferences_hard_train_v1.jsonl \
  --eval-preferences post_training/negbiodb_ct_oracle_boundary_preferences_hard_heldout_v1.jsonl \
  --training-schedule boundary_round_robin \
  --phase-order defer,flag,reject,flag \
  --limit 0 \
  --limit-per-action 8 \
  --eval-limit 0 \
  --eval-limit-per-action 4 \
  --max-steps 24 \
  --batch-size 1 \
  --candidate-batch-size 1 \
  --max-length 768 \
  --train-last-layers 2 \
  --lr 5e-5 \
  --temperature 1.0 \
  --logprob-mode mean \
  --device auto \
  --out-dir post_training/runs/qwen_boundary_preference_candidate_ce_flag_rehearsal_a8_eval4_steps24
```

```text
training_schedule = boundary_round_robin
phase_order = defer, flag, reject, flag
pre_train_action_accuracy = 0.000
post_train_action_accuracy = 0.458
pre_eval_action_accuracy = 0.000
post_eval_action_accuracy = 0.333

post_eval_defer = 0/4, pred reject 4
post_eval_flag = 4/4, pred flag 4
post_eval_reject = 0/4, pred flag 4
```

Stop-after-reject variant, same phase order but `max_steps=23`:

```text
pre_train_action_accuracy = 0.000
post_train_action_accuracy = 0.500
pre_eval_action_accuracy = 0.000
post_eval_action_accuracy = 0.417

post_eval_defer = 1/4, pred defer 1, reject 3
post_eval_flag = 4/4, pred flag 4
post_eval_reject = 0/4, pred flag 4
```

## Interpretation

This is useful but still not solved. All-candidate CE can move top-1 decision
behavior on held-out prompts. Source-order balanced training recovers
`flag_over_ground` 4/4 but loses `defer` and `reject`. Boundary round-robin
training recovers `defer` 3/4 and `reject` 4/4, but loses `flag` 0/4 to
`reject`. Repeating the `flag` phase recovers held-out `flag` 4/4, but reverts
to the source-order failure mode and loses `reject` 0/4. Stopping one step
earlier recovers only one held-out `defer` item and still loses `reject`.

So the next issue is no longer just pairwise-vs-all-candidate evaluation. The
current candidate objective needs checkpoint selection or an explicit
multi-objective boundary criterion so it can preserve `flag` while learning
`defer` and `reject`, instead of trading one boundary success for another.

## Next Safe Step

Do not claim broad decision-level improvement yet. The next objective should
keep the all-candidate decision loss, but add eval-gated checkpoint selection
or a Pareto/multi-objective selector across boundary phases:

```text
defer vs verify empty-evidence boundary
flag vs ground invalid/contradicted-evidence boundary
reject vs flag/ground mixed-endpoint boundary
select checkpoints that do not collapse any held-out action family
```

Only scale to the full held-out hard split after the selected small-slice
checkpoint avoids zero-accuracy action families.

## Eval-Gated Selector Plumbing

The CE smoke runner now supports eval-gated checkpoint selection:

```text
--eval-checkpoint-every N
--checkpoint-selection min_action_accuracy
```

With checkpointing enabled, the runner evaluates held-out all-candidate
accuracy during training, stores each checkpoint summary in `eval_checkpoints`,
restores the selected trainable state before final `post_train`/`post_eval`,
and records `post_evaluation_state = selected_checkpoint`.

Tiny plumbing smoke:

```text
run = post_training/runs/qwen_boundary_preference_candidate_ce_eval_gated_plumbing_a2_eval2_steps6/report.json
train/eval sets = 2 per action
max_steps = 6
eval_checkpoint_every = 1
checkpoint_selection = min_action_accuracy
selected_checkpoint = step 6, phase reject
post_eval_action_accuracy = 0.667
post_eval_defer = 2/2
post_eval_flag = 2/2
post_eval_reject = 0/2, pred flag 2
```

Interpretation: the selector/report/restore path works, but this tiny run still
shows the unresolved `reject` collapse. The next claim-bearing run should use
the same selector on the full balanced small slice or full hard split.

Balanced selector run, same a8/eval4 slice:

```text
run = post_training/runs/qwen_boundary_preference_candidate_ce_eval_gated_a8_eval4_steps24_every3/report.json
train/eval sets = 8/4 per action
max_steps = 24
eval_checkpoint_every = 3
checkpoint_selection = min_action_accuracy
selected_checkpoint = step 15, phase reject
post_eval_action_accuracy = 0.667
post_eval_defer = 4/4
post_eval_flag = 0/4, pred defer 1, reject 3
post_eval_reject = 4/4
```

Checkpoint path:

```text
step 3:  action_accuracy 0.667, defer 4/4, flag 4/4, reject 0/4
step 15: action_accuracy 0.667, defer 4/4, flag 0/4, reject 4/4
step 24: action_accuracy 0.583, defer 3/4, flag 0/4, reject 4/4
```

Interpretation: eval-gated selection improves the small-slice top-1 result
from the previous round-robin final 0.583 to 0.667, but this three-step eval
cadence still cannot find a checkpoint where all three action families are
nonzero. The zero-family problem has moved from simple schedule choice to
explicit objective/selection design.

## Boundary Phase-Batch Follow-up

The CE smoke runner now also supports:

```text
--training-schedule boundary_phase_batch
```

Unlike round-robin, each optimizer step contains one `defer`, one `flag`, and
one `reject` example from the configured `phase_order`. This tests whether
simultaneous family exposure is more stable than sequential family updates.

Tiny phase-batch plumbing smoke:

```text
run = post_training/runs/qwen_boundary_preference_candidate_ce_phase_batch_plumbing_a2_eval2_steps3/report.json
train/eval sets = 2 per action
max_steps = 3
eval_checkpoint_every = 1
selected_checkpoint = step 3, phase phase_batch:defer+flag+reject
post_eval_action_accuracy = 0.333
post_eval_defer = 2/2
post_eval_flag = 0/2, pred defer 2
post_eval_reject = 0/2, pred defer 2
each_loss_step_phase_counts = defer 1, flag 1, reject 1
```

Balanced phase-batch selector run, same a8/eval4 slice:

```text
run = post_training/runs/qwen_boundary_preference_candidate_ce_phase_batch_a8_eval4_steps8_every1/report.json
train/eval sets = 8/4 per action
max_steps = 8
eval_checkpoint_every = 1
checkpoint_selection = min_action_accuracy
selected_checkpoint = step 7, phase phase_batch:defer+flag+reject
selection_score = min-action 0.750, action 0.917, exact 0.917, -margin -0.0007
post_eval_action_accuracy = 0.917
post_eval_defer = 4/4
post_eval_flag = 4/4
post_eval_reject = 3/4, pred flag 1, reject 3
post_train_action_accuracy = 0.833
```

Checkpoint path:

```text
step 1: action_accuracy 0.333, defer 4/4, flag 0/4, reject 0/4
step 4: action_accuracy 0.667, defer 4/4, flag 0/4, reject 4/4
step 7: action_accuracy 0.917, defer 4/4, flag 4/4, reject 3/4
step 8: action_accuracy 0.667, defer 4/4, flag 4/4, reject 0/4
```

Interpretation: this is the first tested small-slice checkpoint where no action
family is at zero accuracy. It is a meaningful positive diagnostic for the
phase-mixed CE schedule plus eval-gated checkpoint selection. It is not yet a
broad post-training claim: the result is small, checkpoint-sensitive, and the
next step should replicate it across folds/seeds or add a robustness-aware
selector before scaling to the full hard split.

## Seeded Phase-Batch Replicate

The CE runner now supports seeded candidate-set selection:

```text
--selection-seed 11
```

When set, train/eval candidate sets are shuffled before per-action and global
limits are applied, and the selected IDs are recorded in the report. This keeps
the default first-N behavior unchanged while making small-slice replication
explicit.

Seed 11 phase-batch selector run:

```text
run = post_training/runs/qwen_boundary_preference_candidate_ce_phase_batch_seed11_a8_eval4_steps8_every1/report.json
selection_seed = 11
train/eval sets = 8/4 per action
max_steps = 8
eval_checkpoint_every = 1
checkpoint_selection = min_action_accuracy
selected_checkpoint = step 8, phase phase_batch:defer+flag+reject
selection_score = min-action 0.250, action 0.750, exact 0.750, -margin -0.0059
post_eval_action_accuracy = 0.750
post_eval_defer = 4/4
post_eval_flag = 1/4, pred defer 1, flag 1, reject 2
post_eval_reject = 4/4
post_train_action_accuracy = 0.750
```

Checkpoint path:

```text
step 1: action_accuracy 0.333, defer 4/4, flag 0/4, reject 0/4
step 6: action_accuracy 0.500, defer 4/4, flag 0/4, reject 2/4
step 7: action_accuracy 0.667, defer 4/4, flag 0/4, reject 4/4
step 8: action_accuracy 0.750, defer 4/4, flag 1/4, reject 4/4
```

Interpretation: the seeded replicate partially supports phase-batch scheduling:
it avoids zero-family collapse at the selected checkpoint. It does not replicate
the earlier 0.917 result. The remaining failure is `flag_over_ground`, so the
next robustness work should focus on preserving `flag` while keeping `defer`
and `reject` intact.

## Flag-Weighted Phase-Batch Diagnostic

The CE runner now supports per-action loss weights:

```text
--action-loss-weights flag=2.0
```

This is a diagnostic knob, not a solved objective. On the same seed 11
candidate selection, a simple `flag` upweight fixes `flag` but collapses
`reject`.

Flag-weighted seed 11 phase-batch selector run:

```text
run = post_training/runs/qwen_boundary_preference_candidate_ce_phase_batch_flagw2_seed11_a8_eval4_steps8_every1/report.json
selection_seed = 11
action_loss_weights = flag 2.0
train/eval sets = 8/4 per action
max_steps = 8
eval_checkpoint_every = 1
checkpoint_selection = min_action_accuracy
selected_checkpoint = step 2, phase phase_batch:defer+flag+reject
selection_score = min-action 0.000, action 0.667, exact 0.667, -margin -0.0690
post_eval_action_accuracy = 0.667
post_eval_defer = 4/4
post_eval_flag = 4/4
post_eval_reject = 0/4, pred flag 4
post_train_action_accuracy = 0.667
```

Checkpoint path:

```text
step 1: action_accuracy 0.667, defer 4/4, flag 4/4, reject 0/4
step 2: action_accuracy 0.667, defer 4/4, flag 4/4, reject 0/4
step 8: action_accuracy 0.667, defer 4/4, flag 4/4, reject 0/4
```

A lower-weight probe shows the same failure mode:

```text
run = post_training/runs/qwen_boundary_preference_candidate_ce_phase_batch_flagw125_seed11_a8_eval4_steps8_every1/report.json
selection_seed = 11
action_loss_weights = flag 1.25
selected_checkpoint = step 7, phase phase_batch:defer+flag+reject
selection_score = min-action 0.000, action 0.667, exact 0.667, -margin -0.0723
post_eval_action_accuracy = 0.667
post_eval_defer = 4/4
post_eval_flag = 4/4
post_eval_reject = 0/4, pred flag 4
post_train_action_accuracy = 0.667
```

`flag=1.25` checkpoint path:

```text
step 1: action_accuracy 0.333, defer 4/4, flag 0/4, reject 0/4
step 6: action_accuracy 0.667, defer 4/4, flag 4/4, reject 0/4
step 7: action_accuracy 0.667, defer 4/4, flag 4/4, reject 0/4
step 8: action_accuracy 0.667, defer 4/4, flag 4/4, reject 0/4
```

Interpretation: flag-only weighting is a negative robustness result at both
`flag=1.25` and `flag=2.0`. It trades the seed 11 run's weak `flag` family for a
full `reject -> flag` collapse, so the next step should be a joint
`flag`/`reject` objective or selector rather than assuming naive class weighting
is enough.

## Lightweight Joint Flag/Reject Probe

The CE runner now has sweep controls:

```text
--skip-train-eval
--skip-pre-eval
```

When checkpoint evaluation is enabled, it also writes an incremental
`checkpoint_report.json` after each checkpoint so long probes can be inspected
or interrupted without losing the checkpoint path.

Joint low-weight seed 11 phase-batch probe:

```text
run = post_training/runs/qwen_boundary_preference_candidate_ce_phase_batch_flagw125_rejectw125_seed11_light_steps6_every2/report.json
checkpoint_report = post_training/runs/qwen_boundary_preference_candidate_ce_phase_batch_flagw125_rejectw125_seed11_light_steps6_every2/checkpoint_report.json
selection_seed = 11
action_loss_weights = flag 1.25, reject 1.25
skip_train_eval = true
skip_pre_eval = true
max_steps = 6
eval_checkpoint_every = 2
checkpoint_selection = min_action_accuracy
selected_checkpoint = step 6, phase phase_batch:defer+flag+reject
selection_score = min-action 0.000, action 0.500, exact 0.500, -margin -0.1074
post_eval_action_accuracy = 0.500
post_eval_defer = 2/4, pred defer 2, reject 2
post_eval_flag = 0/4, pred reject 4
post_eval_reject = 4/4
post_train_action_accuracy = skipped
```

Checkpoint path:

```text
step 2: action_accuracy 0.333, defer 0/4, flag 0/4, reject 4/4
step 4: action_accuracy 0.333, defer 0/4, flag 0/4, reject 4/4
step 6: action_accuracy 0.500, defer 2/4, flag 0/4, reject 4/4
```

Interpretation: joint low `flag`/`reject` weighting preserves `reject`, but it
does so by shifting the collapse toward `reject`; `flag` remains 0/4. The next
objective should change the decision target or selector itself, not just add
more naive action weights.

## Action-Level CE Probe

The CE runner now supports:

```text
--loss-target action
```

This aggregates candidate log probabilities by action using the best candidate
score per action, then trains against the expected action rather than the exact
chosen candidate.

Action-target seed 11 phase-batch probe:

```text
run = post_training/runs/qwen_boundary_preference_action_ce_phase_batch_seed11_light_steps6_every2/report.json
checkpoint_report = post_training/runs/qwen_boundary_preference_action_ce_phase_batch_seed11_light_steps6_every2/checkpoint_report.json
selection_seed = 11
loss_target = action
action_loss_weights = none
skip_train_eval = true
skip_pre_eval = true
max_steps = 6
eval_checkpoint_every = 2
checkpoint_selection = min_action_accuracy
selected_checkpoint = step 6, phase phase_batch:defer+flag+reject
selection_score = min-action 0.000, action 0.667, exact 0.667, -margin -0.0182
post_eval_action_accuracy = 0.667
post_eval_defer = 4/4
post_eval_flag = 4/4
post_eval_reject = 0/4, pred flag 4
post_train_action_accuracy = skipped
```

Checkpoint path:

```text
step 2: action_accuracy 0.333, defer 4/4, flag 0/4, reject 0/4
step 4: action_accuracy 0.417, defer 4/4, flag 0/4, reject 1/4
step 6: action_accuracy 0.667, defer 4/4, flag 4/4, reject 0/4
```

Interpretation: action-level CE recovers `defer` and `flag`, but still
collapses `reject -> flag`. Changing the target from exact candidate to action
is therefore not sufficient; the next objective needs an explicit `flag`/
`reject` floor or margin-aware selector.

## Action-Floor Selector Probe

The checkpoint selector now supports:

```text
--checkpoint-selection action_floor
--checkpoint-action-floors flag=0.25,reject=0.25
```

This does not change the training loss. It changes checkpoint selection so a
checkpoint satisfying explicit action-family floors is preferred over a higher
average-accuracy checkpoint with a collapsed required family. If no checkpoint
satisfies the floors, the report records the best violation via
`selection_action_floor`.

Seed 11 phase-batch floor-gated probe:

```text
run = post_training/runs/qwen_boundary_preference_candidate_ce_action_floor_seed11_a8_eval4_steps8_every1/report.json
checkpoint_report = post_training/runs/qwen_boundary_preference_candidate_ce_action_floor_seed11_a8_eval4_steps8_every1/checkpoint_report.json
selection_seed = 11
loss_target = candidate
checkpoint_selection = action_floor
checkpoint_action_floors = flag 0.25, reject 0.25
skip_train_eval = true
skip_pre_eval = true
max_steps = 8
eval_checkpoint_every = 1
selected_checkpoint = step 8, phase phase_batch:defer+flag+reject
floor_satisfied = true
selection_score = floor 1.000, -deficit -0.000, required-min 0.250, min-action 0.250, action 0.750, exact 0.750, -margin -0.0059
post_eval_action_accuracy = 0.750
post_eval_defer = 4/4
post_eval_flag = 1/4, pred defer 1, flag 1, reject 2
post_eval_reject = 4/4
post_train_action_accuracy = skipped
```

Checkpoint path:

```text
step 1: floor false, action_accuracy 0.333, defer 4/4, flag 0/4, reject 0/4
step 2: floor false, action_accuracy 0.333, defer 4/4, flag 0/4, reject 0/4
step 3: floor false, action_accuracy 0.333, defer 4/4, flag 0/4, reject 0/4
step 4: floor false, action_accuracy 0.333, defer 4/4, flag 0/4, reject 0/4
step 5: floor false, action_accuracy 0.333, defer 4/4, flag 0/4, reject 0/4
step 6: floor false, action_accuracy 0.500, defer 4/4, flag 0/4, reject 2/4
step 7: floor false, action_accuracy 0.667, defer 4/4, flag 0/4, reject 4/4
step 8: floor true, action_accuracy 0.750, defer 4/4, flag 1/4, reject 4/4
```

Interpretation: the selector now makes the `flag`/`reject` floor explicit and
selects the first checkpoint satisfying both nonzero floors on this seed 11
slice. This is useful selector hardening, but it is not a robust behavioral
solve because `flag` remains weak at 1/4. The next objective should apply
asymmetric `flag` vs `reject` pressure rather than only changing checkpoint
selection.

## Asymmetric Flag/Reject Margin Probe

The CE runner now supports opt-in asymmetric action margin penalties:

```text
--action-margin-penalties flag>reject=0.25
--action-margin-weight 1.0
```

This adds a hinge penalty on examples whose expected action is the target
action. For `flag>reject=0.25`, the penalty pushes the best `flag` candidate
score to exceed the best `reject` candidate score by at least 0.25 on expected
`flag` examples. It does not directly alter `reject` examples.

Seed 11 phase-batch margin probe:

```text
run = post_training/runs/qwen_boundary_preference_candidate_ce_flag_reject_margin_seed11_a8_eval4_steps8_every2/report.json
checkpoint_report = post_training/runs/qwen_boundary_preference_candidate_ce_flag_reject_margin_seed11_a8_eval4_steps8_every2/checkpoint_report.json
selection_seed = 11
loss_target = candidate
action_margin_penalties = flag>reject 0.25
action_margin_weight = 1.0
checkpoint_selection = action_floor
checkpoint_action_floors = flag 0.25, reject 0.25
skip_train_eval = true
skip_pre_eval = true
max_steps = 8
eval_checkpoint_every = 2
selected_checkpoint = step 8, phase phase_batch:defer+flag+reject
floor_satisfied = false
selection_score = floor 0.000, -deficit -0.250, required-min 0.000, min-action 0.000, action 0.667, exact 0.667, -margin -0.2650
post_eval_action_accuracy = 0.667
post_eval_defer = 4/4
post_eval_flag = 4/4
post_eval_reject = 0/4, pred flag 4
post_train_action_accuracy = skipped
```

Checkpoint path:

```text
step 2: floor false, action_accuracy 0.333, defer 4/4, flag 0/4, reject 0/4
step 4: floor false, action_accuracy 0.333, defer 4/4, flag 0/4, reject 0/4
step 6: floor false, action_accuracy 0.667, defer 4/4, flag 4/4, reject 0/4
step 8: floor false, action_accuracy 0.667, defer 4/4, flag 4/4, reject 0/4
```

Interpretation: the one-sided margin penalty is active in the loss trace and
does recover held-out `flag`, but it over-rotates into the same `reject -> flag`
collapse as naive flag weighting and action-level CE. The `action_floor`
selector correctly refuses to mark the checkpoint floor-satisfied. The next
objective needs two-sided or floor-aware training pressure, not only
`flag>reject`.

## Two-Sided Flag/Reject Margin Probe

The same margin hook accepts multiple directional constraints:

```text
--action-margin-penalties flag>reject=0.25,reject>flag=0.25
--action-margin-weight 1.0
```

This applies `flag>reject` pressure on expected `flag` examples and
`reject>flag` pressure on expected `reject` examples. The checkpoint selector
remains the explicit `action_floor` gate.

Seed 11 phase-batch two-sided margin probe:

```text
run = post_training/runs/qwen_boundary_preference_candidate_ce_two_sided_margin_seed11_a8_eval4_steps8_every2/report.json
checkpoint_report = post_training/runs/qwen_boundary_preference_candidate_ce_two_sided_margin_seed11_a8_eval4_steps8_every2/checkpoint_report.json
selection_seed = 11
loss_target = candidate
action_margin_penalties = flag>reject 0.25, reject>flag 0.25
action_margin_weight = 1.0
checkpoint_selection = action_floor
checkpoint_action_floors = flag 0.25, reject 0.25
skip_train_eval = true
skip_pre_eval = true
max_steps = 8
eval_checkpoint_every = 2
selected_checkpoint = step 4, phase phase_batch:defer+flag+reject
floor_satisfied = false
selection_score = floor 0.000, -deficit -0.250, required-min 0.000, min-action 0.000, action 0.583, exact 0.583, -margin -0.2728
post_eval_action_accuracy = 0.583
post_eval_defer = 4/4
post_eval_flag = 0/4, pred defer 3, reject 1
post_eval_reject = 3/4, pred defer 1, reject 3
post_train_action_accuracy = skipped
```

Checkpoint path:

```text
step 2: floor false, action_accuracy 0.333, defer 0/4, flag 0/4, reject 4/4
step 4: floor false, action_accuracy 0.583, defer 4/4, flag 0/4, reject 3/4
step 6: floor false, action_accuracy 0.333, defer 4/4, flag 0/4, reject 0/4
step 8: floor false, action_accuracy 0.333, defer 4/4, flag 0/4, reject 0/4
```

Interpretation: the two-sided margin penalty is active in the loss trace, but
it still fails the explicit `flag`/`reject` floor gate. The selected checkpoint
partially preserves `reject`, but `flag` is 0/4 and the later checkpoints keep
`defer` while losing both `flag` and `reject`. This is another negative
objective-pressure result. The next formulation needs a genuinely floor-aware
training objective or a multi-negative action objective before any full
hard-split scaling claim.

## Validation

```bash
python -m py_compile post_training/run_boundary_preference_candidate_ce_smoke.py
python -m pytest -q tests/test_post_training_boundary_preference_candidate_ce_smoke.py tests/test_post_training_boundary_preference_candidate_eval.py tests/test_post_training_boundary_preference_dpo_smoke.py
```

Current result:

```text
candidate_ce_unit_tests = 22 passed
targeted_tests = 30 passed
```

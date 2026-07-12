# Evidence-Derived Boundary Rationale Layer: 2026-06-27

This file records the first reusable evidence-rationale layer extracted from the
held-out rationale ablation. Unlike the oracle ablation, this artifact derives
the rationale and final-action hint from visible native CT tool observations.

## Command

```bash
python3 post_training/apply_evidence_rationale.py
```

## Artifact

```text
source = post_training/negbiodb_ct_native_sft_v1.jsonl
out = post_training/negbiodb_ct_native_sft_evidence_rationale_v1.jsonl
manifest = post_training/negbiodb_ct_native_sft_evidence_rationale_manifest.json
dataset = negbiodb_ct_native_sft_evidence_rationale_v1
strategy = deployable_evidence_boundary_rationale_v1
examples = 40
by_action_class = defer 8, flag 8, ground 8, reject 8, verify 8
by_evidence_action = defer 8, flag 8, ground 8, reject 8, verify 8
by_role = evidence_rationale 40
evidence_action_matches = 40
evidence_action_mismatches = 0
evidence_action_unlabeled = 0
```

## Boundary

The layer is deterministic and CT-native. It reads the existing tool observations:

- `search_failures`
- `check_other_indications`

It then emits a prompt-side `BOUNDARY_RATIONALE` message with an
`Evidence-derived final action` hint. The rule does not read gold labels. When
`action_class` exists, the manifest reports whether the derived action matches it.

## Interpretation

This turns the evidence-rationale ablation into a reusable preprocessor or
distillation artifact. The next modeling choice is whether to use this as an
inference-time guardrail/routing layer, or to train a model to internalize the
same boundary rationale without the rule message.

## Pilot-400 Stress Test

Follow-up full-pilot oracle SFT stress test:

```text
anchor = post_training/EVIDENCE_RATIONALE_PILOT400_STRESS_2026-06-27.md
artifact = post_training/negbiodb_ct_oracle_sft_evidence_rationale_v1.jsonl
manifest = post_training/negbiodb_ct_oracle_sft_evidence_rationale_manifest.json
examples = 400
evidence_action_matches = 400
evidence_action_mismatches = 0
```

This is deterministic-oracle source data, not live runner behavior. It confirms
the evidence-rationale rule is not specific to the original n=40 native runner
reference.

## Guardrail Evaluation

Follow-up guardrail evaluation:

```text
anchor = post_training/EVIDENCE_RATIONALE_GUARDRAIL_EVAL_2026-06-27.md
json = post_training/evidence_rationale_guardrail_eval_2026-06-27.json
strict_model_action_accuracy = 0.500
strict_guardrail_action_accuracy = 1.000
strict_rescued_errors = 20
strict_introduced_errors = 0
constrained_model_action_accuracy = 0.500
constrained_guardrail_action_accuracy = 1.000
constrained_rescued_errors = 20
constrained_introduced_errors = 0
```

This confirms the extracted rule is not only a data artifact: used as a direct
override on the normal held-out model outputs, it fully routes the observed
boundary failures in this n=40 CV diagnostic.

## Distillation Diagnostic

Direct SFT distillation attempt:

```text
anchor = post_training/SFT_EVIDENCE_DISTILL_FOLD0_DIAGNOSTIC_2026-06-27.md
json = post_training/sft_evidence_distill_fold0_summary_2026-06-27.json
train_sft = post_training/negbiodb_ct_oracle_sft_evidence_rationale_v1.jsonl
eval = post_training/cv/negbiodb_ct_native_sft_cv4_v1_fold0_heldout.jsonl
train_examples = 400
train_loss_delta = -1.5084
strict_action_accuracy = 0.200
constrained_loaded_action_accuracy = 0.200
```

This is a negative fold0 diagnostic for internalization. The model can fit the
teacher-rationale rows, but the boundary rule does not transfer to unprompted
native held-out decisions. Keep the external layer as the deployable guardrail
candidate while testing a stronger distillation formulation.

Prompted-rule upper-bound with the same checkpoint:

```text
anchor = post_training/SFT_EVIDENCE_PROMPTED_UPPER_BOUND_FOLD0_2026-06-27.md
json = post_training/sft_evidence_prompted_upper_bound_fold0_summary_2026-06-27.json
base_prompt_strict_action_accuracy = 0.200
base_prompt_constrained_loaded_action_accuracy = 0.200
prompted_strict_action_accuracy = 1.000
prompted_constrained_loaded_action_accuracy = 1.000
```

This separates capability from internalization: the checkpoint follows the
rule when prompted, but the base native prompt does not elicit the rule.

Generative-rationale distillation artifact:

```text
anchor = post_training/SFT_GENERATIVE_RATIONALE_ARTIFACTS_2026-06-27.md
json = post_training/sft_generative_rationale_smoke_summary_2026-06-27.json
train_artifact = post_training/negbiodb_ct_oracle_sft_generative_rationale_v1.jsonl
train_examples = 400
fold0_artifact = post_training/generative_rationale/negbiodb_ct_native_sft_cv4_generative_rationale_v1_fold0_heldout.jsonl
fold0_examples = 10
smoke_loss = 2.3865
```

This is the next distillation formulation: the model is trained to generate the
evidence rationale before the final decision JSON, instead of merely consuming
a supplied rationale message.

Generative-rationale fold0 diagnostic:

```text
anchor = post_training/SFT_GENERATIVE_RATIONALE_FOLD0_DIAGNOSTIC_2026-06-27.md
json = post_training/sft_generative_rationale_fold0_diagnostic_2026-06-27.json
train_examples = 400
train_loss_delta = -2.3574
generative_fold0_strict_action_accuracy = 0.300
native_base_fold0_strict_action_accuracy = 0.000
native_base_fold0_strict_parse_failures = 10
native_base_fold0_constrained_loaded_action_accuracy = 0.400
```

This is a negative diagnostic for the current internalization route. The model
fits the generated-rationale train target by loss, but does not reliably
transfer the boundary rule to held-out generative targets or unprompted native
decisions. The deployable evidence-rationale layer remains the strongest path
until a stronger contrastive or preference distillation objective is tested.

Evidence-boundary preference artifact:

```text
anchor = post_training/BOUNDARY_PREFERENCE_ARTIFACTS_2026-06-27.md
artifact = post_training/negbiodb_ct_oracle_boundary_preferences_v1.jsonl
manifest = post_training/negbiodb_ct_oracle_boundary_preferences_manifest.json
source_examples = 400
preference_pairs = 620
chosen_passed = 620
rejected_passed = 0
```

This is the next contrastive formulation after the negative generative SFT
diagnostic. It keeps visible tool observations fixed and contrasts only the
terminal action, so it can feed a preference-margin diagnostic, DPO smoke run,
or reward/RLVR data path without changing the evidence layer itself.

Base margin diagnostic:

```text
anchor = post_training/BOUNDARY_PREFERENCE_MARGIN_BASE_2026-06-27.md
json = post_training/boundary_preference_margin_base_summary_2026-06-27.json
mean_win_rate = 0.615
hard_negative_modes = boundary_defer_over_verify 0.008/-0.1125, boundary_flag_over_ground 0.000/-0.1656, boundary_reject_over_ground 0.000/-2.6783, boundary_reject_over_flag 0.000/-2.5255
```

This keeps the deployable layer and the trainable objective separated: the
guardrail path is already exact on the current artifacts, while the preference
path should focus on negative-margin boundary modes before any broad DPO/RLVR
claim.

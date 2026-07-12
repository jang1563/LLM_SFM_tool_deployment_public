# SFT Evidence-Rationale Distillation Fold0 Diagnostic: 2026-06-27

This file records the first direct attempt to distill the evidence-rationale
rule into the model weights.

## Command

```bash
python3 post_training/run_sft_oracle_warmstart.py \
  --train-sft post_training/negbiodb_ct_oracle_sft_evidence_rationale_v1.jsonl \
  --train-limit 400 \
  --eval-cv-manifest post_training/negbiodb_ct_native_sft_cv4_manifest.json \
  --out-dir post_training/runs/qwen_oracle400_evidence_distill_base_cvheldout \
  --skip-base-constrained
```

The run completed training and fold0 held-out evaluation, then was stopped
before folds 1-3 because fold0 already showed a clear negative distillation
signal.

## Setup

```text
train_sft = post_training/negbiodb_ct_oracle_sft_evidence_rationale_v1.jsonl
train_examples = 400
eval_sft = post_training/cv/negbiodb_ct_native_sft_cv4_v1_fold0_heldout.jsonl
eval_examples = 10
eval_prompt_condition = native heldout base prompt, no evidence-rationale rule message
model = Qwen/Qwen2.5-0.5B-Instruct
max_steps = 160
batch_size = 2
max_length = 512
train_last_layers = 2
lr = 5e-5
```

## Result

```text
train_first_loss = 1.5101
train_last_loss = 0.0017
train_loss_delta = -1.5084
heldout_loaded_loss = 0.2534

strict_action_accuracy = 0.200
strict_parse_failures = 4
strict_by_class = defer 0/2, flag 0/2, ground 2/2, reject 0/2, verify 0/2

constrained_loaded_action_accuracy = 0.200
constrained_loaded_parse_failures = 0
constrained_loaded_by_class = defer 0/2, flag 0/2, ground 2/2, reject 0/2, verify 0/2
```

## Interpretation

This is a negative distillation diagnostic. The model can fit the 400-row
teacher-rationale artifact at the loss level, but that does not transfer the
evidence boundary rule into unprompted native held-out decisions. Both strict
generation and constrained scoring collapse to the `ground` boundary on fold0.

The external evidence-rationale layer remains useful as a guardrail/routing
component. Distillation now needs a stronger formulation before a full 4-fold
rerun is worth the cost: for example, prompted-rule eval as an upper-bound,
paired preference data around the rule failures, or a target format that trains
the model to generate the evidence-derived rationale rather than only consume
it.

## Prompted-Rule Upper-Bound

Follow-up anchor:

```text
anchor = post_training/SFT_EVIDENCE_PROMPTED_UPPER_BOUND_FOLD0_2026-06-27.md
json = post_training/sft_evidence_prompted_upper_bound_fold0_summary_2026-06-27.json
same_checkpoint = post_training/runs/qwen_oracle400_evidence_distill_base_cvheldout/train/trainable_state.pt
base_prompt_strict_action_accuracy = 0.200
prompted_strict_action_accuracy = 1.000
base_prompt_constrained_loaded_action_accuracy = 0.200
prompted_constrained_loaded_action_accuracy = 1.000
```

The same checkpoint fully recovers fold0 when the evidence-rationale message is
present. The negative result is therefore an internalization/retrieval failure
under the base prompt, not a failure to follow the rule once supplied.

# SFT Evidence-Rationale Prompted Upper-Bound Fold0: 2026-06-27

This file records the follow-up to the direct distillation fold0 negative
diagnostic. It uses the same trained checkpoint but evaluates fold0 with the
evidence-rationale message present in the held-out prompt.

## Commands

```bash
python3 post_training/run_sft_decision_eval.py \
  --state post_training/runs/qwen_oracle400_evidence_distill_base_cvheldout/train/trainable_state.pt \
  --sft post_training/boundary_rationale_heldout_evidence_ablation/negbiodb_ct_native_sft_cv4_boundary_rationale_heldout_evidence_v1_fold0_heldout.jsonl \
  --tasks negbiodb_ct/tasks_pilot.jsonl \
  --limit 10 \
  --device auto \
  --max-new-tokens 64 \
  --out post_training/runs/qwen_oracle400_evidence_distill_prompted_fold0/decision_eval.json

python3 post_training/run_sft_constrained_eval.py \
  --state post_training/runs/qwen_oracle400_evidence_distill_base_cvheldout/train/trainable_state.pt \
  --sft post_training/boundary_rationale_heldout_evidence_ablation/negbiodb_ct_native_sft_cv4_boundary_rationale_heldout_evidence_v1_fold0_heldout.jsonl \
  --tasks negbiodb_ct/tasks_pilot.jsonl \
  --limit 10 \
  --max-length 512 \
  --device auto \
  --score-mode mean \
  --out post_training/runs/qwen_oracle400_evidence_distill_prompted_fold0/constrained_loaded.json
```

## Comparison

Same checkpoint, fold0 heldout:

| eval prompt | strict acc | strict parse failures | constrained-loaded acc | class result |
| --- | ---: | ---: | ---: | --- |
| native base prompt | 0.200 | 4 | 0.200 | defer 0/2, flag 0/2, ground 2/2, reject 0/2, verify 0/2 |
| evidence-rationale prompted | 1.000 | 0 | 1.000 | defer 2/2, flag 2/2, ground 2/2, reject 2/2, verify 2/2 |

## Interpretation

The distillation checkpoint can follow the evidence-rationale rule when the
rule message is present at inference time. The previous negative result is
therefore not a lack of rule-following capacity; it is a failure to internalize
or retrieve the rule from the base native prompt.

This supports keeping the evidence-rationale layer as an external guardrail or
routing component for now. The next distillation formulation should train the
model to generate the evidence rationale itself, or use preference/contrastive
supervision that directly punishes the base-prompt `ground` collapse.

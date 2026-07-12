# Generative Evidence-Rationale Fold0 Diagnostic: 2026-06-27

This records the first held-out diagnostic for training the model to generate
an evidence rationale before the final decision JSON.

## Question

Does target-side generative rationale SFT make the model internalize the
evidence-derived boundary rule, or does it only fit the training format?

## Train Command

```bash
python3 post_training/run_sft_smoke.py \
  --sft post_training/negbiodb_ct_oracle_sft_generative_rationale_v1.jsonl \
  --limit 400 \
  --max-steps 160 \
  --batch-size 2 \
  --max-length 768 \
  --train-last-layers 2 \
  --lr 5e-5 \
  --device auto \
  --out-dir post_training/runs/qwen_oracle400_generative_rationale_fold0_diagnostic/train
```

Training result:

```text
model = Qwen/Qwen2.5-0.5B-Instruct
device = mps
examples = 400
max_steps = 160
train_last_layers = 2
trainable_params = 29825664
first_loss = 2.3930
last_loss = 0.0356
loss_delta = -2.3574
```

## Eval Commands

Strict held-out generative-rationale target:

```bash
python3 post_training/run_sft_decision_eval.py \
  --state post_training/runs/qwen_oracle400_generative_rationale_fold0_diagnostic/train/trainable_state.pt \
  --sft post_training/generative_rationale/negbiodb_ct_native_sft_cv4_generative_rationale_v1_fold0_heldout.jsonl \
  --tasks negbiodb_ct/tasks_pilot.jsonl \
  --limit 10 \
  --device auto \
  --max-new-tokens 192 \
  --out post_training/runs/qwen_oracle400_generative_rationale_fold0_diagnostic/generative_fold0_decision_eval.json
```

Strict native/base held-out prompt:

```bash
python3 post_training/run_sft_decision_eval.py \
  --state post_training/runs/qwen_oracle400_generative_rationale_fold0_diagnostic/train/trainable_state.pt \
  --sft post_training/cv/negbiodb_ct_native_sft_cv4_v1_fold0_heldout.jsonl \
  --tasks negbiodb_ct/tasks_pilot.jsonl \
  --limit 10 \
  --device auto \
  --max-new-tokens 192 \
  --out post_training/runs/qwen_oracle400_generative_rationale_fold0_diagnostic/base_fold0_decision_eval.json
```

Constrained native/base held-out candidate scoring:

```bash
python3 post_training/run_sft_constrained_eval.py \
  --state post_training/runs/qwen_oracle400_generative_rationale_fold0_diagnostic/train/trainable_state.pt \
  --sft post_training/cv/negbiodb_ct_native_sft_cv4_v1_fold0_heldout.jsonl \
  --tasks negbiodb_ct/tasks_pilot.jsonl \
  --limit 10 \
  --max-length 768 \
  --device auto \
  --score-mode mean \
  --out post_training/runs/qwen_oracle400_generative_rationale_fold0_diagnostic/base_fold0_constrained_loaded.json
```

## Result

| condition | action accuracy | parse failures | class accuracy |
| --- | ---: | ---: | --- |
| generative-rationale strict fold0 | 0.300 | 0 | defer 1/2, flag 0/2, ground 2/2, reject 0/2, verify 0/2 |
| native/base strict fold0 | 0.000 | 10 | defer 0/2, flag 0/2, ground 0/2, reject 0/2, verify 0/2 |
| native/base constrained fold0 | 0.400 | 0 | defer 0/2, flag 0/2, ground 2/2, reject 0/2, verify 2/2 |

Machine-readable summary:

```text
post_training/sft_generative_rationale_fold0_diagnostic_2026-06-27.json
```

## Interpretation

This is a negative fold0 diagnostic for the current generative-rationale SFT
formulation. The model fits the 400-row training artifact by loss, but held-out
generative-rationale generation reaches only 0.300 action accuracy. The same
checkpoint fails native/base strict generation with 10/10 parse failures, and
parse-free constrained scoring recovers only `ground` and `verify`.

The result does not invalidate the evidence-rationale rule. Earlier guardrail
and prompted-rule diagnostics still show the rule is strong when supplied or
used externally. This result says the current target-side generative-rationale
SFT objective is not yet a reliable internalization route.

## Next Safe Action

Do not scale this exact formulation to full CV before changing the objective.
Prefer a contrastive/preference boundary objective around `defer`/`verify`,
`flag`/`ground`, and `reject`/`ground`, or keep the evidence-rationale layer as
an external guardrail/routing component while testing that next formulation.

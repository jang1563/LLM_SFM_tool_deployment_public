# SFT Generative Evidence-Rationale Artifacts: 2026-06-27

This file records the first artifact for training the model to generate the
evidence rationale itself before emitting the final decision JSON.

## Commands

```bash
python3 post_training/build_sft_generative_rationale_data.py

python3 post_training/build_sft_generative_rationale_data.py \
  --sft post_training/cv/negbiodb_ct_native_sft_cv4_v1_fold0_heldout.jsonl \
  --out post_training/generative_rationale/negbiodb_ct_native_sft_cv4_generative_rationale_v1_fold0_heldout.jsonl \
  --manifest-out post_training/negbiodb_ct_native_sft_cv4_generative_rationale_fold0_manifest.json \
  --dataset negbiodb_ct_native_sft_generative_rationale_fold0_v1

python3 post_training/run_sft_smoke.py \
  --dry-run \
  --sft post_training/negbiodb_ct_oracle_sft_generative_rationale_v1.jsonl \
  --limit 20 \
  --max-length 768 \
  --out-dir post_training/runs/qwen_oracle400_generative_rationale_encode_dry

python3 post_training/run_sft_smoke.py \
  --dry-run \
  --sft post_training/generative_rationale/negbiodb_ct_native_sft_cv4_generative_rationale_v1_fold0_heldout.jsonl \
  --limit 10 \
  --max-length 768 \
  --out-dir post_training/runs/qwen_generative_rationale_fold0_encode_dry

python3 post_training/run_sft_smoke.py \
  --sft post_training/negbiodb_ct_oracle_sft_generative_rationale_v1.jsonl \
  --limit 4 \
  --max-steps 1 \
  --batch-size 1 \
  --max-length 768 \
  --train-last-layers 1 \
  --device auto \
  --out-dir post_training/runs/qwen_generative_rationale_smoke
```

## Artifacts

```text
train_artifact = post_training/negbiodb_ct_oracle_sft_generative_rationale_v1.jsonl
train_manifest = post_training/negbiodb_ct_oracle_sft_generative_rationale_manifest.json
train_examples = 400
train_by_action_class = defer 120, flag 40, ground 140, reject 40, verify 60
train_evidence_action_matches = 400
train_evidence_action_mismatches = 0

fold0_eval_artifact = post_training/generative_rationale/negbiodb_ct_native_sft_cv4_generative_rationale_v1_fold0_heldout.jsonl
fold0_eval_manifest = post_training/negbiodb_ct_native_sft_cv4_generative_rationale_fold0_manifest.json
fold0_eval_examples = 10
fold0_eval_by_action_class = defer 2, flag 2, ground 2, reject 2, verify 2
fold0_eval_evidence_action_matches = 10
fold0_eval_evidence_action_mismatches = 0
```

The prompt does not include `BOUNDARY_RATIONALE`. The target text includes a
`BOUNDARY_RATIONALE` line followed by `FINAL_SUBMIT_DECISION_JSON`.

## Smoke Checks

```text
oracle_encode_dry_examples = 20
oracle_encode_dry_max_length = 768
oracle_encode_dry_length_range = 422..751

fold0_encode_dry_examples = 10
fold0_encode_dry_max_length = 768
fold0_encode_dry_length_range = 424..570

smoke_examples = 4
smoke_max_steps = 1
smoke_batch_size = 1
smoke_train_last_layers = 1
smoke_trainable_params = 14913280
smoke_loss = 2.3865
```

## Boundary

This is an artifact and plumbing checkpoint, not a held-out performance claim.
It creates the next formulation for distillation: train the model to generate
the evidence-derived rationale from tool observations, rather than only
consume a rationale supplied by the prompt.

## Next Action

Run a fold0 generative-rationale diagnostic: train on the 400-row generative
rationale artifact, then evaluate on the fold0 generative-rationale held-out
artifact and the base native fold0 held-out artifact. The question is whether
generating the rationale improves unprompted boundary retrieval.

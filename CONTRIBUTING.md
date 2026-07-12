# Contributing

Thanks for your interest in improving LLM-SFM Tool Deployment.

This repository is a research benchmark artifact, so contributions should
preserve three boundaries:

- Benchmark rows must keep model-visible prompts separate from hidden evaluator
  metadata.
- Public artifacts must not include local paths, API keys, private databases,
  raw run logs, or private infrastructure breadcrumbs.
- Biological claims should stay tied to explicit evidence, validation commands,
  and limitations.

## Development Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For public-safe changes that do not require local model training, run:

```bash
pip install -r requirements-public.txt
python scripts/check_public_release.py
python scripts/check_public_git_history.py
python post_training/validate_post_training_data.py
python examples/run_public_demo.py
python post_training/run_stage_a_sft_smoke_eval.py --json
python post_training/generate_stage_a_predictions.py \
  --mode self_answer \
  --sft post_training/stage_a_sft_heldout_v1.jsonl \
  --out /tmp/stage_a_self_answer_predictions.jsonl \
  --run-id self_answer_saved_prediction_smoke
python post_training/evaluate_stage_a_predictions.py \
  --predictions /tmp/stage_a_self_answer_predictions.jsonl \
  --expected-sft post_training/stage_a_sft_heldout_v1.jsonl \
  --run-id self_answer_saved_prediction_smoke \
  --json
python -m pytest -q \
  tests/test_trajectory_evaluator.py \
  tests/test_public_demo.py \
  tests/test_public_release_checker.py \
  tests/test_stage_a_manifest.py \
  tests/test_stage_a_manifest_eval_script.py \
  tests/test_stage_a_export.py \
  tests/test_stage_a_split.py \
  tests/test_stage_a_sft_smoke_eval.py \
  tests/test_stage_a_prediction_eval.py \
  tests/test_stage_a_prediction_generator.py \
  tests/test_post_training_data_validator.py
```

Run the full suite when changing shared evaluator, adapter, or post-training
logic:

```bash
python -m pytest -q
```

## Pull Request Checklist

- Explain the benchmark behavior changed by the PR.
- Add or update tests for evaluator, manifest, split, or validation logic.
- Re-run the public-release checker before submitting.
- Avoid adding new private-data dependencies to public demos or CI.

## Scope

Good contributions include:

- Better validators or leakage checks.
- Clearer benchmark documentation and dataset cards.
- Public-safe synthetic examples.
- New deterministic baselines or audited evaluation slices.

Out of scope:

- Clinical decision support.
- Raw private database exports.
- Unaudited LLM-judge rewards.
- Claims that oracle-derived data reflects live model discovery.

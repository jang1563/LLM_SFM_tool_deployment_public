# Reviewer Guide

This guide is for research collaborators, benchmark users, auditors, and
open-source maintainers evaluating the repository quickly.

## Five-Minute Review

Read these in order:

1. `README.md`
2. `BENCHMARK_CARD.md`
3. `REPRODUCIBILITY.md`
4. `RESEARCH_SUMMARY.md`
5. `research/2026-06-25_posttrain_tool_use_landscape/LONG_TERM_RESEARCH_PLAN_2026-07-04.md`

What to look for:

- the project evaluates trajectories, not prose;
- hidden labels stay out of the model-visible prompt;
- the public path runs without private databases, model weights, API keys, or
  HPC access;
- negative or null results are preserved rather than smoothed over.

## Fifteen-Minute Technical Review

Inspect these files:

| File | Why it matters |
|---|---|
| `llm_sfm_tool_deployment/trajectory.py` | Shared schema and deterministic evaluator |
| `negbiodb_ct/stage_a_manifest.py` | Stage A manifest adapter, validator, oracle, and bad trajectories |
| `post_training/export_stage_a_data.py` | SFT, preference, and process-supervision export logic |
| `post_training/validate_post_training_data.py` | Data-integrity validator |
| `post_training/run_stage_a_sft_smoke_eval.py` | No-API Stage A SFT smoke/eval harness |
| `post_training/generate_stage_a_predictions.py` | Artifact-first producer for saved prediction JSONL |
| `post_training/evaluate_stage_a_predictions.py` | Offline scorer for saved API/local-SFT/prompt-only prediction JSONL |
| `scripts/check_public_release.py` | Public-surface safety and artifact checker |
| `tests/test_stage_a_manifest.py` | Hidden-label isolation and baseline failure tests |
| `tests/test_post_training_data_validator.py` | Post-training data validator tests |

The key design question is whether the project correctly separates:

- model-visible task text;
- hidden evaluator metadata;
- tool trajectory;
- evidence packet;
- terminal action;
- public-release boundary.

## Reproduce the Public Path

```bash
python -m venv .venv
source .venv/bin/activate
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
python post_training/evaluate_stage_a_predictions.py \
  --predictions post_training/stage_a_sft_heldout_v1.jsonl \
  --expected-sft post_training/stage_a_sft_heldout_v1.jsonl \
  --run-id heldout_oracle_adapter_smoke \
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

Expected outcome:

- release check passes;
- history check passes;
- post-training validator reports no issues;
- public demo prints passing and failing trajectory examples;
- Stage A SFT smoke/eval prints split-aware deterministic policy baselines;
- Stage A saved-prediction producer writes a no-API self-answer prediction file
  before scoring it;
- Stage A prediction scorer can score saved held-out prediction trajectories
  without API calls;
- public-safe tests pass.

## Claim Boundary

Strong claims this repo supports:

- deterministic validators can catch shortcut tool-use trajectories;
- Stage A has public-safe SFT, preference, and process-supervision artifacts;
- train/held-out Stage A splits are checked for case and source overlap;
- prompt-only/API/local-SFT result generation is artifact-first, with live API
  calls kept outside the public-safe path;
- saved model/API/local-SFT outputs can be scored offline against the same
  hidden-eval metadata and violation taxonomy;
- public release safety is machine-checked through a manifest and scanner.

Claims this repo does not make:

- clinical decision support;
- autonomous biological discovery;
- broad clinical benchmark coverage;
- proof that RLVR solves scientific reasoning;
- proof that explanation quality alone is a reliable reward.

## Reviewer Heuristics

Good signs:

- bad trajectories fail for specific violation codes;
- hidden labels are absent from prompt messages;
- artifact counts and checksums are registered;
- the public demo is synthetic and runnable;
- limitations are explicit.

Things to challenge:

- whether Stage A is large enough for the intended claim;
- whether future HF uploads preserve the same leakage boundary;
- whether C5 transfer will require stronger calibration artifacts;
- whether any future RL reward uses audited deterministic slices only.

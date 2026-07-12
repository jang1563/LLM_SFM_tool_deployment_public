# Reproducibility

This repository has two reproducibility paths:

- public-safe validation, which runs without private databases, model weights,
  API keys, or HPC access;
- full local experimentation, which may require model-training dependencies and
  private NegBioDB/A2 artifacts.

The public-safe path is the canonical reviewer path.

## Public-Safe Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-public.txt
```

This installs only the lightweight packages needed for validators, public demo,
and public-safe tests.

## Public-Safe Checks

```bash
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
python post_training/run_stage_a_saved_output_calibration_margin_sft.py \
  --dry-run \
  --out-dir /tmp/stage_a_saved_output_calibration_margin_sft \
  --run-id stage_a_saved_output_calibration_margin_sft_dry \
  --pairwise-margin-weight 1 \
  --pairwise-margin 0.05 \
  --score-base-margins \
  --score-train-margins
python -m pytest -q \
  tests/test_trajectory_evaluator.py \
  tests/test_public_demo.py \
  tests/test_public_release_checker.py \
  tests/test_stage_a_manifest.py \
  tests/test_stage_a_manifest_eval_script.py \
  tests/test_stage_a_export.py \
  tests/test_stage_a_strict_contract_export.py \
  tests/test_stage_a_strict_contract_sft_smoke.py \
  tests/test_stage_a_split.py \
  tests/test_stage_a_sft_smoke_eval.py \
  tests/test_stage_a_prediction_eval.py \
  tests/test_stage_a_prediction_generator.py \
  tests/test_stage_a_saved_output_calibration_margin_sft.py \
  tests/test_post_training_data_validator.py
```

Expected high-level outcome:

- public release check passes;
- public git-history check passes;
- post-training validator reports `"issues": []`;
- public demo shows both passing and failing trajectory examples;
- Stage A SFT smoke/eval reports held-out oracle 5/5 and shortcut policies 0/5;
- Stage A saved-prediction producer writes a self-answer artifact that scores
  0/5, preserving shortcut-failure behavior;
- Stage A prediction scorer reports held-out oracle adapter smoke 5/5;
- Stage A saved-output calibration margin SFT dry-run reports 16 train-only
  pairs, 4 held-out evaluation-only pairs, and no issues;
- public-safe pytest subset passes.

These commands are also run by the GitHub Actions `Public QA` workflow.

## Full Local Checks

For local experiment development:

```bash
pip install -r requirements.txt
python -m pytest -q
```

The full dependency set includes model-training and API-client packages. It is
not required to review the public benchmark substrate.

## Artifact Integrity

Public artifact paths, record counts, and SHA-256 checksums are registered in:

```text
release/public_release_manifest.json
```

The checker verifies:

- required public-surface files exist;
- JSONL counts match the manifest;
- checksums match for registered artifacts;
- the public demo remains synthetic;
- tracked files do not include common secret, local-path, private
  infrastructure, or generated-cache patterns.

## Data Boundary

The public path does not require:

- private NegBioDB SQLite databases;
- raw private database exports;
- OpenAI, Anthropic, or Hugging Face tokens;
- local model-cache paths;
- cluster account, allocation, partition, or scratch-storage identifiers.

Private or site-specific paths may appear only as generic placeholders in docs,
for example `/path/to/...` or `<local-workspace>/...`.

## Stage A Determinism

Stage A exports are deterministic for the tracked manifest:

- 25 manifest cases;
- 25 SFT rows;
- 150 preference pairs;
- 25 process-supervision rows;
- 20/5 train-held-out split with no case, split-group, or source-task overlap.
- 25 strict-contract SFT rows, 50 strict-contract preference pairs, and 25
  strict-contract process rows for the `stage_a_v2_strict` JSON output contract.

The validator checks that chosen preference trajectories pass, rejected
trajectories fail, strict-contract observed-collapse rejected targets fail, and
train/eval source overlap remains zero.

Stage A prediction-output scoring is also deterministic. The public smoke
command reuses the held-out SFT oracle trajectories as saved predictions to
exercise the offline scorer. Real API or cluster model runs should write the
same JSONL shape first, then score the saved file.

`post_training/generate_stage_a_predictions.py` is the artifact-first producer.
Public modes are deterministic and no-API. Live OpenAI chat generation is
available only through `--mode openai_chat --allow-live-api` with
`OPENAI_API_KEY` set; it is not part of the public-safe path.
For GPU model inference, use the Cayuga or Expanse sbatch templates in
`post_training/` so raw run artifacts stay under ignored `post_training/runs/`.
The first compact tracked cluster summary is
`post_training/STAGE_A_CAYUGA_HF_CHAT_BASELINE_2026-07-04.md`; raw model
outputs remain untracked.
The strict-contract follow-up is tracked at
`post_training/STAGE_A_CAYUGA_STRICT_CONTRACT_2026-07-04.md` and likewise keeps
raw model outputs untracked.
The next cluster-only follow-up entrypoint is
`post_training/run_stage_a_strict_contract_sft_smoke.py`; its public-safe
`--dry-run` validates the strict train/held-out artifacts without loading model
weights, while Cayuga/Expanse sbatch templates run the full tiny SFT smoke.

## CI

The public repository uses:

```text
.github/workflows/public-qa.yml
```

The workflow runs on pushes and pull requests to `main` with read-only
repository permissions.

## Non-Reproducible From Public Mirror Alone

The following are intentionally outside the public-safe path:

- live LLM API calls;
- full private NegBioDB-CT task regeneration;
- local adapter checkpoints or model caches;
- HPC job execution;
- unpublished Hugging Face uploads.

Those paths should be treated as local research workflows until a separate
public-compatible artifact package is approved.

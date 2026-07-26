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
python -m c5_antibody_ood.manifest \
  --out /tmp/c5_policy_test_manifest_v1.jsonl
cmp c5_antibody_ood/c5_policy_test_manifest_v1.jsonl \
  /tmp/c5_policy_test_manifest_v1.jsonl
python -m c5_antibody_ood.evaluate_baselines \
  --out-json /tmp/c5_policy_baseline_result.json \
  --out-md /tmp/C5_POLICY_BASELINE_RESULT.md
python -m pytest -q tests/test_c5_source_pilot.py
python -m pytest -q tests/test_c5_independent_calibration.py
python -m c5_antibody_ood.prospective_panel \
  --check c5_antibody_ood/c5_prospective_panel_preregistration_v1.json
python -m pytest -q \
  tests/test_c5_prospective_panel.py \
  tests/test_c5_prospective_source.py \
  tests/test_c5_prospective_inputs.py \
  tests/test_c5_af3_preflight.py \
  tests/test_c5_prospective_predictions.py \
  tests/test_c5_prospective_native_lock.py \
  tests/test_c5_prospective_reveal.py
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
  tests/test_c5_manifest.py \
  tests/test_c5_policies.py \
  tests/test_c5_source_pilot.py \
  tests/test_c5_independent_calibration.py \
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
- C5 manifest regeneration is byte-identical and its no-API fail-closed policy
  passes 12/12 with zero unsafe trust;
- C5 source-pilot tests verify source hash/schema failure, target-grouped
  splitting, deterministic tie handling, hidden-label isolation, and privacy
  projection without downloading the external score table;
- C5 independent-calibration tests verify source identity, source-protocol tie
  handling, PDB-level overlap exclusion, format-specific certificates, locked
  Fromm transfer, hidden-label isolation, and privacy projection;
- C5 prospective tests verify protocol immutability, exact panel roles,
  prior-PDB/source-cluster isolation, no-label-read boundaries, reserve
  promotion, input commitments, and fail-closed AF3 environment attestation;
- public-safe pytest subset passes.

These commands are also run by the GitHub Actions `Public QA` workflow.

## External C5 Source Replay

The raw 24.6 MB source table is not redistributed. After obtaining
`scores_alphafold3.csv` from the `CC-BY-4.0` Zenodo release documented in
`c5_antibody_ood/SOURCE_BACKED_PILOT_PROVENANCE.md`, reproduce the tracked
derived manifest and compact report with:

```bash
python -m c5_antibody_ood.evaluate_source_pilot \
  --scores /external/path/scores_alphafold3.csv \
  --out-manifest /tmp/c5_source_backed_manifest_v1.jsonl \
  --out-json /tmp/c5_source_backed_pilot_result.json \
  --out-md /tmp/C5_SOURCE_BACKED_PILOT_RESULT.md
cmp c5_antibody_ood/c5_source_backed_manifest_v1.jsonl \
  /tmp/c5_source_backed_manifest_v1.jsonl
```

The adapter fails on a source checksum, row/target/sample count, preset,
required-field, duplicate-ID, or unit-interval mismatch. The compact report
stores source fingerprints and aggregates, never the local input path.

## External C5 Independent Calibration

The raw Hitawala-Gray table is not redistributed. Obtain the pinned
`datafiles/final_af3_rmsds.csv` documented in
`c5_antibody_ood/INDEPENDENT_CALIBRATION_PROVENANCE.md`, then reproduce the
tracked overlap-excluded manifest and compact report with:

```bash
python -m c5_antibody_ood.evaluate_independent_calibration \
  --gray-scores /external/path/final_af3_rmsds.csv \
  --out-manifest /tmp/c5_gray_independent_calibration_manifest_v1.jsonl \
  --out-json /tmp/c5_gray_independent_calibration_result.json \
  --out-md /tmp/C5_GRAY_INDEPENDENT_CALIBRATION_RESULT.md
cmp c5_antibody_ood/c5_gray_independent_calibration_manifest_v1.jsonl \
  /tmp/c5_gray_independent_calibration_manifest_v1.jsonl
```

The adapter fails on source drift, malformed scientific values, missing bound
labels, duplicate sample IDs, or residual PDB overlap with Fromm. The external
CSV path and source filenames are never emitted.

## Prospective C5 Freeze And Execution

The public method, panel, retained-target, input-freeze, and readiness
artifacts can be validated without raw SAbDab2 structures or AF3 dependencies:

```bash
python -m c5_antibody_ood.prospective_panel \
  --check c5_antibody_ood/c5_prospective_panel_preregistration_v1.json
python scripts/check_research_plan.py
python -m pytest -q \
  tests/test_c5_prospective_panel.py \
  tests/test_c5_prospective_source.py \
  tests/test_c5_prospective_inputs.py \
  tests/test_c5_af3_preflight.py
```

Full prediction requires a private environment. Run the preflight with
site-specific paths and the expected private checksums:

On Cayuga, do not use the login node's host Python for these module commands.
Invoke them inside the pinned AF3 image with `--pwd /app/alphafold`,
`PYTHONPATH` bound to this repository, and `uv run --no-sync python3`, matching
`run_c5_af3_cayuga.sbatch`.

Build the parameter-free v3.0.3 image on a CPU node from a clean pinned AF3
source tree:

```bash
sbatch --account=<allocation> --partition=scu-cpu \
  --export=ALL,WORK=$PWD,AF3_SOURCE_DIR=<clean-v3.0.3-source>,AF3_SIF_OUT=<new-image.sif> \
  c5_antibody_ood/build_c5_af3_container_cayuga.sbatch
```

The job writes a SIF, SHA-256 sidecar, and path-free private build manifest
only after embedded source/package, provenance-label, and runner-import tests
pass. `c5_af3_container_readiness_2026-07-25.json` is the public compact
projection of the first completed build and GPU runtime smoke.

Provision the official databases to a new private directory. The job requires
at least 750 GB free, uses the official fetch script, hashes every completed
database file into a private inventory, and promotes the staging directory
only after validation:

```bash
sbatch --account=<allocation> --partition=scu-cpu \
  --export=ALL,WORK=$PWD,AF3_SIF=<image>,AF3_SIF_SHA256=<sha256>,AF3_DB_OUT=<new-database-dir>,AF3_DB_MANIFEST_OUT=<new-private-manifest> \
  c5_antibody_ood/fetch_c5_af3_databases_cayuga.sbatch
```

The first completed run atomically promoted all 9 required entries and bound
195,867 files by content SHA-256. Its private sidecar recheck passes, while
`c5_af3_database_readiness_2026-07-26.json` exposes only aggregate sizes,
content identities, and gate status.

Model parameters must be requested through the
[official AlphaFold 3 access form](https://forms.gle/svvpY4u2jsHEwWYS6) and
received directly from Google under its terms. After placing the authorized
download in a private source directory, provision it atomically:

```bash
sbatch --account=<allocation> --partition=scu-cpu \
  --export=ALL,WORK=$PWD,AF3_SIF=<image>,AF3_SIF_SHA256=<sha256>,AF3_MODEL_SOURCE=<authorized-download-dir>,AF3_MODEL_OUT=<new-model-dir>,AF3_MODEL_MANIFEST_OUT=<new-private-manifest>,AF3_AUTHORIZED_SOURCE_CONFIRMED=YES \
  c5_antibody_ood/provision_c5_af3_parameters_cayuga.sbatch
```

The job rejects an absent authorization assertion, symlinks, multiple parameter
families, dirty output boundaries, and copy drift. It promotes only a
content-hashed private model directory and manifest. The confirmation is a
user assertion of provenance, not independent license verification.

After those private dependencies are present, run the following module
commands inside the pinned image:

```bash
uv run --no-sync python3 -m c5_antibody_ood.af3_preflight inventory \
  --database-dir <af3-database-dir> \
  --out <private-database-inventory>

uv run --no-sync python3 -m c5_antibody_ood.af3_preflight run \
  --preregistration c5_antibody_ood/c5_prospective_panel_preregistration_v1.json \
  --input-freeze c5_antibody_ood/c5_sabdab2_prospective_af3_input_freeze_2026-07-25.json \
  --retained-manifest c5_antibody_ood/c5_sabdab2_prospective_retained_manifest_v1.jsonl \
  --input-dir <private-af3-input-dir> \
  --source-dir <pinned-af3-source-dir> \
  --container <af3-singularity-image> \
  --expected-container-sha256 <sha256> \
  --model-dir <authorized-model-parameter-dir> \
  --expected-model-sha256 <sha256> \
  --database-dir <af3-database-dir> \
  --database-manifest <private-database-inventory> \
  --expected-database-manifest-sha256 <sha256> \
  --output-dir <new-output-dir> \
  --attestation-out <private-attestation-json>
```

The command returns nonzero unless every component passes. The private
attestation and its SHA-256 are then required by
`c5_antibody_ood/run_c5_af3_cayuga.sbatch`. Parameters, databases, sequences,
structures, predictions, paths, and scheduler logs must remain uncommitted.

Before submission, rerun the mounted dependencies in full mode. Full mode
rehashes the container, model parameters, and every required database file.
The array repeats a lower-cost size, nanosecond-mtime, manifest, and
deterministic-sentinel identity check for every target:

```bash
uv run --no-sync python3 -m c5_antibody_ood.af3_preflight verify-runtime \
  --attestation <private-attestation-json> \
  --expected-attestation-sha256 <sha256> \
  --preregistration c5_antibody_ood/c5_prospective_panel_preregistration_v1.json \
  --input-freeze c5_antibody_ood/c5_sabdab2_prospective_af3_input_freeze_2026-07-25.json \
  --retained-manifest c5_antibody_ood/c5_sabdab2_prospective_retained_manifest_v1.jsonl \
  --input-dir <private-af3-input-dir> \
  --container <af3-singularity-image> \
  --model-dir <authorized-model-parameter-dir> \
  --database-dir <af3-database-dir> \
  --database-manifest <private-database-inventory> \
  --mode full
```

Submission must export `AF3_DB_MANIFEST` as well as the image, model, database,
input, output, attestation, and attestation-SHA variables. Keep
`AF3_RUNTIME_VERIFY_MODE=quick` for the array unless intentionally running a
small full-verification diagnostic.

Prefer the split path so CPU-heavy MSA/template search does not consume GPU
allocations. The first array promotes only a validated `<job>_data.json`; the
second stages inference from that file and replaces the processed target only
after the canonical five-sample intake passes:

```bash
sbatch --account=<allocation> --partition=scu-cpu \
  --export=ALL,WORK=$PWD,AF3_SIF=<image>,AF3_MODEL_DIR=<models>,AF3_DB_DIR=<databases>,AF3_DB_MANIFEST=<database-manifest>,AF3_INPUT_DIR=<inputs>,AF3_OUTPUT_DIR=<outputs>,AF3_ATTESTATION=<attestation>,AF3_ATTESTATION_SHA256=<sha256> \
  c5_antibody_ood/run_c5_af3_data_pipeline_cayuga.sbatch

sbatch --account=<allocation> --partition=scu-gpu --gres=gpu:a100:1 \
  --export=ALL,WORK=$PWD,AF3_SIF=<image>,AF3_MODEL_DIR=<models>,AF3_DB_DIR=<databases>,AF3_DB_MANIFEST=<database-manifest>,AF3_INPUT_DIR=<inputs>,AF3_OUTPUT_DIR=<outputs>,AF3_ATTESTATION=<attestation>,AF3_ATTESTATION_SHA256=<sha256> \
  c5_antibody_ood/run_c5_af3_inference_cayuga.sbatch
```

Both arrays rebind the private attestation before every task. The original
`run_c5_af3_cayuga.sbatch` remains a combined-stage fallback; both paths produce
the same output tree consumed below.

After the 120 jobs complete, freeze prediction outputs before opening any
DockQ label file:

```bash
python -m c5_antibody_ood.prospective_predictions \
  --preregistration c5_antibody_ood/c5_prospective_panel_preregistration_v1.json \
  --input-freeze c5_antibody_ood/c5_sabdab2_prospective_af3_input_freeze_2026-07-25.json \
  --retained-manifest c5_antibody_ood/c5_sabdab2_prospective_retained_manifest_v1.jsonl \
  --private-input-dir <private-af3-input-dir> \
  --attestation <private-attestation-json> \
  --expected-attestation-sha256 <sha256> \
  --af3-output-root <private-af3-output-root> \
  --private-lock-out <private-prediction-lock-json> \
  --public-freeze-out <public-prediction-freeze-json>

python -m c5_antibody_ood.prospective_native_lock \
  --candidate-manifest c5_antibody_ood/c5_sabdab2_prospective_panel_manifest_v1.jsonl \
  --retained-manifest c5_antibody_ood/c5_sabdab2_prospective_retained_manifest_v1.jsonl \
  --input-freeze c5_antibody_ood/c5_sabdab2_prospective_af3_input_freeze_2026-07-25.json \
  --structures-dir <private-native-structure-dir> \
  --private-out <private-native-structure-lock-json>
```

Only then reveal calibration labels. The second command remains blocked until
the calibration lock exists and passes recomputation:

```bash
python -m c5_antibody_ood.prospective_reveal calibrate \
  --preregistration c5_antibody_ood/c5_prospective_panel_preregistration_v1.json \
  --input-freeze c5_antibody_ood/c5_sabdab2_prospective_af3_input_freeze_2026-07-25.json \
  --retained-manifest c5_antibody_ood/c5_sabdab2_prospective_retained_manifest_v1.jsonl \
  --prediction-lock <private-prediction-lock-json> \
  --native-structure-lock <private-native-structure-lock-json> \
  --labels <private-calibration-dockq-jsonl> \
  --private-out <private-calibration-lock-json> \
  --public-out <public-calibration-freeze-json>

python -m c5_antibody_ood.prospective_reveal evaluate \
  --preregistration c5_antibody_ood/c5_prospective_panel_preregistration_v1.json \
  --input-freeze c5_antibody_ood/c5_sabdab2_prospective_af3_input_freeze_2026-07-25.json \
  --retained-manifest c5_antibody_ood/c5_sabdab2_prospective_retained_manifest_v1.jsonl \
  --prediction-lock <private-prediction-lock-json> \
  --native-structure-lock <private-native-structure-lock-json> \
  --calibration-lock <private-calibration-lock-json> \
  --labels <private-evaluation-dockq-jsonl> \
  --private-out <private-evaluation-lock-json> \
  --public-out <public-evaluation-result-json>
```

The private label schema binds every DockQ value to the selected model
checksum, native-structure checksum, committed chain mapping, metric scope,
and pinned evaluator identity. Calibration and evaluation target sets must be
exact and disjoint.

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

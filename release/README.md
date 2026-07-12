# Release Surface

This directory tracks the machine-readable public-release gate for the
LLM-SFM Tool Deployment benchmark package.

## Current Profile

- Release profile: `benchmark_public_surface`
- Primary object: Stage A biology tool-use trajectory benchmark substrate
- Repository form: published and validated clean-snapshot release surface
- GitHub: `https://github.com/jang1563/LLM_SFM_tool_deployment_public`
- Earlier private development history is intentionally not included
- License status: all rights reserved until a final open-source or
  lab-approved license decision is made

## Public Manifest

`public_release_manifest.json` lists the files that are intended to be part of
the public surface. For artifacts, it records:

- path
- artifact kind
- JSONL record count when applicable
- SHA-256 checksum
- `safe_to_publish` flag

The manifest currently includes the synthetic trajectory demo, public figure
inputs, Stage A manifest, Stage A SFT/preference/process exports, Stage A
train/held-out split artifacts, strict-contract compact JSON targets,
component-slice targets, evidence-conditioned component targets, and compact
public-safe Stage A result summaries. It also includes the routing action/status
contrast artifacts that isolate the unresolved constrained-routing failure
families, plus the compact public-safe Cayuga routing contrast SFT/margin
and candidate-rank results. Raw cluster reports, candidate JSONL files, model
states, run folders, and scheduler logs remain out of release scope.
The latest Stage A routing substrate also includes the 10-row defer-vs-verify
boundary contrast set for insufficient-evidence versus verification-needed
routing; it is included as corrective benchmark data, not as a model result.
The corresponding compact Cayuga smoke result is included as a negative/partial
component checkpoint; raw run artifacts remain out of release scope.
The follow-up fail-closed gate diagnostic is included as a compact runtime
enforcement readout, not as deployment calibration.
The deterministic evidence-boundary gate is included because it runs on public
artifacts and represents the no-model runtime-enforcement baseline to beat.
The all-family routing evidence gate extends that runtime baseline across all
25 Stage A `routing_after_loop` evidence-conditioned rows.
The routing gate baseline-comparison report is included because it shows how
collapse and citationless component baselines fail against that runtime gate.
The model-readiness report is included because it compares compact Cayuga
summaries against that scorecard without reading raw run outputs.
The full-trajectory arbitration report is included because it projects runtime
policies through the canonical Stage A trajectory evaluator, not only component
scorers.
The saved-prediction readiness report is included because it compares compact
real Cayuga saved-output summaries against those full-trajectory baselines
without committing raw prediction JSONL.
The `stage_a_v3_tool_trace` Cayuga summary is included because it records the
next saved-output blocker without committing raw prediction JSONL: the model
uses invalid `evidence_status: verified` in all held-out rows.
The `stage_a_v4_canonical_json` Cayuga summary is included because it records
the follow-up negative result: invalid `verified` status disappears, but the
canonical top-level action/JSON envelope still fails.
The saved-prediction candidate-readout summaries are included because they
record the constrained follow-up: parse/tool/query gates pass, but finite
candidate scoring still collapses to `ground` / `supported`.
The saved-output calibration probe and margin-SFT dry-run path are included as
targeted corrective scaffolding; raw margin JSONL, trainable state, and
scheduler logs remain out of release scope unless compact summaries are curated.
The first compact margin-SFT result is included because it records partial
movement without overclaiming repair: held-out wins improve from 0/4 to 1/4,
with mean margin still below zero.
The focused compact margin-SFT follow-up is included because it records the
next corrective diagnosis: non-verify family oversampling improves held-out
wins to 3/4, but `flag` / `invalid_value` remains unresolved, so readiness and
optimization escalation remain gated.
The saved-output target-format diagnostic path is included because it lets the
next Cayuga run isolate action/status learning from full JSON, citation, and
tool-call target coupling without publishing raw model outputs.
The target-format result is included because it records that isolated
`flag`, `invalid_value`, and `flag` + `invalid_value` projections can pass, but
the full JSON target remains unresolved.
The same-model target-format scoring path is included because it separates
projection scoring effects from separate projection-specific training runs.
The same-model target-format result is included because it records the first
teacher-forced full-target `flag` / `invalid_value` repair signal while keeping
candidate ranking, free generation, and full-trajectory readiness out of scope.
The saved-output margin SFT runner now also has an optional finite-candidate
rank readout; raw candidate-score JSONL remains out of release scope unless a
compact summary is curated.
The first candidate-rank result is included because it shows the key negative
transfer: teacher-forced margin repair does not yet solve finite-candidate
selection.
The candidate field diagnostic is included because it explains that failure as
combined `flag` / `invalid_value` over-selection rather than one isolated field
miss.
The arbitration report is included because it compares public-safe runtime
policies without requiring raw model outputs.

For human-readable review, start with `REVIEWER_GUIDE.md`,
`BENCHMARK_CARD.md`, `BENCHMARK_VERIFIER_MAP.md`, `REPRODUCIBILITY.md`, and
`ROADMAP.md`.

## Validation Gate

Run before changing the public surface, tagging a release, or uploading to
Hugging Face:

```bash
python scripts/refresh_public_manifest.py --date YYYY-MM-DD
python scripts/check_public_release.py
python scripts/check_public_git_history.py
python post_training/validate_post_training_data.py
python post_training/evaluate_stage_a_routing_evidence_gate.py \
  --out-json /tmp/stage_a_routing_evidence_gate.json \
  --out-md /tmp/STAGE_A_ROUTING_EVIDENCE_GATE.md
python post_training/evaluate_stage_a_routing_gate_baseline_comparison.py \
  --out-json /tmp/stage_a_routing_gate_baseline_comparison.json \
  --out-md /tmp/STAGE_A_ROUTING_GATE_BASELINE_COMPARISON.md
python post_training/evaluate_stage_a_routing_model_readiness.py \
  --out-json /tmp/stage_a_routing_model_readiness.json \
  --out-md /tmp/STAGE_A_ROUTING_MODEL_READINESS.md
python post_training/evaluate_stage_a_full_trajectory_arbitration.py \
  --out-json /tmp/stage_a_full_trajectory_arbitration.json \
  --out-md /tmp/STAGE_A_FULL_TRAJECTORY_ARBITRATION.md
python post_training/evaluate_stage_a_saved_prediction_readiness.py \
  --out-json /tmp/stage_a_saved_prediction_readiness.json \
  --out-md /tmp/STAGE_A_SAVED_PREDICTION_READINESS.md
python post_training/run_stage_a_saved_prediction_candidate_readout.py \
  --dry-run \
  --out-dir /tmp/stage_a_saved_candidate_readout \
  --run-id stage_a_saved_candidate_readout_dry
python post_training/run_stage_a_saved_output_calibration_margin_sft.py \
  --dry-run \
  --out-dir /tmp/stage_a_saved_output_calibration_margin_sft \
  --run-id stage_a_saved_output_calibration_margin_sft_dry \
  --pairwise-margin-weight 1 \
  --pairwise-margin 0.05 \
  --score-base-margins \
  --score-train-margins
python -m pytest -q
git diff --check
```

The public-release checker validates manifest paths, JSONL counts, checksums,
demo safety, local path leakage, token-like strings, private key markers, and
private infrastructure breadcrumbs. It also blocks common generated or
private-file paths such as bytecode caches, virtual environments, environment
files, local databases, key files, and logs.

The git-history checker scans existing refs and commits. It should continue to
pass in this mirror before any public release tag or artifact upload.

## GitHub Repository

The public repository is:

```text
https://github.com/jang1563/LLM_SFM_tool_deployment_public
```

If this mirror needs to be recreated from a clean export, create an empty public
GitHub repository:

```text
owner: jang1563
name: LLM_SFM_tool_deployment_public
visibility: public
initialize with README/license/gitignore: no
```

Then push the recreated mirror:

```bash
cd <local-public-mirror>
git push -u origin main
```

Do not publish from the source-private repository. This mirror is the release
repository because it has fresh history and passes `check_public_git_history.py`.

## Release Boundary

Publishable:

- Stage A benchmark JSONL/JSON artifacts listed in the manifest
- Synthetic demo cases
- Public-safe figure inputs
- Human-readable cards, checklist, audit, and README files

Do not publish:

- private SQLite databases or raw DB exports
- API keys, tokens, private key files, local paths, or model cache paths
- cluster account/allocation details, scheduler logs, or scratch paths
- ignored generated run folders or local adapter checkpoints

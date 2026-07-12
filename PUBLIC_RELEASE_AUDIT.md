# Public Release Audit

Last checked: 2026-07-12.

## Redaction Applied

- Removed personal absolute paths from tracked public-facing docs and command logs.
- Removed non-project career-planning references from the release surface.
- Replaced local workspace paths with `<local-workspace>/...` placeholders.
- Replaced cluster allocation identifiers with `<allocation>` / `<gpu-partition>` placeholders.
- Removed concrete scheduler IDs from curated aggregate result summaries.
- Replaced local Anthropic key-file reads with `ANTHROPIC_API_KEY`.
- Added `.gitignore` protections for `.env`, key files, local databases, logs, virtualenvs, and generated runs.
- Added a synthetic public demo and public-safe figure inputs that do not rely on private data.
- Added `release/public_release_manifest.json` and `scripts/check_public_release.py` so the public surface,
  checksums, demo safety, and common secret/path patterns can be checked before release.
- Registered Stage A benchmark artifacts in the public-release manifest with
  record counts and checksums.
- Added Stage A train/held-out split validation with no
  `source_manifest_case_id`, `split_group`, or `source_task_id` overlap.
- Expanded the public-release checker to catch Hugging Face token patterns and
  private-key markers in addition to local paths, API tokens, and infrastructure
  breadcrumbs.
- Expanded the public-release checker to fail on tracked or unignored generated
  caches, bytecode, environment files, local databases, key files, and logs.
- Added `scripts/check_public_git_history.py` to distinguish current-surface
  safety from full repository-history safety.
- Added career-framing, internal-document, and numeric scheduler-ID guards to
  the public-release checker.
- Added deterministic manifest checksum and JSONL record-count refresh tooling.
- Added Stage A saved-prediction producer and scorer smoke checks. Public modes
  are deterministic no-API paths; live API outputs should remain generated
  artifacts, not committed raw run logs.
- Added Stage A strict-contract SFT/preference/process artifacts to target the
  saved-prediction JSON output contract. Validator checks keep hidden Stage A
  labels out of prompt messages and verify chosen-pass/rejected-fail direction.
- Added strict-contract SFT smoke sbatch templates with placeholder allocation
  fields only. Generated cluster outputs remain ignored under `post_training/runs/`.
- Added evidence-conditioned component targets, constrained-routing summaries,
  and routing action/status contrast artifacts with manifest checksums and
  validator coverage for hidden-label isolation and split/source overlap.
- Added routing contrast SFT/margin dry-run and cluster templates; full model
  outputs remain ignored under `post_training/runs/`.
- Recorded the Cayuga routing contrast SFT/margin result as compact public-safe
  JSON/Markdown only; raw reports, model state, and Slurm logs remain
  uncommitted.
- Added routing contrast finite-candidate rank instrumentation to the existing
  SFT/margin runner; public CI validates the candidate-space dry-run without
  loading model weights.
- Recorded the Cayuga routing contrast candidate-rank result as compact
  public-safe JSON/Markdown only; raw candidate JSONL, model state, and Slurm
  logs remain uncommitted.
- Added Stage A routing defer-vs-verify boundary contrast pairs as public-safe
  JSONL/JSON artifacts; these are a targeted corrective substrate, not a new
  live model result.
- Recorded the Cayuga defer-vs-verify routing smoke as compact public-safe
  JSON/Markdown only; raw JSONL reports, model state, and scheduler logs remain
  uncommitted.
- Added a compact routing fail-closed gate diagnostic over ignored candidate
  score JSONL; the public artifact keeps only thresholds, compact candidate
  labels, scores, and checksums.
- Added a no-model routing evidence-boundary gate over public defer-vs-verify
  pair artifacts; it uses prompt-visible tool-result fields and no hidden labels.
- Added an all-family no-model routing evidence gate over public
  evidence-conditioned component targets; it uses prompt-visible tool-result
  fields and no hidden labels.
- Added a routing gate baseline-comparison report over public deterministic
  component baselines; it stores compact counts only and requires no raw model
  outputs or cluster run directories.
- Added a routing model-readiness report over compact public Cayuga summaries;
  it does not inspect raw predictions, model states, Slurm logs, or ignored run
  folders.
- Added a full-trajectory arbitration report over public deterministic policies;
  it uses the canonical Stage A evaluator and no raw model outputs.
- Added a saved-prediction readiness report over compact public Cayuga
  summaries and deterministic public smokes; it does not read raw prediction
  JSONL, scheduler logs, model states, or ignored run folders.
- Added the `stage_a_v3_tool_trace` prompt contract as a public-safe tool/query
  compliance diagnostic; tests assert that hidden labels, source task IDs, and
  held-out NCT source IDs are not exposed in the prompt.
- Added the compact v3 Cayuga saved-output summary and
  `stage_a_v4_canonical_json` prompt contract; neither artifact includes raw
  model text, scheduler logs, local cache paths, hidden labels, or source task
  IDs.
- Added the compact v4 Cayuga saved-output summary after scoring the ignored
  raw run artifact offline; it records only parse-error counts and gate outcome,
  not raw model text, scheduler logs, local cache paths, hidden labels, or
  source task IDs.
- Added finite-candidate saved-prediction readout scripts and submit templates.
  Public CI runs only the no-model dry-run; model candidate scores remain under
  ignored `post_training/runs/` unless a compact summary is intentionally
  curated.
- Added compact Cayuga summaries for train-observed and all-valid candidate
  readouts; they include only aggregate scores, gate accuracies, top-pair
  counts, and rank summaries, not raw candidate-score tables or scheduler logs.
- Added a saved-candidate fail-closed gate analyzer for ignored candidate-score
  JSONL. Public outputs are compact threshold summaries and do not copy prompts,
  raw model text, scheduler logs, model state, or full candidate-score tables.
- Added compact saved-candidate gate diagnostics for train-observed and
  all-valid policies. They report threshold aggregates, case IDs, and compact
  top-candidate summaries only; raw run folders remain ignored.
- Updated the saved-prediction readiness report to consume compact
  saved-candidate gate diagnostics without reading raw run folders.
- Added a saved-output next-decision checkpoint that reads only compact public
  readiness/gate summaries and records no raw saved predictions, candidate-score
  JSONL, scheduler logs, model state, or ignored run directories.
- Added saved-output calibration probe artifacts derived from the
  next-decision checkpoint. They contain split-safe prompt/output pairs and
  compact evaluator scores only; raw saved predictions, full candidate-score
  tables, scheduler logs, model state, and ignored run directories remain
  uncommitted.
- Added saved-output calibration probe readout runner and cluster templates.
  Public CI uses dry-run mode only; full model score readouts stay under
  ignored `post_training/runs/` unless a compact summary is intentionally
  curated.
- Recorded the Cayuga saved-output calibration probe readout as compact
  JSON/Markdown only; raw readout JSONL, scheduler logs, model state, and raw
  prompt/model text remain uncommitted.
- Added saved-output calibration margin SFT runner and cluster templates.
  Public CI uses dry-run mode only; full base/train/held-out margin JSONL,
  trainable state, and scheduler logs stay under ignored `post_training/runs/`
  unless a compact summary is intentionally curated.
- Recorded the Cayuga saved-output calibration margin SFT result as compact
  JSON/Markdown only; raw margin JSONL, full reports, trainable state, and
  scheduler logs remain uncommitted.
- Recorded the focused saved-output calibration margin SFT follow-up as compact
  JSON/Markdown only; raw focused margin JSONL, full reports, trainable state,
  and scheduler logs remain uncommitted.
- Added saved-output target-format diagnostics to the margin SFT runner and
  cluster templates; public CI exercises dry-run mode only, and full model
  target-format outputs remain ignored under `post_training/runs/`.
- Recorded the saved-output target-format flag diagnostic as compact
  JSON/Markdown only; raw target-format margin JSONL, full reports, trainable
  state, and scheduler logs remain uncommitted.
- Added same-model multi-target-format scoring to the saved-output margin SFT
  runner; extra reports stay under ignored run folders unless compact summaries
  are intentionally curated.
- Recorded the same-model saved-output target-format result as compact
  JSON/Markdown only; raw margin JSONL, full reports, trainable state, and
  scheduler logs remain uncommitted.
- Added optional finite-candidate rank scoring to the saved-output margin SFT
  runner; candidate-score JSONL stays under ignored run folders unless a compact
  summary is intentionally curated.
- Recorded the saved-output candidate-rank result as compact JSON/Markdown
  only; raw candidate-score JSONL, full reports, trainable state, and scheduler
  logs remain uncommitted.
- Added a saved-output candidate field-rank analyzer; it reads ignored raw
  candidate-score JSONL and emits compact action/status rank summaries only.
- Recorded the saved-output candidate field diagnostic as compact JSON/Markdown
  only; raw candidate-score JSONL and full field reports remain uncommitted.
- Added routing gate arbitration over compact public gate reports; it does not
  require raw candidate JSONL or cluster run directories.

## Current Public-Release Rule

Do not publish private databases, raw generated run folders, API keys, local machine paths, allocation/account
identifiers, or non-project planning folders. Use the README, research summary, model card, dataset card,
release checklist, Stage A benchmark artifacts, and release manifest as the public surface.

Stage A artifacts are publish candidates because they are small JSONL/JSON
benchmark artifacts with no local filesystem paths, no API keys, no raw SQLite
database payload, and validator-checked separation between model-visible prompts
and hidden evaluator metadata. This includes the strict-contract compact JSON
targets, observed-collapse preference pairs, evidence-conditioned component
targets, routing action/status contrast pairs, and routing defer-vs-verify
boundary contrast pairs, plus the saved-output calibration probe pairs. Compact
runtime gate results and dry-run margin-SFT plans are publish candidates only as
benchmark/verifier baselines.
They should still be presented as
benchmark substrate, not as clinical evidence or live model behavior.

Publish only the validated clean-snapshot history. Live repository visibility
and hosted CI state are verified separately from this file-backed release gate.

## Residual Placeholders

Some internal design docs intentionally retain placeholders such as `<local-workspace>/...` to preserve project
provenance without exposing a real filesystem path. These placeholders are safe for review, but the cleanest public
release omits older session-recovery documents from the curated surface.

## Automated Check

Run this before changing a public branch, tagging a release, or preparing any
Hugging Face upload:

```bash
python scripts/refresh_public_manifest.py --date YYYY-MM-DD
python scripts/check_public_release.py
python scripts/check_public_git_history.py
```

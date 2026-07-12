# Release Checklist

Current status: the benchmark-first clean-snapshot release surface is validated
for `jang1563/LLM_SFM_tool_deployment_public`. Live GitHub visibility is checked
separately. Project-specific Hugging Face repositories have not been published.

Important: publish only the clean-snapshot history represented by this release
surface. Earlier private development history is not part of the release.

## GitHub Readiness

- [x] Public-facing README with problem, method, results, limitations, and quickstart.
- [x] Benchmark card with action space, evaluator gates, baselines, and limitations.
- [x] Reproducibility card with public-safe and full local validation paths.
- [x] Reviewer guide with 5-minute, 15-minute, and deeper audit paths.
- [x] Roadmap and changelog describe current public status and next milestones.
- [x] Long-term research execution plan anchors Stage A -> post-training -> C5.
- [x] Concise research summary with measured results and claim boundaries.
- [x] Requirements file for lightweight setup.
- [x] Citation metadata draft.
- [x] Conservative license placeholder.
- [x] De-leaked A2 claims corrected away from full-parity overclaiming.
- [x] Public-safe synthetic demo that runs without private DB/API dependencies.
- [x] Public-safe Stage A trajectory benchmark manifest and post-training
  artifacts with checksums in `release/public_release_manifest.json`.
- [x] Deterministic Stage A train/held-out split with no case, split-group, or
  source-task overlap.
- [x] No-API Stage A SFT smoke/eval harness with split-aware gate metrics.
- [x] Artifact-first Stage A saved-prediction producer with no-API smoke mode.
- [x] Offline Stage A prediction-output scorer for saved API/local-SFT/prompt
  prediction JSONL.
- [x] Stage A strict-contract SFT/preference/process artifacts for the saved
  prediction JSON contract.
- [x] Cayuga/Expanse strict-contract SFT smoke runbook with public-safe dry-run.
- [x] Stage A evidence-conditioned component targets, constrained routing
  readout, and compact routing candidate-rank analyzer.
- [x] Stage A routing action/status contrast artifacts for the unresolved
  constrained-routing failure families, with train/held-out overlap checks.
- [x] Stage A routing contrast SFT/margin dry-run and Cayuga/Expanse templates
  for the next targeted routing repair experiment.
- [x] Compact Cayuga routing contrast SFT/margin result recorded without raw
  cluster logs, model states, or run directories.
- [x] Routing contrast finite-candidate rank instrumentation added to the
  runner and public dry-run, with raw candidate reports kept under ignored
  run artifacts.
- [x] Compact Cayuga routing contrast candidate-rank result recorded without
  raw candidate JSONL, model states, or scheduler logs.
- [x] Stage A routing defer-vs-verify boundary contrast artifacts added for
  the remaining insufficient-evidence routing confusion, with 8 train and 2
  held-out pairs.
- [x] Compact Cayuga defer-vs-verify routing result recorded without raw JSONL,
  trainable state, or scheduler logs.
- [x] Compact routing fail-closed gate diagnostic recorded without raw
  candidate JSONL, model state, or scheduler logs.
- [x] No-model routing evidence-boundary gate recorded and added to public QA.
- [x] All-family no-model routing evidence gate recorded and added to public QA.
- [x] Routing gate baseline comparison recorded and added to public QA.
- [x] Routing model-readiness gate recorded and added to public QA.
- [x] Full-trajectory arbitration scaffold recorded and added to public QA.
- [x] Saved-prediction readiness gate recorded and added to public QA.
- [x] Stage A v3 tool-trace prompt contract added without hidden-label leakage.
- [x] Compact v3 Cayuga saved-output failure recorded without raw predictions
  or scheduler logs, and v4 canonical JSON contract added for the next smoke.
- [x] Compact v4 Cayuga saved-output failure recorded without raw predictions
  or scheduler logs; prompt-only repair remains below readiness gates.
- [x] Finite-candidate saved-prediction readout path added with public-safe
  dry-run and Cayuga/Expanse submit templates.
- [x] Compact finite-candidate Cayuga readout results recorded without raw
  candidate-score tables, raw predictions, or scheduler logs.
- [x] Public-safe saved-candidate gate analyzer added for post-hoc score-gap
  fail-closed diagnostics over ignored raw candidate-score JSONL.
- [x] Compact saved-candidate gate results recorded without raw prompts, raw
  model text, scheduler logs, model state, or full candidate-score tables.
- [x] Saved-prediction readiness report now includes compact saved-candidate
  gate diagnostics without reading raw run folders.
- [x] Saved-output next-decision checkpoint added and registered in public QA,
  selecting targeted action/status calibration before any escalation.
- [x] Saved-output calibration probe artifacts added and registered in public
  QA, with 16 train-allowed and 4 held-out evaluation-only pairs and no
  train/held-out source overlap.
- [x] Saved-output calibration probe readout runner and Cayuga/Expanse
  templates added, with public dry-run validation and ignored full-model
  outputs.
- [x] Compact Cayuga saved-output calibration probe readout result recorded
  without raw readout JSONL, scheduler logs, model state, or raw model text.
- [x] Saved-output calibration margin SFT dry-run and Cayuga/Expanse templates
  added, with train-only split validation and ignored full-model outputs.
- [x] Compact Cayuga saved-output calibration margin SFT result recorded
  without raw margin JSONL, full reports, trainable state, or scheduler logs.
- [x] Compact focused Cayuga saved-output calibration margin SFT result
  recorded without raw focused margin JSONL, full reports, trainable state, or
  scheduler logs.
- [x] Saved-output target-format diagnostic dry-run added for action/status
  isolation before any DPO/RLVR or `tool_query` escalation.
- [x] Compact saved-output target-format flag diagnostic result recorded
  without raw target-format margin JSONL, full reports, trainable state, or
  scheduler logs.
- [x] Same-model saved-output target-format scoring path added with public
  dry-run coverage and ignored full-model outputs.
- [x] Compact same-model saved-output target-format result recorded without raw
  margin JSONL, full reports, trainable state, or scheduler logs.
- [x] Saved-output margin SFT runner can score optional finite-candidate
  held-out ranks while keeping candidate-score JSONL under ignored run folders.
- [x] Compact saved-output candidate-rank result recorded without raw
  candidate-score JSONL, full reports, trainable state, or scheduler logs.
- [x] Saved-output candidate field-rank analyzer added with public-safe tests.
- [x] Compact saved-output candidate field diagnostic recorded without raw
  candidate-score JSONL or full field reports.
- [x] Public-safe routing gate arbitration report recorded and added to public QA.
- [x] No-model component saved-prediction baselines for deterministic
  oracle/collapse/citation/tool-shape gates.
- [x] Public-safe architecture and results snapshot files.
- [x] Machine-readable public-release manifest and automated safety checker.
- [x] Public-release checker scans GitHub/OpenAI/Hugging Face token patterns,
  private key markers, local paths, and private infrastructure breadcrumbs.
- [x] Git-history preflight script exists for deciding whether the existing
  private repo can be made public in place.
- [x] `python scripts/check_public_git_history.py` passes in this mirror.
- [x] Compact Cayuga evidence candidate-routing full-smoke result recorded
  without raw candidate scores, model state, scheduler logs, or run folders.
- [x] Sealed Stage A extension commitment records aggregate balance, hashes,
  and zero public-source overlap without publishing private rows or labels.
- [ ] Decide final license: all-rights-reserved, MIT, Apache-2.0, or lab-approved release.
- [x] Build a clean-snapshot release surface from the validated current tree.
- [ ] Verify live public visibility for `jang1563/LLM_SFM_tool_deployment_public`.
- [ ] Verify the public-safe GitHub Actions QA workflow on the released commit.
- [x] Add pull-request checklist for benchmark leakage and public-surface hygiene.
- [ ] Publish a GitHub release tag once cards and public data policy are final.

## Hugging Face Readiness

Recommended layout:

- Dataset repo: `jang1563/llm-sfm-stage-a-trajectories`
- Dataset repo: `jang1563/llm-sfm-tool-deployment-a2-band`
- Model repo: `jang1563/a2-mondo-band-reranker-qwen2p5-1p5b-lora`
- Optional Space: `jang1563/llm-sfm-negative-evidence-demo`

Before upload:

- [x] Draft model card exists: `MODEL_CARD.md`.
- [x] Draft dataset card exists: `DATASET_CARD.md`.
- [x] Stage A JSONL artifacts, including strict-contract, evidence-conditioned,
  routing contrast, defer-vs-verify boundary, and saved-output calibration
  probe artifacts, are listed with record counts and checksums. Margin-SFT raw
  run outputs are excluded unless compact summaries are curated later.
- [ ] Confirm that every uploaded JSONL row is free of private DB-only fields.
- [ ] Upload only de-leaked A2 prompts/candidates.
- [ ] Exclude ignored generated arrays, local adapters, raw ontology downloads, logs, and private database files.
- [ ] Add exact evaluation command snippets to the cards after final upload paths are known.
- [ ] Add checksum/manifests for uploaded train/eval files.
- [ ] Re-run `python scripts/check_public_release.py` immediately before any upload.

## Research Communication

- [x] Honest headline metrics are visible in the README.
- [x] Negative findings are framed as evidence of scientific rigor, not as failure.
- [x] The project story is benchmark-first and bounded by measured evidence.
- [x] Public demo command is visible and safe to run.
- [ ] Add a 60-90 second demo video or GIF after a small public demo is available.
- [x] Add a diagram showing the flow: free text -> resolver -> tool query -> trajectory decision -> validator.

## Do Not Publish Yet

- Private NegBioDB SQLite database paths or full DB files.
- Raw API keys, model-cache paths, account/allocation identifiers, or cluster logs.
- Leaked-era A2 result language claiming Qwen/open models fully match the proprietary reference.

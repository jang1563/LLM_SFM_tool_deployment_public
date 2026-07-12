# Changelog

All notable public-facing changes are tracked here.

## 0.1.0 - 2026-07-12

- Curate a clean-snapshot release surface: remove career-planning and session
  recovery documents, replace the former one-page summary with a research
  summary, and correct release identity metadata.
- Remove concrete scheduler identifiers from compact cluster summaries while
  preserving aggregate scientific results.
- Add fail-closed checks for career framing, numeric scheduler metadata, and
  internal release-planning filenames.
- Add deterministic manifest checksum and JSONL record-count refresh tooling.
- Add reviewer navigation, roadmap, and release-note scaffolding.
- Add long-term scientific research execution plan and update the active
  research status checkpoint.
- Add no-API Stage A SFT smoke/eval harness and deterministic baseline report.
- Add offline Stage A prediction-output scorer for saved API/local-SFT/prompt
  outputs, with fail-closed missing/extra case handling.
- Add artifact-first Stage A prediction producer with deterministic no-API
  modes and opt-in live API generation.
- Add Stage A strict-contract SFT/preference/process artifacts targeting the
  observed Cayuga strict-prompt failure shape.
- Add Cayuga/Expanse strict-contract SFT smoke runbook with a public-safe
  no-model dry-run.
- Record the first Cayuga strict-contract SFT smoke result as a negative
  trajectory-gate checkpoint.
- Add strict-contract component diagnostics for enum validity, structured tool
  calls, query fields, and evidence/action routing.
- Add strict component target exports for enum/action, tool-query, and
  routing-after-loop train/held-out slices.
- Add strict component SFT smoke runner with no-model dry-run and public-safe
  Cayuga/Expanse submit templates.
- Update the long-term roadmap to a research-first 6-8 week execution plan and
  add a drift checker for Stage A component, DPO/RLVR, C5, and release gates.
- Record the first Cayuga `enum_action` component SFT smoke as a negative
  component-gate checkpoint.
- Add finite-candidate `enum_action` decoding for the next component repair run.
- Record the finite-candidate `enum_action` repair result as a partial fix:
  schema/enum gates pass, exact enum-pair selection remains weak.
- Add enum candidate-rank diagnostics and retain full candidate-score tables in
  ignored raw run artifacts for future compact margin summaries.
- Record the full-rank `enum_action` candidate diagnostic: still 1/5 pass, with
  low gold ranks for insufficient and invalid-value cases.
- Add the train-observed valid-pair counterfactual, showing candidate-space
  pruning alone still collapses to `ground` / `supported`.
- Add Stage A enum corrective contrast pairs against the observed
  `ground` / `supported` collapse, with train/held-out validation.
- Add enum corrective SFT/margin smoke runner and Cayuga/Expanse submit
  templates for the next evidence-conditioned enum diagnostic.
- Record the first Cayuga enum corrective SFT/margin result as partial evidence:
  2/4 held-out contrast wins, with insufficient and invalid-value collapse
  unresolved.
- Add base-vs-trained and train-pair margin diagnostics to the enum corrective
  smoke runner.
- Record Cayuga enum corrective margin-delta result: all held-out families move
  positively, but invalid-value and defer/insufficient remain unresolved.
- Add targeted enum corrective sampling controls for weak invalid-value and
  defer/insufficient pairs.
- Record targeted enum sampling as a negative repair result: mean margin
  improves, but held-out wins drop and invalid-value remains unresolved.
- Add enum field-rank diagnostics showing invalid-value failure is primarily a
  weak `flag` action-representation problem, not only an `invalid_value` status
  problem.
- Add Stage A enum action-contrast pairs that preserve evidence status while
  forcing the rejected action to `ground`.
- Record Cayuga enum action-contrast diagnostic as partial movement, not repair:
  1/4 -> 2/4 held-out wins, with `flag` / `invalid_value` and
  `defer` / `insufficient` still losing to `ground`.
- Add an `action_only` target-format diagnostic mode for enum corrective
  SFT/margin runs, so the next Cayuga smoke can separate action-label learning
  from full action-plus-status JSON coupling.
- Record Cayuga enum action-only diagnostic as negative for target-format-only
  repair: held-out wins stay 1/4 and `flag`, `defer`, and `reject` train
  margins remain below zero.
- Add a supervised pairwise-margin objective hook to the enum corrective runner
  so the next Cayuga smoke can directly push chosen action targets above
  rejected `ground` targets without starting DPO/RLVR.
- Record the Cayuga enum pairwise-margin diagnostic: action-only held-out
  margins improve from 1/4 to 4/4 wins, with full action/status scoring still
  left as the next gate.
- Record the Cayuga enum full pairwise-margin diagnostic: full
  action-plus-status scoring reaches 4/4 held-out wins on both same-status
  action contrasts and `ground` / `supported` corrective contrasts, while
  trajectory-level claims remain gated.
- Add finite-candidate enum selection readouts to the pairwise-margin runner so
  the next Cayuga smoke can compare base vs trained candidate top-1/rank
  without starting DPO/RLVR.
- Record the Cayuga enum candidate-selection readout as a negative/partial
  result: margins repair to 4/4, but 30-way and 5-way candidate top-1 remain
  0/4 while gold ranks improve.
- Add an optional finite-candidate enum cross-entropy objective to the
  pairwise-margin runner, so the next Cayuga smoke can train directly on the
  candidate-selection failure without starting DPO/RLVR.
- Record the Cayuga candidate-CE result as another negative/partial
  `enum_action` result: margins remain 4/4, but 5-way candidate top-1 only
  improves from 0/4 to 1/4.
- Add and run a post-hoc enum candidate gate diagnostic showing current
  score-gap confidence has zero useful zero-false-trust coverage.
- Add a factorized `action` / `evidence_status` candidate-CE mode for the next
  `enum_action` repair run, without changing the DPO/RLVR gate.
- Record the Cayuga field-CE result: margins remain 4/4, candidate top-1 stays
  1/4, and fail-closed trust coverage remains 0/4.
- Add Stage A component visibility audit showing no hidden-label leaks, but
  50/75 strict component rows require evidence routing without model-visible
  evidence content or tool-result payloads.
- Add Stage A evidence-conditioned component targets that expose public-safe
  evidence state to enum/routing slices and close the visibility audit's
  underdetermined evidence-routing count from 50/75 to 0/75.
- Record the first Cayuga evidence-conditioned `enum_action` component result:
  observed-pair scoring remains 1/5 with `ground` / `supported` selected for
  all held-out cases, so evidence visibility alone is not enough.
- Record the first Cayuga evidence-conditioned `routing_after_loop` component
  result: free-form routing remains 0/5 and fails schema/enum gates despite
  visible tool-result content.
- Add constrained evidence-conditioned `routing_after_loop` readout mode:
  `routing_observed_pair_score` scores train-observed action/status pairs and
  fills citations only from prompt-visible tool-result content.
- Add a public-safe constrained-routing candidate analyzer for compact
  full-target rank, action/status rank, and citation-failure summaries.
- Add no-model Stage A component saved-prediction baselines for oracle,
  ground/supported collapse, missing-citation routing, and tool-shape failures.
- Record the Cayuga constrained evidence-conditioned `routing_after_loop` result:
  2/5 held-out pass, schema/enum gates fixed, with remaining action/status
  routing failures in insufficient, verify, and invalid-value families.
- Add Stage A routing action/status contrast pairs for those three constrained
  routing failure families, with train/held-out validation and prompt-leak
  checks.
- Add a routing contrast SFT/margin smoke runner plus Cayuga/Expanse submit
  templates for the next targeted `routing_after_loop` repair experiment.
- Record the Cayuga routing contrast SFT/margin result: teacher-forced held-out
  margins improve from 0/3 to 3/3 wins and mean margin -0.117 -> 0.115, while
  finite-candidate/free-form routing and full trajectory gates remain open.
- Add routing contrast finite-candidate rank instrumentation to the SFT/margin
  runner, so the next Cayuga run can compare base vs trained routing candidate
  top-1/rank before `tool_query`, DPO, or RLVR.
- Record the Cayuga routing candidate-rank result as partial transfer:
  finite-candidate exact top-1 improves from 0/3 to 2/3, with the remaining
  failure at the `defer` / `insufficient` versus `verify` / `insufficient`
  boundary.
- Add Stage A routing defer-vs-verify boundary pairs: 10 public-safe contrast
  pairs, 8 train and 2 held-out, targeting the remaining insufficient-evidence
  routing confusion without broad retraining.
- Record the Cayuga defer-vs-verify routing smoke as negative/partial:
  finite-candidate exact top-1 stays 1/2 and the unresolved
  `defer` / `insufficient` case still loses to `verify` / `insufficient`.
- Add and run a routing candidate fail-closed gate diagnostic: threshold 0.025
  has 0 unsafe trusted rows and 2/2 strict final correctness on the tiny
  defer-vs-verify held-out slice.
- Add and run a no-model routing evidence-boundary gate using only
  prompt-visible tool-result fields; it reaches 10/10 overall and 2/2 held-out
  on the defer-vs-verify boundary.
- Add and run an all-family no-model routing evidence gate using only
  prompt-visible tool-result fields; it reaches 25/25 overall and 5/5 held-out
  across all Stage A `routing_after_loop` rows.
- Add and run a routing gate baseline-comparison report: runtime/oracle are
  25/25, `ground`/`supported` collapse is 5/25 with 20 unsafe overrides, and
  citationless routing is 15/25 with 10 citation mismatches.
- Add and run a routing model-readiness report over compact Cayuga summaries:
  best all-family model readout is 2/5, below citationless routing at 3/5 and
  runtime gate at 5/5, so escalation remains gated.
- Add and run a full-trajectory arbitration report through the canonical Stage
  A evaluator: runtime and hybrid policies are 25/25, collapse is 5/25 with 20
  unsafe overrides, and citationless routing is 15/25 with attribution failures.
- Add and run a saved-prediction readiness report over compact public Cayuga
  summaries: the best real saved output remains 0/5 held-out, below collapse,
  citationless, and runtime full-trajectory baselines.
- Add a Stage A `stage_a_v3_tool_trace` prompt contract for the next saved
  prediction smoke: it requires an ordered four-tool trace with `drug_id` and
  `condition_id` arguments while keeping hidden labels and source IDs out of the
  prompt.
- Record the v3 Cayuga saved-output smoke as an enum/schema failure: Qwen2.5-0.5B
  scores 0/5 with invalid `evidence_status: verified` in all held-out rows, and
  add `stage_a_v4_canonical_json` for the next public-safe prompt-contract
  smoke.
- Record the v4 Cayuga saved-output smoke: invalid `verified` status disappears,
  but Qwen2.5-0.5B still scores 0/5 because the canonical top-level
  action/JSON envelope is missing or malformed.
- Add a finite-candidate saved-prediction readout path with public-safe dry-run
  and Cayuga/Expanse submit templates, so the next saved-output smoke can test
  constrained action/status selection without free-form JSON envelope failure.
- Record Cayuga finite-candidate saved-output readouts: parse/tool/query gates
  pass, but both train-observed and all-valid candidate spaces collapse to
  `ground` / `supported` in 5/5 held-out cases.
- Add a public-safe saved-candidate gate analyzer for ignored raw candidate
  readout JSONL, so score-gap fail-closed thresholds can be reported without
  publishing prompts, raw model text, scheduler logs, or full candidate-score
  tables.
- Record compact Cayuga saved-candidate gate diagnostics for train-observed and
  all-valid candidate policies: both trust 1/5 supported row with 0 unsafe
  trust and reach 2/5 strict final correctness after fail-closed routing.
- Wire saved-candidate fail-closed gate diagnostics into the saved-prediction
  readiness report, so escalation remains blocked on both raw saved-output
  performance and gated strict-final correctness.
- Add and run a saved-output next-decision checkpoint that selects a targeted
  action/status calibration probe from compact readiness/gate artifacts and
  keeps `tool_query`, DPO/RLVR, HF publication, release tagging, and broad
  retraining gated.
- Add the saved-output calibration probe artifacts: 20 target-vs-`ground` /
  `supported` pairs, split into 16 train-allowed and 4 held-out
  evaluation-only rows with no case, split-group, or source-task overlap.
- Add a saved-output calibration probe readout runner and Cayuga/Expanse
  templates, with public dry-run coverage and train-selected held-out gate
  reporting before any readiness escalation.
- Record the Cayuga saved-output calibration probe readout as a negative
  result: Qwen2.5-0.5B scores `ground` / `supported` above target outputs on
  all 20 probe rows.
- Add a saved-output calibration margin SFT runner with public dry-run,
  train-only split validation, held-out margin/delta reporting, and
  Cayuga/Expanse submit templates for the next narrow collapse-repair
  diagnostic.
- Record the Cayuga saved-output calibration margin SFT result as
  partial-negative: held-out target-vs-collapse wins improve from 0/4 to 1/4,
  but mean margin remains below zero.
- Record the focused Cayuga saved-output calibration margin SFT follow-up:
  non-verify family oversampling improves held-out wins to 3/4, but
  `flag` / `invalid_value` remains unresolved.
- Add saved-output target-format diagnostics for `full`, `action_status_only`,
  `action_only`, and `status_only` margin SFT scoring.
- Record the Cayuga saved-output target-format result: isolated action/status
  targets repair `flag` / `invalid_value`, while full JSON remains unresolved.
- Add same-model multi-target-format scoring for saved-output margin SFT.
- Record the Cayuga same-model target-format result: full-target
  `flag`-focused SFT repairs the teacher-forced full JSON held-out margin, while
  finite-candidate, free-generation, and full-trajectory gates remain closed.
- Add finite-candidate rank scoring to the saved-output margin SFT runner so the
  next Cayuga diagnostic can test whether teacher-forced repair survives
  candidate selection before any `tool_query`, DPO/RLVR, or release escalation.
- Record the Cayuga saved-output candidate-rank result as negative transfer:
  teacher-forced full margins stay repaired, but 5-way candidate top-1 is 1/4
  and the trained model over-selects `flag` / `invalid_value`.
- Add a saved-output candidate field-rank analyzer for compact action/status
  bias diagnostics over ignored candidate-score JSONL artifacts.
- Record the Cayuga saved-output candidate field diagnostic: non-flag held-out
  rows fail both action and status fields while `flag` / `invalid_value` is the
  only pair top-1.
- Add train-side candidate scoring to the saved-output margin SFT runner so the
  next Cayuga diagnostic can estimate calibration bias from train rows before
  applying any held-out candidate interpretation.
- Record the Cayuga saved-output train-candidate diagnostic: trained train rows
  top-rank `flag` / `invalid_value` in 16/16 cases with 4/16 exact top-1, so
  the candidate failure is a global focused-objective prior rather than
  held-out-only noise.
- Add a train-derived saved-output candidate calibration analyzer that applies
  pair-mean centering from train candidate scores without held-out tuning.
- Record the Cayuga calibration diagnostic: held-out candidate top-1 improves
  from 1/4 to 2/4, but the `flag` / `invalid_value` row is lost to
  `ground` / `supported`, so this is not trust-readiness evidence.
- Add a train-selected zero-unsafe gate readout to the calibration diagnostic;
  it trusts 0/4 held-out rows and reaches only 1/4 strict final with static
  `defer` / `insufficient` fail-closed routing.
- Add a saved-output candidate arbitration report over compact calibration
  results and tracked model-visible evidence targets. Runtime evidence/hybrid
  policies are 4/4 on the held-out calibration slice, while raw candidate top-1
  is 1/4, calibrated top-1 is 2/4, and score-gap gating is 1/4.
- Update the saved-output next-decision checkpoint so the targeted calibration
  probe is now closed out: future model-heavy checkpoints must meet or beat
  runtime evidence/hybrid arbitration at 4/4 with zero unsafe candidate trust
  before `tool_query`, DPO/RLVR, HF publication, release tagging, or broad
  retraining reopens.
- Add a saved-output meet-or-beat gate that turns the next-decision checkpoint
  into a reusable public-safe acceptance check. The current raw, calibrated,
  and score-gap candidate policies all fail against the 4/4 runtime
  evidence/hybrid baseline, while raw run artifacts remain uncommitted.
- Extend the meet-or-beat gate with a repeatable `--model-policy-summary`
  compact JSON input so future Cayuga summaries can be checked without reading
  raw predictions, candidate-score JSONL, scheduler logs, or model state.
- Add a public-safe saved-output policy-summary adapter that converts compact
  saved-output summaries, candidate-gate summaries, or arbitration policy rows
  into the meet-or-beat gate input contract.
- Make the meet-or-beat gate fail closed when external compact policy summaries
  indicate committed raw predictions, candidate-score JSONL, raw eval reports,
  scheduler logs, or model state.
- Add a saved-output intake contract report that verifies compact bundle hashes,
  next-decision criteria, meet-or-beat inputs, and future-policy public-safe
  flags before accepting a new Cayuga policy summary.
- Tighten the meet-or-beat gate so external compact policy summaries fail
  closed on impossible or non-comparable counts, including negative counts,
  `exact > rows`, trust counts above rows, and row counts that do not match the
  runtime baseline slice.
- Tighten the saved-output policy-summary adapter and meet-or-beat gate so
  count fields must be JSON integers, not strings, floats, or booleans. This
  prevents future compact Cayuga summaries from passing through implicit
  `int(...)` coercion.
- Require external `--model-policy-summary` inputs to carry the adapter
  provenance contract: dataset, source kind, readable source report,
  source-report SHA-256, explicit public-safety contract, and all raw-artifact
  commit flags. Hand-written score-only JSON no longer satisfies the gate.
- Extend Public QA so the saved-output policy-summary adapter smoke is passed
  back through the meet-or-beat gate. The CI path now fails if adapter output
  breaks provenance validation instead of merely writing a JSON file.
- Tighten saved-output policy summary source provenance: `source_report` must
  be repo-relative, listed in `release/public_release_manifest.json`, marked
  `safe_to_publish`, and match the manifest SHA-256. Runtime-local `/tmp`
  source reports no longer satisfy the public gate.
- Add and run a public-safe routing gate arbitration report comparing raw
  candidate top-1, score-gap fail-closed, evidence-boundary override, and
  hybrid policies on the defer-vs-verify held-out slice.
- Make torch-dependent post-training helpers import torch lazily so public-safe
  tests can collect without the optional model-training dependency installed.
- Keep license, Hugging Face upload, and release tag decisions open.

- Record the approved evidence-conditioned candidate-routing Cayuga smoke as a
  negative result: 4/20 train, 1/5 held-out, and 1/4 bridge-focus exact, with
  all rows routed to `verify` / `insufficient`; freeze the reused slice as
  diagnostic data before building a new sealed evaluation extension.
- Add a fail-closed sealed-extension builder that keeps private candidate and
  manifest rows outside the repository, excludes all declared public task
  sources and claims, and emits only aggregate commitments.
- Record the 25-row sealed extension commitment with five balanced action
  families and zero source-task, split-group, or normalized-claim overlap
  against declared public exclusions.

## 0.1.0 - Public Benchmark Snapshot

Initial public GitHub snapshot.

Highlights:

- Stage A public-safe manifest with 25 benchmark cases.
- Deterministic trajectory evaluator for tool sequence, evidence status,
  attribution, terminal action, and policy compliance.
- Stage A post-training exports:
  - 25 oracle SFT rows;
  - 150 preference pairs;
  - 25 process-supervision rows;
  - deterministic 20/5 train-held-out split.
- Synthetic public demo with passing and failing trajectory examples.
- Public release manifest with artifact counts and SHA-256 checksums.
- Public QA workflow for release checks, git-history checks, post-training
  validation, demo execution, and public-safe tests.
- Public-release checker for common token, local-path, private-infrastructure,
  generated-cache, database, key-file, and log-file leakage.

Limitations:

- The public Stage A manifest is a compact benchmark substrate, not a broad
  clinical benchmark.
- Larger NegBioDB-CT experiments may depend on private database material not
  included in this mirror.
- Hugging Face uploads and final open-source license selection are not yet
  complete.

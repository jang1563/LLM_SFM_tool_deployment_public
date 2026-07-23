# Roadmap

This roadmap keeps the public project aligned with the benchmark-first thesis:
biology agents need trainable tool-use trajectories plus runtime enforcement.

## Current Public State

Status: public benchmark substrate.

Completed:

- clean public GitHub mirror with fresh history;
- public-safe Stage A manifest with hidden evaluator metadata separated from
  model-visible task text;
- deterministic trajectory evaluator and Stage A adapter;
- oracle SFT, preference, process-supervision, and train/held-out split
  artifacts;
- strict-contract compact JSON SFT/preference/process targets for the saved
  prediction output contract;
- Cayuga/Expanse strict-contract SFT smoke runner and sbatch templates;
- strict component targets, diagnostics, and component-slice SFT smoke runner for
  `enum_action`, `tool_query`, and `routing_after_loop`;
- compact Cayuga `enum_action` component SFT smoke result;
- finite-candidate `enum_action` repair result that fixes schema/enum validity
  but leaves enum-pair selection weak;
- enum candidate-rank diagnostic tooling plus a full 30-candidate Cayuga rerun
  showing gold ranks 1, 5, 13, 4, and 24;
- observed-pair counterfactual showing train-observed valid pair pruning still
  selects `ground` / `supported` for all held-out cases;
- enum corrective contrast pairs that reject the observed `ground` / `supported`
  collapse while preserving the Stage A train/held-out split;
- enum corrective SFT/margin smoke runner and Cayuga/Expanse templates for the
  next cluster diagnostic;
- compact Cayuga enum corrective SFT/margin smoke result: 2/4 held-out contrast
  wins, with insufficient and invalid-value cases still losing to
  `ground` / `supported`;
- enum corrective base-vs-trained margin delta hooks, so the next Cayuga run can
  separate actual corrective movement from final held-out margin alone;
- compact Cayuga margin-delta result showing positive movement in all held-out
  families, but invalid-value still failing even on train-pair margins;
- targeted enum oversampling hooks for the weak `flag/invalid_value` and
  `defer/insufficient` train pairs;
- compact Cayuga targeted-sampling result showing mean margin improvement but
  worse held-out wins and persistent invalid-value train failure;
- same-status action-contrast pairs and a Cayuga diagnostic showing useful
  action-field movement, but no stable repair: held-out wins improve from 1/4
  base to 2/4 trained while `flag` / `invalid_value` and
  `defer` / `insufficient` remain below zero;
- an `action_only` target-format diagnostic hook for the enum corrective runner,
  so the next Cayuga smoke can test whether the weak families fail even when
  `evidence_status` is removed from the output target;
- compact Cayuga action-only result showing target-format-only repair is
  insufficient: held-out wins stay 1/4 and `flag`, `defer`, and `reject` train
  margins remain below zero;
- a supervised pairwise-margin hook for enum corrective SFT, which can test
  whether explicitly pushing chosen actions above `ground` repairs the weak
  action families before any preference/RLVR escalation;
- compact Cayuga pairwise-margin result showing action-only held-out margins
  improve from 1/4 to 4/4 wins, while full action/status enum scoring remains
  the next gate;
- compact Cayuga full pairwise-margin result showing full action/status
  held-out margins improve to 4/4 wins on both same-status action contrasts and
  `ground` / `supported` corrective contrasts;
- finite-candidate enum selection readouts in the pairwise-margin runner, so
  the next Cayuga smoke can compare base vs trained candidate top-1/rank before
  leaving `enum_action`;
- compact Cayuga enum candidate-selection readout showing pairwise-margin SFT
  improves finite-candidate rank but leaves both 30-way and 5-way top-1 at
  0/4;
- optional finite-candidate enum cross-entropy objective in the pairwise-margin
  runner, with no-model validation and Cayuga/Expanse submit hooks;
- compact Cayuga candidate-CE result showing margin repair remains 4/4, but
  finite-candidate top-1 only moves from 0/4 to 1/4;
- post-hoc candidate gate diagnostic showing score-gap confidence cannot trust
  any held-out enum candidate without risking false trust;
- factorized `action` / `evidence_status` candidate-CE objective support for
  the next `enum_action` repair probe;
- compact Cayuga field-CE result showing the factorized loss does not improve
  beyond 1/4 candidate top-1 or 0/4 useful trust coverage;
- Stage A component visibility audit showing hidden-label leaks are 0/75, but
  all current `enum_action` and `routing_after_loop` evidence-routing rows lack
  model-visible evidence or tool-result content;
- Stage A evidence-conditioned component targets exposing public-safe evidence
  state to `enum_action` and tool-result payloads to `routing_after_loop`, with
  the visibility audit closing underdetermined routing rows from 50/75 to 0/75;
- no-API Stage A smoke/eval harness, saved-prediction producer, and offline
  prediction-output scorer;
- Stage A v3 tool-trace Cayuga saved-output smoke showing a clean enum/schema
  blocker: Qwen2.5-0.5B emits invalid `evidence_status: verified` in all 5
  held-out rows;
- Stage A v4 canonical JSON prompt contract for the next saved-output smoke,
  explicitly blocking `verified`/`valid`-style non-enum statuses without
  exposing hidden labels;
- Stage A v4 Cayuga saved-output smoke showing prompt-only repair is still
  insufficient: invalid `verified` status disappears, but canonical action/JSON
  envelope generation still fails 5/5;
- Stage A saved-prediction finite-candidate readout path with a no-model
  dry-run baseline: constrained action/status/tool/query construction scores
  3/5 held-out and mean 0.943, failing only hidden-source attribution cases;
- compact Cayuga finite-candidate saved-output result showing parse/tool/query
  gates pass but Qwen2.5-0.5B selects `ground` / `supported` in 5/5 cases;
- public-safe saved-candidate gate analyzer for ignored candidate-score JSONL,
  so the next compact Cayuga checkpoint can test zero-unsafe fail-closed
  coverage without publishing raw prompts, model text, scheduler logs, or full
  score tables;
- compact Cayuga saved-candidate gate results showing score-gap thresholds can
  trust only the supported row in 1/5 cases with 0 unsafe trust, reaching 2/5
  strict final correctness after fail-closed routing;
- saved-output calibration probe artifacts and a Cayuga readout showing
  Qwen2.5-0.5B still scores `ground` / `supported` above the target output in
  all 20 target-vs-collapse probe rows;
- saved-output calibration margin SFT runner plus Cayuga/Expanse templates, so
  the next corrective experiment can train on 16 train-only probe pairs and
  evaluate 4 held-out probe pairs without using local heavy compute;
- compact Cayuga saved-output calibration margin SFT result showing partial
  movement but no repair: held-out target-vs-collapse wins move from 0/4 to
  1/4 and mean margin remains below zero;
- compact focused saved-output calibration margin SFT result showing stronger
  but still incomplete repair: held-out wins move to 3/4, while
  `flag` / `invalid_value` remains below `ground` / `supported`;
- saved-output target-format diagnostics in the margin SFT runner, so the
  unresolved `flag` / `invalid_value` case can be split into action-only,
  status-only, action+status, and full-JSON scoring surfaces;
- compact saved-output target-format result showing isolated action/status
  targets repair `flag` / `invalid_value`, while the full JSON target remains
  below `ground` / `supported`;
- same-model target-format scoring for saved-output margin SFT, so a full
  trained model can be scored across full/action/status projections without
  comparing separate projection-specific training runs;
- compact same-model target-format result showing a first teacher-forced
  full-target repair signal: held-out full JSON wins move 0/4 -> 4/4 and the
  `flag` / `invalid_value` margin crosses from -0.175 to +0.026, while
  candidate-ranking and trajectory gates remain closed;
- finite-candidate rank scoring attached to the saved-output margin SFT runner,
  so the next Cayuga diagnostic can test whether the repaired full target beats
  train-observed target pairs plus `ground` / `supported` collapse;
- compact candidate-rank result showing negative transfer: full teacher-forced
  margins remain 4/4, but 5-way candidate top-1 is 1/4 and the trained model
  over-selects `flag` / `invalid_value` in 4/4 held-out rows;
- saved-output candidate field-rank analyzer for diagnosing whether candidate
  failures are action-field, evidence-status-field, or joint pair-selection
  failures without publishing raw candidate-score JSONL;
- compact candidate field result showing the trained non-flag rows fail both
  action and evidence-status fields, while only the `flag` / `invalid_value`
  target is pair top-1;
- public demo with synthetic trajectory cases;
- public release manifest with record counts and checksums;
- public QA workflow and local release/history validators.
- compact Cayuga `tool_query` placeholder-schema result: 0/5 held-out pass,
  with no generated `tool_calls`;
- hashed candidate-routing policy freeze using the existing saved trainable
  state and no sealed-set retraining;
- one-time source-separated sealed candidate-routing result: 5/25 exact,
  collapsing to `verify` / `insufficient` on all 25 rows against a 25/25
  deterministic runtime oracle;
- private sealed-manifest and one-time-lock workflow with aggregate-only public
  results and no row-level labels or candidate scores.

Not completed:

- final open-source license decision;
- Hugging Face dataset/model/Space publication;
- a future release beyond the existing v0.1.0 reproducibility snapshot;
- public demo video or GIF;
- prospective real-query component data with actual model-visible identifier
  values or an explicit entity-resolution interface;
- runtime-hybrid perturbation evaluation over attribution, source, value,
  contradiction, partial-query, wrong-tool, and unavailable-tool failures;
- Stage B C5 antibody-antigen OOD transfer package.

## Near-Term Milestones

The detailed scientific execution plan is
`research/2026-06-25_posttrain_tool_use_landscape/LONG_TERM_RESEARCH_PLAN_2026-07-04.md`.

### 1. Stage A Component Smoke Results

Goal: measure the three strict component slices before escalating to DPO/RLVR.

Exit criteria:

- Cayuga runs are attempted in order: `enum_action`, `tool_query`, then
  `routing_after_loop`;
- Expanse is used only as fallback or replication;
- each compact summary reports held-out pass rate, mean score, and violation
  counts;
- raw `post_training/runs/` outputs remain uncommitted;
- `STATUS.md` records thesis, result, source changes, and next decision.
- current `enum_action` result is negative: fix constrained enum output or target
  format before broad retraining.
- candidate scoring fixes the enum contract but still needs ranking/calibration
  diagnosis before moving to `tool_query`;
- full-rank `enum_action` rerun should be treated as negative method evidence:
  insufficient and invalid-value cases are low-ranked, not simple near misses.
- observed-pair pruning is also negative: the current bottleneck is
  evidence-conditioned enum routing, not candidate-space size.
- enum corrective pair artifacts exist, but they are not yet a post-training
  result;
- enum corrective SFT/margin dry-run exists; the next Cayuga result should report
  held-out contrast wins over 4 held-out corrective pairs before changing method.
- first Cayuga corrective result is partial, not a stable repair: 2/4 held-out
  contrast wins and mean margin below zero.
- action-contrast SFT also remains partial: base 1/4 -> trained 2/4 held-out
  wins, but the weak `flag` and `defer` action families still lose to
  `ground`.
- the next enum diagnostic should run `TARGET_FORMAT=action_only` before
  changing component slices, so target-format coupling is measured rather than
  assumed.
- action-only target format does not repair the slice; next work should add
  targeted action rows or a constrained action-head objective before
  `tool_query`, DPO, or RLVR.
- the constrained action-head diagnostic should be run as supervised
  pairwise-margin SFT first; keep DPO/RLVR gated until this deterministic
  objective has a held-out margin report.
- pairwise-margin SFT repairs the action-only margin readout, but does not yet
  prove full JSON enum/action or trajectory repair.
- full-target pairwise-margin SFT repairs two teacher-forced enum margin
  readouts, but still needs finite-candidate/free-generation and trajectory
  checks before method escalation.
- finite-candidate readout support is implemented; the next `enum_action`
  Cayuga run should report base and trained candidate top-1/rank before moving
  to `tool_query`.
- finite-candidate readout is negative for top-1 repair: mean gold rank moves
  in the right direction, but `enum_action` still needs an explicit candidate
  selection objective before `tool_query`.
- candidate-selection objective support is implemented; the next Cayuga smoke
  should test whether candidate CE repairs 5-way or 30-way top-1 before moving
  to `tool_query`.
- candidate CE is still a negative/partial result: 5-way top-1 improves to only
  1/4, so the next `enum_action` gate should test candidate calibration,
  action/status factorization, or constrained candidate routing before
  `tool_query`.
- score-gap gating must currently fail closed: zero-false-trust coverage is
  0/4, which supports runtime enforcement but does not repair the model.
- factorized candidate-CE support is implemented; the next Cayuga smoke should
  test whether field-level pressure reduces the `both_field_failure` pattern.
- field CE does not reduce the held-out bottleneck enough: top-1 remains 1/4
  and the zero-false-trust gate still trusts 0/4 rows.
- component visibility audit shows the current enum/routing targets are
  underconditioned for evidence routing; the next substrate should expose
  evidence-conditioned state before more enum CE, `tool_query`, DPO, or RLVR.
- evidence-conditioned component targets now provide that substrate; the next
  Cayuga smoke should measure `enum_action` and `routing_after_loop` against
  model-visible evidence state before reviving `tool_query`, DPO, RLVR, release
  tagging, or Hugging Face publication.
- evidence-conditioned `enum_action` remains negative/partial: the substrate is
  fixed, but 5-way observed-pair top-1 still collapses to `ground` /
  `supported` in 5/5 held-out cases. Run evidence-conditioned
  `routing_after_loop` next before `tool_query`, DPO, or RLVR.
- evidence-conditioned `routing_after_loop` is also negative in free-form mode:
  0/5 pass, mean 0.200, target-key accuracy 0.0, and enum-validity accuracy
  0.0. Next repair should constrain routing output or split action/status and
  citation selection before `tool_query`, DPO, or RLVR.
- constrained routing readout support exists as `routing_observed_pair_score`:
  it scores train-observed action/status pairs and attaches citations only from
  model-visible tool-result state. Next Cayuga smoke should use this before
  any broad routing retraining or method escalation.
- constrained evidence-conditioned `routing_after_loop` now has a Cayuga result:
  2/5 pass, mean score 0.850, schema/enum gates fixed, and 3 action/status
  target mismatches. This is component progress, but insufficient,
  verification-needed, and invalid-value routing still need targeted repair
  before `tool_query`, DPO, or RLVR.
- routing action/status contrast pairs now isolate those three unresolved
  constrained-routing failures with 12 train and 3 held-out examples, preserving
  source-disjoint split checks and prompt hidden-label isolation.
- a routing contrast SFT/margin smoke runner and Cayuga/Expanse templates now
  validate the 12/3 pair split locally without model load.
- the Cayuga routing contrast SFT/margin result is positive on the
  teacher-forced contrast slice: held-out wins improve from 0/3 to 3/3 and
  mean margin from -0.116919 to 0.114900. This moves the next gate to
  finite-candidate or saved-prediction routing, not DPO/RLVR.
- routing contrast candidate-rank instrumentation is now attached to the same
  runner: dry-run validates the train-observed 5-way routing candidate space,
  and full cluster mode can write base/trained held-out candidate rank reports
  before any `tool_query`, DPO, or RLVR escalation.
- the Cayuga candidate-rank result is partial-positive: exact top-1 improves
  from 0/3 to 2/3 and mean gold rank from 3.0 to 1.333333, but
  `defer` / `insufficient` still loses to `verify` / `insufficient`.

### 2. Stage A Training Diagnosis

Goal: choose the next training change from component-level evidence.

Exit criteria:

- `enum_action` failure maps to constrained decoding or enum target format;
- after constrained decoding, `enum_action` ranking failure maps to candidate
  rank/margin, top-action bias, or slice-specific supervision need;
- current full-rank result points to enum-specific corrective supervision or a
  narrower valid-pair target before `tool_query`;
- field-rank reanalysis points the invalid-value case to weak `flag` action
  representation: pair rank 24, action rank 6, evidence-status rank 2;
- observed-pair counterfactual rules out target-space pruning alone, so the
  next fix should be corrective/contrastive enum supervision;
- corrective data must stay enum-specific and be scored against held-out
  contrast pairs before method escalation;
- current corrective SFT helps contradicted and verification-needed held-out
  pairs, but still fails insufficient and invalid-value pairs;
- the next enum diagnostic should compare base margins, trained train margins,
  and trained held-out margins before adding broader data or moving slices;
- margin-delta result points to targeted invalid-value and defer/insufficient
  enum repair, not DPO/RLVR escalation;
- targeted oversampling must be interpreted as an SFT sampling diagnostic over
  existing train pairs, not new preference optimization;
- targeted oversampling is negative as a repair strategy: invalid-value remains
  0/4 on train-pair margins;
- action-contrast pairs now isolate same-status wrong-action failures, with
  `flag/invalid_value` rejecting `ground/invalid_value` instead of
  `ground/supported`;
- `tool_query` failure maps to structured argument generation and required query
  fields;
- `routing_after_loop` failure maps to evidence/action routing and citation
  grounding;
- current constrained-routing failure maps specifically to insufficient,
  verification-needed, and invalid-value action/status selection; use the new
  routing contrast pairs for a small margin/candidate-routing smoke before
  `tool_query`, DPO, or RLVR;
- routing contrast margin scoring now has a compact positive Cayuga result;
  the next result should test finite-candidate or saved-prediction routing
  before any preference-optimization or RLVR claim;
- routing contrast candidate-rank instrumentation exists; the next Cayuga run
  should enable `SCORE_BASE_ROUTING_CANDIDATES=1` and
  `SCORE_TRAINED_ROUTING_CANDIDATES=1` and interpret candidate top-1/rank
  before changing the training objective;
- routing contrast candidate-rank result points the next repair to the
  defer-vs-verify boundary for insufficient-evidence routing, not broad
  `tool_query`, DPO, or RLVR escalation;
- defer-vs-verify boundary data now exists as 10 routing contrast pairs with
  8 train and 2 held-out rows; run this small Cayuga smoke before changing
  objectives, adding preference rows, or reopening `tool_query`;
- the defer-vs-verify Cayuga smoke is negative/partial: exact top-1 remains
  1/2 and the unresolved insufficient-evidence case still routes to
  `verify` / `insufficient`, so the next repair should diagnose calibration or
  fail-closed boundary routing;
- fail-closed gate diagnostic is promising but tiny: threshold 0.025 gives
  0 unsafe trusted rows and 2/2 strict final correctness on the defer-vs-verify
  held-out slice, but it is not a calibration proof and should be expanded
  before any release or optimization escalation;
- deterministic evidence-boundary gate reaches 10/10 overall and 2/2 held-out
  using only model-visible tool-result fields; this is now the runtime baseline
  any model-routing repair must beat before `tool_query`, DPO/RLVR, or HF;
- all-family routing evidence gate reaches 25/25 overall and 5/5 held-out
  across Stage A `routing_after_loop` rows using only model-visible tool-result
  fields; this broadens the runtime-enforcement baseline beyond the tiny
  defer-vs-verify slice before release or optimization escalation;
- routing gate baseline comparison shows runtime/oracle at 25/25,
  `ground`/`supported` collapse at 5/25 with 20 unsafe overrides, and
  citationless routing at 15/25 with 10 citation mismatches; future model
  outputs should beat these deterministic baselines before escalation;
- routing model-readiness gate shows the best all-family Cayuga routing readout
  is 2/5, below citationless routing at 3/5 and runtime gate at 5/5; keep
  runtime enforcement in the system and keep optimization/release escalation
  gated;
- full-trajectory arbitration projects runtime policies through the canonical
  Stage A trajectory evaluator: runtime and hybrid policies are 25/25, collapse
  is 5/25 with 20 unsafe overrides, and citationless routing is 15/25 with
  attribution failures;
- saved-prediction readiness compares compact real Cayuga saved-output
  summaries against that full-trajectory scorecard; the best real saved output
  remains 0/5 held-out, below collapse, citationless, and runtime gates;
- saved-prediction readiness now also consumes saved-candidate gate summaries:
  the best fail-closed gate reaches 2/5 strict final with 0 unsafe trust, still
  below citationless and runtime gates;
- saved-output next-decision checkpoint selects a targeted action/status
  calibration probe as the next Stage A saved-output experiment, with minimum
  next-gate target of 0 unsafe trust and 4/5 fail-closed strict final;
- saved-output calibration probe exports 20 target-vs-`ground`/`supported`
  pairs, split into 16 train-allowed and 4 held-out evaluation-only rows with
  no case, split-group, or source-task overlap;
- the v3 saved-prediction smoke isolates an enum/schema blocker: all held-out
  outputs use invalid `evidence_status: verified`, so the next smoke should use
  `stage_a_v4_canonical_json` before interpreting evidence-label routing;
- the v4 saved-prediction smoke moves the failure from invalid status to
  missing top-level `action`/JSON envelope; next repair should use constrained
  decoding or component-level enum/action target formatting rather than broader
  prompt wording;
- finite-candidate saved-prediction readout is implemented; the next Cayuga
  smoke should test `CANDIDATE_POLICY=train_observed_pairs` before any broader
  prompt tuning, `tool_query`, DPO/RLVR, HF publication, or release tagging;
- candidate readout confirms action/status selection collapse rather than JSON
  shape as the active saved-output blocker; next work should target candidate
  calibration, fail-closed routing, or action/status supervision before leaving
  Stage A saved-output diagnostics;
- saved-candidate gate analysis is now wired as a public-safe post-hoc step over
  ignored candidate-score JSONL; the next checkpoint should record whether
  score-gap thresholds can trust any held-out candidate decisions with zero
  unsafe trust;
- saved-candidate gate results show narrow zero-unsafe coverage only: both
  train-observed and all-valid policies trust 1/5 supported case and fail closed
  on the other 4/5; keep deterministic runtime enforcement as the baseline;
- saved-output next-decision logic now derives the next experiment from compact
  checkpoints: run a targeted action/status calibration probe before any
  `tool_query`, DPO/RLVR, HF, release tag, or broad retraining step;
- the targeted calibration probe is now a public-safe artifact; use it for
  candidate calibration or action/status supervision only, then require a new
  compact saved-output summary and fail-closed gate before escalation;
- the saved-output calibration readout runner and Cayuga/Expanse templates are
  now available; run the full model path on cluster compute, select thresholds
  on train probe rows only, and evaluate the four held-out probe rows before
  changing readiness claims;
- the first Cayuga saved-output calibration readout is negative: Qwen2.5-0.5B
  scores `ground` / `supported` above the target in 20/20 probe rows, so no
  default train-selected zero-unsafe threshold is useful;
- saved-output calibration margin SFT is now implemented as the next narrow
  corrective diagnostic: run it on Cayuga first, keep held-out probe rows
  evaluation-only, and interpret only compact base-vs-trained margin summaries
  before considering any broader optimization;
- the first saved-output calibration margin SFT result is partial-negative:
  all four held-out families move positively, but only
  `verify` / `insufficient` crosses zero; keep runtime enforcement and do not
  reopen `tool_query`, DPO/RLVR, HF publication, release tagging, or broad
  retraining;
- the focused non-verify follow-up improves held-out wins to 3/4, but
  `flag` / `invalid_value` still fails on train and held-out margins; next
  isolate the `flag` action / invalid-value failure before method escalation;
- target-format isolation is implemented for saved-output margin SFT; run
  `flag` / `invalid_value` with `TARGET_FORMAT=action_only` before changing
  datasets or moving to `tool_query`, DPO/RLVR, HF publication, or release
  tagging;
- target-format isolation shows the isolated labels are learnable: next work
  should test full-output candidate scoring, length/field-normalized scoring,
  or field-wise objectives before changing optimizers;
- same-model target-format scoring is implemented; run full-target
  `flag` / `invalid_value` training with extra action/status scoring before
  choosing a field-wise or candidate-scoring objective;
- same-model target-format scoring produced the first full-target
  teacher-forced `flag` / `invalid_value` repair signal, so the next saved-output
  gate should test finite-candidate ranking or a fail-closed candidate boundary
  before any `tool_query`, DPO/RLVR, HF, release-tag, or broad-retraining step;
- finite-candidate rank scoring is implemented for that next gate; run it on
  Cayuga with base and trained candidate reports before interpreting the
  teacher-forced repair as candidate-selection progress;
- candidate-rank transfer is negative: the repair should now be treated as a
  calibration/field-routing problem, not a reason to reopen `tool_query`,
  DPO/RLVR, HF publication, release tagging, or broad retraining;
- field-rank analysis is the next compact diagnostic to run on the ignored
  candidate-score JSONL before adding any new training objective;
- field-rank analysis points away from a single-field patch: candidate
  selection now needs calibration or pair/field routing tested against runtime
  enforcement;
- non-flag pair oversampling is now measured and remains negative for candidate
  selection: raw and calibrated held-out top-1 stay at 1/4, so the next Cayuga
  checkpoint should use the explicit saved-output candidate CE pair+field
  objective and still require 4/4 held-out exact with zero trusted-candidate
  incorrect cases before any `tool_query`, DPO/RLVR, HF publication, release
  tag, or broad-retraining step;
- the saved-output candidate CE pair+field checkpoint is also negative:
  candidate top-1 remains 1/4 after calibration and raw top pairs collapse to
  `verify` / `insufficient`, so standalone candidate-routing SFT is not enough
  on this slice; keep runtime evidence arbitration as the baseline to beat;
- post-candidate-CE next decision now selects an evidence-conditioned
  saved-output bridge before more standalone SFT: map failed candidate choices
  to prompt-visible evidence reasons and full-trajectory violations before any
  `tool_query`, DPO/RLVR, HF publication, release tag, or broad-retraining step;
- the evidence-conditioned bridge joins 18/18 compact candidate-failure rows to
  prompt-visible evidence-gate rows across four held-out failure cases; next
  build a small evidence-conditioned candidate-routing substrate from these
  visible features before running another model-heavy candidate objective;
- the evidence-conditioned candidate-routing substrate is now exported as 25
  finite-candidate rows with the original 20/5 split preserved and bridge
  failures marked held-out evaluation-only; next add a no-model scorer/dry-run
  before any Cayuga model-heavy candidate objective;
- the evidence-conditioned candidate-routing readout now passes its no-model
  gate: runtime evidence routing is 5/5 held-out exact and 4/4 bridge-focus
  exact, while static priors remain at 1/5; next prepare a small Cayuga smoke
  spec for this candidate-routing slice;
- the evidence-conditioned candidate-routing smoke spec is now defined with a
  5/5 held-out and 4/4 bridge-focus model gate; next implement the runner or
  adapter before any Cayuga submission;
- the evidence-conditioned candidate-routing runner and Cayuga wrapper are now
  implemented with dry-run validation and `--allow-model-load` required for
  full mode; next run the Cayuga dry-run mirror check before any full job;
- the evidence-conditioned candidate-routing result adapter is now implemented
  and fails closed if compact eval reports contain raw candidate scores or raw
  model text; use it to curate any future Cayuga result summary;
- the Cayuga mirror dry-run checkpoint passes at commit `6820498` with 20 train
  rows, 5 held-out rows, 4 bridge-focus held-out rows, and no dry-run issues;
  the approved full smoke is complete;
- the evidence-conditioned candidate-routing full smoke fails at 4/20 train,
  1/5 held-out, and 1/4 bridge-focus exact, selecting
  `verify` / `insufficient` for every row; freeze this repeatedly inspected
  slice as diagnostic/development data and build a new source-separated sealed
  Stage A evaluation extension before further model claims;
- the sealed-extension builder now requires its candidate pool and selected
  manifest to remain outside the public repository, excludes all declared
  public source-task IDs, split groups, and normalized claims, and emits only
  aggregate balance/overlap counts plus cryptographic commitments;
- the private sealed extension now contains 25 rows balanced at 5 per action
  family, with zero source-task, split-group, or normalized-claim overlap
  against the declared public exclusions; keep labels private, complete
  `tool_query`, freeze the policy, and evaluate this extension once;
- routing gate arbitration shows raw candidate top-1 is 1/2 while score-gap
  fail-closed, evidence-boundary override, and hybrid runtime policies are 2/2;
  prioritize system enforcement before new optimization;
- any corrective data is component-specific, not broad retraining.

### 3. Preference / Process Supervision Gate

Goal: use paired rows only after component failures are measured.

Exit criteria:

- chosen rows pass validators and rejected rows fail for the intended reason;
- failure modes include self-answering, missing tool, partial query, missing
  attribution, unsupported trust, and insufficient-as-negative;
- preference improvements are not final-answer shortcuts.

### 4. Audited RLVR Gate

Goal: use RLVR only for deterministic verifier slices.

Exit criteria:

- verifier audit exists before reward use;
- eligible rewards cover schema, source existence, tool sequence, query
  completeness, citation IDs, hidden labels, or gate compliance;
- explanation quality and broad biological interpretation stay out of RLVR;
- shortcut tests fail closed.

### 5. Stage B C5 Transfer

Goal: reuse the same trajectory/evidence/action schema for antibody-antigen OOD
trust routing.

Exit criteria:

- C5 records include complex ID, chain-role mapping, metric type/scope/value,
  calibration dataset, regime match, baseline result, hidden interface label
  status, and expected action;
- uncalibrated specialist outputs route to verify, baseline, or defer;
- calibrated regime-matched records may pass only with complete metadata;
- fail-closed behavior is tested.

### 6. Release v0.1 And Hugging Face Package

Goal: publish only after the benchmark path has one compact component result.

Exit criteria:

- public QA passes on `main`;
- `scripts/check_public_release.py` and
  `scripts/check_public_git_history.py` pass;
- `post_training/validate_post_training_data.py` reports no issues;
- release notes summarize Stage A artifacts, component result, and limitations;
- license status is explicit;
- Hugging Face files match `release/public_release_manifest.json`.

## Non-Goals

- pretraining a biology foundation model;
- generic biomedical QA;
- clinical recommendation or treatment guidance;
- non-research communication as the primary deliverable;
- unaudited LLM-judge rewards;
- trusting specialist model confidence without calibration.

## Release Discipline

Before each public release:

```bash
python scripts/check_public_release.py
python scripts/check_public_git_history.py
python scripts/check_research_plan.py
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
python post_training/run_stage_a_strict_component_diagnostics.py --compact
python post_training/export_stage_a_evidence_conditioned_component_targets.py
python post_training/analyze_stage_a_component_visibility.py \
  --targets post_training/stage_a_evidence_conditioned_component_targets_v1.jsonl \
  --out-json /tmp/stage_a_evidence_conditioned_component_visibility_audit.json \
  --out-md /tmp/stage_a_evidence_conditioned_component_visibility_audit.md
python post_training/run_stage_a_strict_component_sft_smoke.py --dry-run --component enum_action
python post_training/run_stage_a_strict_component_sft_smoke.py \
  --dry-run \
  --component routing_after_loop \
  --decode-mode routing_observed_pair_score \
  --targets post_training/stage_a_evidence_conditioned_component_targets_v1.jsonl \
  --train-targets post_training/stage_a_evidence_conditioned_component_targets_train_v1.jsonl \
  --heldout-targets post_training/stage_a_evidence_conditioned_component_targets_heldout_v1.jsonl
python post_training/export_stage_a_routing_action_status_contrast_pairs.py
python post_training/run_stage_a_routing_contrast_sft_smoke.py \
  --dry-run \
  --pairwise-margin-weight 1 \
  --pairwise-margin 0.05
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
python -m pytest -q
git diff --check
```

Every milestone should update:

- `RELEASE_CHECKLIST.md`
- `release/public_release_manifest.json`
- `CHANGELOG.md`
- `PUBLIC_RELEASE_AUDIT.md` if the release boundary changes

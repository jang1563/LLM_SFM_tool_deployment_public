# Long-Term Research Plan: LLM-SFM Tool Deployment

Date: 2026-07-04

Purpose: preserve the scientific research goal while turning the current Stage A
benchmark substrate into a longer-running post-training and deployment research
program.

## Fixed Thesis

The project is a code benchmark first:

```text
tool -> evidence packet -> trust / verify / baseline / defer
```

The research claim is not that RLVR solves biology. The claim is that biology
agents need trainable trajectories plus runtime enforcement, and that each
trajectory slice should use the weakest sufficient learning or enforcement
method:

- SFT for valid tool-call format, argument completion, evidence-packet shape,
  and action enum use.
- Preference/DPO/process supervision for better-vs-worse trajectories and
  explanation/evidence-use judgment.
- RLVR/tool-use RL only for audited deterministic slices.
- Runtime gates for calibrated trust, unsupported evidence, baseline dominance,
  and fail-closed deployment behavior.

## Current State

Stage A is now code-backed:

- 25 public-safe manifest cases.
- Hidden evaluator metadata separated from `model_visible_task`.
- Oracle SFT, preference, process-supervision, and deterministic split artifacts.
- Validator checks prompt/hidden boundaries, chosen-pass/rejected-fail direction,
  process target shape, and train/held-out source overlap.
- No-API baselines verify that oracle trajectories pass and self-answer,
  wrong-tool, and partial-query shortcuts fail.
- Offline prediction-output scoring is available for saved API, local-SFT,
  prompt-only, or oracle JSONL rows. The scorer does not call APIs or load model
  weights; it only parses saved predictions and applies the same Stage A gates.
- Saved-prediction generation is artifact-first: deterministic no-API modes
  create regression fixtures, while live API generation is explicitly gated and
  still writes JSONL before scoring.
- Strict prompt-contract and strict SFT smoke runs on Cayuga are tracked as
  negative baselines: they reduce formatting or training loss but still fail
  held-out trajectory gates.
- Strict component diagnostics show the next measurable slices:
  `enum_action`, `tool_query`, and `routing_after_loop`.
- Component-slice SFT smoke has a public-safe dry-run and opt-in Cayuga/Expanse
  runner. DPO/RLVR remains gated until all three slices have held-out reports.
- The first Cayuga `enum_action` component smoke is a negative result: 0/5 pass,
  mean score 0.250, with invalid enum value and target-key violations in all
  held-out cases.
- The immediate repair path is finite-candidate decoding:
  `--decode-mode enum_candidate_score`, which constrains `enum_action` outputs
  to valid `(action, evidence_status)` JSON candidates.
- The candidate-scored Cayuga repair improves the held-out component result from
  0/5 to 1/5 and fixes schema/enum validity, but still leaves enum-pair
  selection weak.
- The full 30-candidate enum-action rerun keeps the same 1/5 held-out pass rate
  and exposes low gold ranks for insufficient and invalid-value cases.
- The observed-pair counterfactual over the same Cayuga scores still selects
  `ground` / `supported` in 5/5 held-out cases, so target-space pruning alone is
  not sufficient.
- A small enum-only corrective pair substrate now targets that collapse without
  changing `tool_query`, `routing_after_loop`, DPO, or RLVR gates.
- The first Cayuga enum corrective SFT/margin smoke is partial: 2/4 held-out
  contrast wins, with insufficient-evidence and invalid-value cases still losing
  to `ground` / `supported`.
- The follow-up margin-delta diagnostic shows useful movement from base to
  trained margins in all held-out families, but invalid-value still fails even
  on train-pair margins.
- Targeted oversampling of weak enum pairs improves mean margin but reduces
  held-out wins and leaves invalid-value at 0/4 train wins, so sampling pressure
  alone is not a stable repair.
- The evidence-conditioned candidate-routing full smoke collapses to
  `verify` / `insufficient` on all rows and reaches only 1/5 held-out exact.
  The repeatedly inspected 20/5 split is now development/diagnostic data, not
  an independent final evaluation set.
- A private 25-row sealed extension is now committed by hash, balanced at five
  rows per action family, and has zero source-task, split-group, or normalized
  claim overlap against the declared public task and manifest exclusions.

Stage B now has a synthetic no-API contract prototype:

- 12 balanced policy-test rows reuse the same action/evidence schema;
- Specialist trust must depend on metric type, metric scope, calibration dataset,
  regime match, baseline result, and fail-closed policy.
- `trust_all` and a generic threshold produce unsafe trusts; the strict
  regime-specific certifier has zero unsafe trust but no operational fallback;
  the fail-closed router passes all 12 fixture-defined actions.
- This is contract validation, not biological calibration evidence.

Stage B now also has a source-backed published-label replay:

- the exact `CC-BY-4.0` Fromm et al. archive member is checksum-verified;
- 22,000 AF3 samples over 110 targets pass strict schema and shape validation;
- a nine-column allowlist prevents raw path, structure, sequence, feature, and
  source sample-ID publication;
- confidence-ranked target selection and a frozen 55/55 target-group split are
  deterministic, with zero overlap;
- trust-all has 28 failures among 55 evaluation targets;
- fixed `ipTM >= 0.80` has 3 failures among 20 trusted evaluation targets and
  is not labeled as general-PPI calibration;
- a uniform Hoeffding search certifies no trusted set at `alpha = 0.30`,
  `0.20`, or `0.10`, so the runtime policy fails closed to verification;
- this is not an independent hidden test or new structure-prediction result.

Stage B now also has independent-source published-label calibration evidence:

- the Hitawala-Gray AF3 table is commit pinned and checksum verified;
- 1,565 complete bound predictions cover 108 Ab-Ag/Nb-Ag targets;
- 9 PDB IDs representing 11 complex copies shared with Fromm are excluded
  before calibration;
- 97 targets remain: 44 antibodies and 53 nanobodies;
- the source ranking protocol is deterministic: ranking score, then ipTM-HA,
  then lexical sample ID;
- neither format has a uniformly corrected ranking-score trust certificate at
  `alpha = 0.30`, `0.20`, or `0.10`;
- the locked Fromm evaluation therefore remains 0/55 trusted and 55/55 routed
  to verify;
- this is not a blinded hidden test, new structure prediction, or general-PPI
  transfer result.

## Drift Guard

Do not drift into:

- pretraining a biology foundation model;
- generic biomedical QA;
- clinical recommendation or treatment guidance;
- non-research communication as the primary deliverable;
- unaudited LLM-judge rewards;
- trusting SFM confidence without regime-matched calibration;
- broad weekly source scans that do not change the verifier, reward, policy, or
  benchmark design.

Every milestone should ask:

1. What is the current thesis?
2. What result changed?
3. Did any source change the verifier/reward/policy argument?
4. What decision is next?

Record the answer in `STATUS.md`.

A repeatedly inspected held-out slice must be frozen as development data. A
source-separated sealed extension with private row-level labels is required
before further model claims. Its candidate pool and selected manifest must stay
outside the public repository; only aggregate balance, overlap counts, and
cryptographic commitments may be public before the one-time evaluation.

## Research Workstreams

### Workstream A: Stage A Benchmark Maturity

Goal: measure Stage A component failures before escalating method complexity.

Next deliverables:

- Run component-slice SFT smoke on Cayuga in this order:
  `enum_action`, `tool_query`, then `routing_after_loop`.
- Use Expanse only as a fallback or replication target.
- Track only compact summaries: held-out pass rate, mean score, and violation
  counts by component.
- For repaired `enum_action`, add compact candidate-rank and top-gold margin
  diagnostics before switching slices.
- For the next `enum_action` repair, prioritize evidence-conditioned corrective
  supervision over further candidate-space pruning.
- Corrective experiments should train only on the enum train pairs and report
  held-out contrast accuracy before moving to the next component.
- Keep raw cluster outputs ignored under `post_training/runs/`.
- Update `STATUS.md` after each component run before changing the training
  method.
- Exclude every publicly exposed Stage A task, split group, normalized claim,
  and source-task ID when constructing the sealed extension.
- Keep sealed rows unavailable for training, prompt tuning, threshold selection,
  and per-case error analysis until the missing component diagnostics are frozen.

Exit criteria:

- All three component slices have a held-out report.
- Reports separate exact key, enum, structured tool-query, target-match, and
  trajectory-gate failures.
- The next method choice is justified by the failing slice, not by aggregate
  loss or prose quality.
- Negative results are reported as boundary information, not hidden.
- The sealed extension is balanced across all five action families, has zero
  overlap with declared public task/manifest exclusions, and is used only once
  for the frozen model-policy evaluation.

### Workstream B: Preference And Process Supervision

Goal: test whether paired and process-level training data reduce known
trajectory failures.

Next deliverables:

- Use the existing Stage A preference pairs as the first chosen/rejected
  substrate only after component results are known.
- Add a preference-data audit table by failure mode:
  - self-answering without tools;
  - wrong tool;
  - missing tool;
  - partial query;
  - missing attribution;
  - invalid value missed;
  - unsupported trust;
  - insufficient evidence treated as negative evidence.
- Compare prompt-only, SFT, preference-style scoring, and deterministic
  guardrails on the same held-out split.
- Add small corrective data only for the failing component slice; avoid broad
  retraining that mixes enum, tool-query, and routing failures.

Exit criteria:

- Chosen trajectories remain passing and rejected trajectories remain failing.
- Preference improvements do not come from final-answer shortcuts.
- Process-supervision targets preserve prompt/tool/final-action structure.

### Workstream C: Audited RLVR / Tool-Use RL

Goal: identify which Stage A slices can support RLVR without turning benchmark
defects into reward.

Allowed first reward slices:

- JSON/schema validity;
- required tool sequence;
- required query-field completeness;
- source existence and source ID match;
- hidden evidence-status match;
- terminal action match;
- gate compliance;
- optional cost-aware routing after the above are stable.

Disallowed first reward slices:

- explanation fluency;
- broad biological interpretation;
- unaudited LLM-judge scores;
- SFM confidence trust without calibration;
- any reward that can be satisfied by final prose alone.

Exit criteria:

- A verifier audit exists before the reward is used.
- A shortcut baseline fails under the reward.
- The reward report separates L0/L1/L2/L5 verifier slices from expert-judgment
  slices.

### Workstream D: Stage B C5 Transfer

Goal: port the Stage A evaluator/action schema to antibody-antigen OOD trust
routing.

Minimal C5 record fields:

- `complex_id`;
- chain and role mapping;
- specialist name and output ID;
- metric type, scope, and value;
- interaction regime;
- calibration dataset ID;
- calibration regime match;
- RCPS or calibration threshold ID when available;
- baseline result;
- hidden interface label status;
- expected terminal action.

Comparators:

- `trust_all`;
- free-form LLM;
- general gate;
- regime-specific gate;
- fail-closed policy.

Expected scientific result:

If calibration does not transfer, the expected action is
`verify`, `baseline`, or `defer`, not "LLM decides to trust."

Exit criteria:

- Missing metric scope, calibration dataset, or threshold fails.
- Uncalibrated Ab-Ag specialist output cannot be trusted.
- Calibrated regime-matched records may pass only with complete metadata.
- Failure is reported as a calibration result, not as model indecision.

### Workstream E: Public Research Package

Goal: keep public polish secondary but useful.

Next deliverables:

- v0.1 release tag after license status is explicit.
- Tag v0.1 only after the component path has at least one compact cluster result.
- Hugging Face Stage A dataset package only after artifact rows and cards pass
  the public release checker.
- Optional short demo video or GIF after the runnable story is stable.

Exit criteria:

- Public artifacts match `release/public_release_manifest.json`.
- No raw DB, local path, token, run log, model cache, or private infrastructure
  breadcrumb is included.
- Public claims stay benchmark-first.

## Research-First 6-8 Week Execution Board

This board supersedes the older release-first ordering. Public polish continues
only when it supports reproducibility.

| Window | Primary action | Decision gate |
| --- | --- | --- |
| Week 1 | Dry-run all component slices, then run Cayuga `enum_action`. | If enum/action fails, fix constrained decoding or target format before other training. |
| Week 2 | Run Cayuga `tool_query`, then `routing_after_loop`; use Expanse only if Cayuga is blocked. | Do not start DPO/RLVR until all three components have held-out violation reports. |
| Week 3 | Diagnose the worst failing slice and add the smallest corrective target/data change. | The correction must target one component, not broad retraining. |
| Week 4 | Re-run the corrected slice and decide whether preference/process rows are justified. | DPO requires passing chosen rows and intentionally failing rejected rows. |
| Week 5 | Write the verifier audit for any proposed RLVR reward. | RLVR is allowed only for deterministic slices with shortcut tests. |
| Weeks 6-8 | Build the first C5 manifest prototype and fail-closed trust-gate tests. | Missing metric scope, calibration dataset, or threshold means `verify`, `baseline`, or `defer`. |

## Sprint Cadence

### Sprint 0: Re-anchor

Status: this document.

Output:

- fixed thesis;
- active next code ticket;
- drift guard;
- milestone map.

### Sprint 1: Stage A Component Smoke Results

Question:

> Which Stage A component fails first under a tiny cluster-side SFT smoke?

Status: first no-API harness implemented in
`post_training/run_stage_a_sft_smoke_eval.py`; component-slice smoke implemented
in `post_training/run_stage_a_strict_component_sft_smoke.py`.

Implementation:

- dry-run `enum_action`, `tool_query`, and `routing_after_loop`;
- submit Cayuga `enum_action` first, then `tool_query`, then
  `routing_after_loop`;
- record compact summaries with pass rate, mean score, and violation counts;
- keep raw run artifacts out of git.

Decision:

- If `enum_action` fails, fix enum-constrained decoding or target format.
- If `tool_query` fails, fix structured argument generation and required query
  fields.
- If `routing_after_loop` fails, focus on evidence/action routing and citation
  grounding.

### Sprint 2: Stage A Preference/Process Diagnostic

Question:

> Do paired bad/good trajectories reduce known process failures?

Implementation:

- audit preference pairs by failure mode;
- add held-out scoring by violation type;
- build paired rows only after component results identify the failing behavior;
- test preference-style scoring or lightweight DPO smoke only after chosen rows
  pass and rejected rows fail for the intended reason.

Decision:

- If pairwise margins improve but all-candidate action selection fails, treat it
  as preference signal only and keep runtime guardrails.

### Sprint 3: Verifier Audit For RLVR

Question:

> Which reward slices are safe enough for tool-use RL?

Implementation:

- map every candidate reward to verifier level L0-L5;
- add shortcut tests for each reward;
- document what cannot be rewarded automatically.

Decision:

- Add RLVR only where shortcut tests fail closed and the reward cannot be gamed
  by final prose.

### Sprint 4: C5 Manifest Prototype

Question:

> Can Stage A's evidence/action schema express Ab-Ag specialist trust routing?

Implementation:

- build a small C5 manifest prototype with typed metric fields;
- score uncalibrated and calibrated examples through the shared evaluator;
- compare `trust_all`, free-form, general gate, regime gate, fail-closed.

Decision:

- If calibration metadata is missing, do not continue to trust experiments.
  Improve the manifest/gate first.

Status:

- synthetic contract prototype complete;
- source-backed public-score intake complete;
- independent-source calibration replay complete with zero residual PDB
  overlap;
- regime-specific trust not certified;
- next evidence gate is a larger antibody-only panel or pre-registered
  prospective Cayuga panel, not model training.

### Sprint 5: Public Research Snapshot

Question:

> Is the story understandable and reproducible from public artifacts?

Implementation:

- update `BENCHMARK_CARD.md`, `REPRODUCIBILITY.md`, `ROADMAP.md`, and
  `CHANGELOG.md`;
- run public release checks;
- tag v0.1 only after at least one compact component cluster result;
- optionally prepare Hugging Face Stage A package.

Decision:

- Tag only after license and release boundary are explicit.

## Source Refresh Cadence

Do focused source refreshes only:

- before Stage A SFT results are interpreted;
- before RLVR reward design is finalized;
- before C5 calibration/gate design is finalized;
- before public release, if a source changes the claim boundary.

Do not perform broad scans during implementation sprints. Add a source only if
it changes at least one of:

- verifier design;
- reward design;
- runtime policy;
- benchmark split/leakage design;
- calibration or fail-closed logic.

## Completed C5 Public-Score Ticket

The July 24 synthetic C5 checkpoint established the contract:

- 12 rows are balanced across `trust`, `baseline`, `verify`, and `defer`;
- oracle and fail-closed trajectories pass 12/12;
- `trust_all` produces 9 unsafe trusts and the general gate produces 8;
- the regime-specific certifier produces zero unsafe trust but only 6/12 exact
  because it returns no trusted set instead of choosing a fallback;
- hidden interface labels remain outside model-visible tasks and trajectories.

The July 25 source-backed output completes the former minimum ticket:

- documented schema/license audit for the 2026 110-complex Ab-Ag benchmark and
  FoldBench artifacts;
- an adapter that groups every sampled prediction by `complex_id`;
- frozen calibration/evaluation splits with no target crossing partitions;
- DockQ/interface success held only in evaluator metadata;
- trust-all, fixed-threshold, regime-specific certification, and fail-closed
  aggregate baselines;
- no model training or new structure prediction during intake.

The July 25 independent-source output completes the next evidence ticket:

- the Gray input is commit pinned and source-shape validated;
- all Fromm PDB IDs are blocked before selecting calibration targets;
- 44 antibody and 53 nanobody targets remain after overlap exclusion;
- antibody and nanobody ranking-score gates are calibrated separately over the
  existing finite threshold family;
- neither format certifies a trusted set, so the locked Fromm gate remains
  fail closed;
- no model training or new structure prediction was used.

General-PPI transfer remains unmeasured because no validated full per-sample
PPI/Ab-Ag confidence table is available through the audited FoldBench
GitHub/Zenodo release.

## Completed Prospective C5 Freeze Ticket

The next evidence panel is now locked without reading prediction scores or
DockQ/interface labels:

- SAbDab2 v0.1.0 and its official antibody-and-antigen sequence-aware split are
  checksum and schema locked;
- target inclusion, all prior Fromm/Gray PDB exclusions, AF3 v3.0.3, five
  samples, ranking metric, DockQ v2.1.3, threshold family, alpha, delta,
  correction, reveal order, and stopping rule are preregistered;
- deterministic metadata-only selection yields 80 calibration, 40 evaluation,
  20 calibration-reserve, and 10 evaluation-reserve targets;
- 150/150 native structures pass private chain/sequence QC with no reserve
  promotions;
- 120 template-free AF3 inputs are frozen by an aggregate checksum, with no
  raw sequence, structure, local path, or label in the public artifacts;
- the 80-target calibration panel can support a zero-failure certificate at
  primary `alpha = 0.30` and potentially 0.20, but cannot support 0.10 under
  the preregistered uniform bound.

This is prospective method/source/input evidence, not an AF3 result or a
blinded hidden test.

## Completed Prospective C5 Phase-Gate Implementation Ticket

The code path after GPU execution is now fixed before any real prediction or
label exists:

- AF3 v3.0.3 intake requires one sanitized target directory, the exact five
  seed/sample directories, official top-level artifacts, and no extra or
  symlinked files;
- full-precision ranking CSV values are cross-checked against the rounded
  summary JSON, and target selection applies the preregistered
  ranking-score/ipTM/lexical rule;
- all 600 samples and 120 selected models receive immutable aggregate
  commitments without public target-level scores or paths;
- a private 150-target native-structure map must reconstruct the aggregate
  checksum frozen before prediction;
- calibration reveal accepts exactly the 80 calibration targets and binds each
  DockQ value to the selected model, native structure, chain mapping, evaluator
  version/commit, and metric scope;
- the finite-grid certificate or `verify_all` action is frozen before the
  40-target evaluation file can be opened;
- evaluation applies only trust-all, fixed ranking-score 0.80,
  regime-specific calibrated, and fail-closed preregistered policies.
- Cayuga runtime binding executes from the pinned AF3 image rather than the
  incompatible host Python, and each task verifies that the mounted
  container, model files, database manifest/sentinels, and input set match the
  passed private attestation before inference.
- The official-Dockerfile-aligned v3.0.3 Apptainer build now completes on a
  Cayuga CPU node. Its SIF checksum, embedded source/package tests,
  runner-import test, and one-device JAX GPU smoke pass; the image includes no
  parameters or databases.
- The official AF3 database fetch now completes with atomic promotion. A
  private v3 inventory content-hashes all 9 required entries and 195,867 files;
  the public checkpoint contains only aggregate size, content identities, and
  gate status. Terms-confirmed materialization of the official model object is
  now the sole unresolved runtime dependency.
- The official split-stage path is implemented before execution: the CPU array
  promotes only an exact processed data JSON, and the GPU array runs
  inference-only in private staging, validates the canonical five-sample
  output, and then replaces the processed target. The combined path remains a
  fallback and final prediction intake is unchanged.
- The 2026-07-23 official distribution change is implemented as a private
  atomic provisioning gate. The Cayuga job requires action-time terms
  acceptance, pins the official Google Storage object generation and expected
  byte count, restricts the transfer and redirects to HTTPS, rejects symlinks
  and multiple model families, and content-hashes the private copy. No
  parameter artifact is present. The earlier user-asserted direct-from-Google
  path remains a legacy fallback.
- The v2 private runtime-attestation path is implemented before parameter
  access. One Cayuga CPU job binds the clean benchmark commit, clean AF3 source,
  container, provenance-bound model manifest, database inventory, frozen inputs, and
  output boundary; it then performs a same-job quick mount recheck before
  sidecar-first atomic promotion. Arrays and the final prediction lock reject
  benchmark or model-manifest drift.

Synthetic tests cover certified, uncertified, target-mixing, model/native
drift, partial-output, score-drift, and threshold-mutation paths. This closes a
software phase-gate milestone only. The real workflow remains
`panel_locked_prediction_pending`.

## Next Concrete Ticket

Complete `stage_b_c5_af3_environment_attestation_and_prediction`.

Minimum next output:

- confirm the current official AlphaFold 3 Model Parameters Terms of Use and
  materialize the generation-pinned object in private Cayuga storage;
- run the dedicated Cayuga attestation job until every benchmark, source,
  container, provenance-bound model manifest, parameter, database, input, and
  output-boundary component passes;
- retain quick mounted-dependency verification inside every array task after
  the attestation job's full scan and immediate quick recheck;
- submit the 120-target Cayuga CPU data-pipeline array and then the GPU
  inference array only from a checksum-frozen private attestation, with the
  combined array and Expanse retained as fallbacks;
- freeze five outputs per target and select one by the preregistered rule before
  any DockQ computation;
- reveal calibration labels first, freeze a certificate or `verify_all`, then
  reveal the evaluation labels once;
- evaluate the preregistered trust-all, fixed-0.80, certified, and fail-closed
  policies; keep any later shuffled/inverted diagnostics explicitly
  non-primary;
- no DPO/RLVR, model training, evaluation tuning, or external specialist trust
  while prediction and calibration evidence remain incomplete.

Do not repeat or tune on the completed 25-row sealed set. Keep DPO, RLVR, and
Hugging Face publication closed until a learned routing repair beats static
baselines, adds useful decisive coverage, and survives independent evaluation.

## One-Paragraph Research Story

This project studies how biology agents should use scientific tools and
specialist foundation models under uncertainty. The first substrate, Stage A,
tests whether an agent can produce valid tool-use trajectories for biomedical
negative-evidence claims: call the right tool, build a complete evidence packet,
cite sources, and choose `ground`, `reject`, `verify`, or `defer` without
self-answering. Post-training methods are then assigned according to the
available feedback signal: SFT for reference trajectories, preference/process
supervision for better evidence use, RLVR only for audited verifier slices, and
runtime gates for calibrated trust. The second substrate, C5, transfers the same
schema to antibody-antigen specialist trust routing, where high confidence is
not permission unless metric scope and calibration regime match.

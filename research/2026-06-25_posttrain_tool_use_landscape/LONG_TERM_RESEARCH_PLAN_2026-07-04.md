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

Stage B is designed but not yet built:

- C5 antibody-antigen OOD trust routing should reuse the same action/evidence
  schema.
- Specialist trust must depend on metric type, metric scope, calibration dataset,
  regime match, baseline result, and fail-closed policy.

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

## Next Concrete Ticket

Diagnose Stage A `enum_action` candidate ranking.

Minimum output:

- compare candidate-score margins for the five held-out cases;
- identify why `ground` / `supported` dominates mismatched cases;
- test a slice-specific ranking or calibration diagnostic without changing the
  broader Stage A schema;
- `STATUS.md` checkpoint with thesis, result, source changes, and next decision;
- no raw cluster logs or private infrastructure identifiers committed.

Do not start DPO, RLVR, v0.1 tagging, or Hugging Face publication until at least
one component cluster result is summarized and the gate decision is explicit.
The first result is now summarized; the active gate decision is to repair
`enum_action` pair selection before broad retraining or method escalation.

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

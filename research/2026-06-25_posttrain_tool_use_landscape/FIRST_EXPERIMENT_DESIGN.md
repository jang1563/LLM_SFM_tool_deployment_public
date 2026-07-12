# First Experiment Design: Biology SFM Trust Routing

Purpose: translate the source map into the first trainable/evaluable substrate
for this repo.

Active operational bridge: `STAGE_A_C5_RESEARCH_BRIDGE_2026-07-04.md`. Use that
file for the current Stage A mini-manifest and C5 ticket sequence; keep this
file as the higher-level experiment design.

## Design Thesis

The first experiment should not ask whether an LLM can "do biology." It should
ask whether post-training or deployment enforcement improves the tool trajectory:

```text
scientific intent -> specialist/tool call -> evidence packet
-> calibrated action: trust | verify | baseline | defer
```

The local Phase 4 result already says free-form LLM routing can lose to a
deterministic calibrated-risk gate. The new source map clarifies why: SFM outputs
are useful but their confidence metrics are metric-specific, regime-specific,
and often weaker than they look.

## Recommended Two-Stage Substrate

### Stage A: Verifiable Retrieval / Evidence-Status Dry Run

Use a low-cost tool environment before spending HPC/API budget.

- Task: database or NullAtlas-style claim verification.
- Actions: `call_tool`, `verify`, `reject_or_flag_unsupported_claim`,
  `defer_or_request_more_evidence`.
- Clean rewards:
  - valid schema,
  - correct database/source selected,
  - complete filter/query construction,
  - hidden evidence-status label,
  - gate-compliance with unsupported or negative evidence.
- Training path:
  1. prompt-only baseline,
  2. curated/expert trajectory SFT,
  3. preference pairs for better evidence use,
  4. RLVR only after the evaluator and source audit are stable.

Why first: this builds the trajectory schema and evaluator without depending on
SFM calibration quality.

### Stage B: C5 Antibody-Antigen OOD Trust Gate

Port the same schema to the local C5 boundary test.

- Task: decide when to trust, verify, baseline, or defer for antibody-antigen
  complex predictions.
- Specialist outputs:
  - Boltz / AlphaFold-family structure predictions,
  - pLDDT/PAE/pTM/ipTM or model-specific confidence,
  - optional affinity prediction,
  - interface success labels when available.
- Conditions:
  - `trust_all`,
  - `C1_free_form_llm`,
  - `C2_general_gate`,
  - `C5_regime_gate`,
  - `C5_fail_closed`.
- Clean-ish rewards:
  - record/schema validity,
  - correct extraction of confidence metrics,
  - calibrated gate compliance,
  - hidden interface-success label when available,
  - cost penalty for unnecessary verification,
  - policy penalty for trusting uncalibrated OOD confidence.

## Evidence Packet Additions

Add these fields to the local minimal harness contract when moving from Stage A
to Stage B:

```text
specialist_metric_type
confidence_metric_scope        # local, pairwise, global, interface, affinity
interaction_regime             # generic PPI, antibody-antigen, ligand, etc.
calibration_dataset_id
calibration_regime_match
rcps_threshold_id
interface_label_source
assay_or_structure_truth_status
fail_closed_reason
```

## Reward / Policy Split

| layer | can be trained/scored | should be enforced |
| --- | --- | --- |
| Tool syntax | valid inputs, complete fields, parseable outputs. | schema validation. |
| Evidence retrieval | correct accession/source/record and hidden evidence label. | source existence and provenance logging. |
| SFM metric extraction | correct pLDDT/PAE/ipTM/affinity extraction and metric scope. | typed metric schema. |
| SFM trust action | trust only under calibrated threshold. | uncalibrated/OOD requires verify, baseline, or defer. |
| Biological interpretation | evidence-aware explanation and uncertainty. | no unsupported causal or deployment claim. |

## Method Ladder For Stage A

| method | use only after | target behavior |
| --- | --- | --- |
| Prompt baseline | task schema exists | measure untrained tool/evidence behavior. |
| SFT | reference trajectories exist | valid tool choice, complete evidence packet, correct action enum. |
| DPO / preference optimization | paired better/worse traces exist | prefer complete, calibrated, source-backed trajectories over plausible shortcuts. |
| Process supervision | step labels or validators exist | reward intermediate tool/evidence/gate steps before final answer. |
| RLVR / tool-use RL | audited verifiable rewards exist | optimize query/tool/action policy without letting sparse final rewards teach bad shortcuts. |

## Minimal Success Criteria

1. The evaluator rejects trajectories that trust an SFM confidence score without
   recording metric type, metric scope, and calibration regime.
2. A prompt-only model can be compared against a deterministic gate using the
   same hidden labels and action costs.
3. SFT improves valid tool-call and evidence-packet completion before any RLVR
   claim is made.
4. RLVR is restricted to verifiable slices: schema validity, source retrieval,
   hidden labels, metric extraction, and policy compliance.
5. In C5, if the antibody-antigen regime has no certifiable threshold, the
   correct output is `fail_closed -> verify/defer`, not "LLM decides to trust."

## Tutor-Mode Hook

Explain this as:

1. A foundation model is a specialist witness, not a judge.
2. Its confidence is evidence, not permission.
3. Calibration turns evidence into bounded trust.
4. If calibration breaks under OOD, the deployment layer must fail closed.
5. Post-training helps when it teaches the agent to produce the right evidence
   packet and obey the gate, not when it memorizes a confident style.

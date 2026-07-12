# Stage A -> C5 Research Bridge

Date: 2026-07-04

Purpose: turn the current landscape map into the next research action for this
repo. This is not a new literature dump. It is the bridge from source cards to a
trainable/evaluable experiment.

## Decision

Proceed with a two-lane research program:

1. **Stage A: verifiable retrieval / evidence-status trajectories.**
   Build the low-cost environment where rewards are mostly auditable: schema
   validity, source lookup, evidence status, attribution, value validation, and
   policy-compliant `verify/defer/reject` actions.
2. **Stage B: C5 antibody-antigen OOD trust gate.**
   Port the same action/evidence schema to SFM specialist outputs where trust
   depends on regime-matched calibration, not on a generic confidence score.

The research contribution is the connection between the two lanes:

```text
database/tool verifier -> evidence packet -> calibrated action policy
                         -> post-training data/reward hooks
                         -> C5 specialist-trust boundary test
```

## Source Recheck

Checked on 2026-07-04 from official or primary sources.

| source | verified anchor | design implication |
| --- | --- | --- |
| Anthropic agents-in-biology / gget virus | Deterministic retrieval made biological sequence retrieval far more reliable and reproducible across agents; outputs include machine-readable logs. | Stage A should treat retrieval infrastructure as part of the scientific system, not as incidental prompting. |
| EMBL-EBI AlphaFold-Multimer confidence guide | pTM, ipTM, pLDDT, and PAE have different scopes; ipTM has confident, failed, and grey-zone regimes, and overall multimer confidence should combine metrics. | Stage B evidence packets must type the metric and scope before any `trust_specialist_output` action is valid. |
| Boltz official GitHub docs | Boltz affinity outputs include separate fields with different supervision and intended contexts. | Do not collapse structure confidence, affinity value, and binder probability into one trust scalar. |
| Risk-controlling prediction sets / conformal risk control | Risk control uses holdout calibration to choose thresholds with explicit finite-sample or expected-loss guarantees. | A trust gate is a calibrated decision rule; if calibration does not transfer to C5, the correct behavior is fail-closed. |

## Core Research Questions

1. Can post-training improve **trajectory mechanics** before it improves final
   biological interpretation?
2. Which parts of biology tool use are legitimate RLVR targets, and which parts
   require preference/process supervision or hard runtime enforcement?
3. Does a trajectory trained on verifiable evidence-status tasks transfer to C5
   as better metric extraction, regime recognition, and fail-closed action
   choice?
4. When a specialist model emits a high-confidence output outside the calibrated
   regime, can the agent learn or be forced to choose `verify`, `baseline`, or
   `defer` instead of trusting it?

## Stage A: Verifiable Retrieval / Evidence-Status Dry Run

### Unit Of Work

One record is one scientific claim or evidence request:

```text
input_id
claim
allowed_tools
required_tool_sequence
gold_evidence_status
gold_source_ids
expected_terminal_action
cost_profile
```

One trajectory is:

```text
tool choice -> valid tool call -> observation -> evidence packet
-> terminal action -> cited source ids
```

### Reward Slices

| slice | reward type | can support RLVR? | note |
| --- | --- | --- | --- |
| JSON/schema validity | deterministic | yes | Cheap first reward. |
| Required tool sequence | deterministic | yes | Prevents self-answer shortcuts. |
| Query/filter completeness | deterministic or rubric | partial | Exact for small curated tasks; rubric for messy tasks. |
| Source attribution | deterministic | yes | Compare source IDs. |
| Evidence status | hidden label | yes if curated | `supported`, `unsupported`, `invalid_value`, `insufficient`. |
| Terminal action | hidden label/policy | yes for curated cases | `verify`, `reject`, `defer`, etc. |
| Explanation quality | expert/preference | no | Keep out of RLVR unless separately judged. |

### Minimal Experiment

1. Freeze action enum and evidence-packet schema from
   `llm_sfm_tool_deployment/trajectory.py`.
2. Use the public synthetic demo only as a smoke test.
3. Build a small private Stage A manifest from NegBioDB/NullAtlas-style cases,
   separating:
   - supported negative evidence,
   - unsupported claim,
   - invalid value,
   - insufficient evidence,
   - attribution failure.
4. Run prompt-only baseline.
5. Generate reference trajectories.
6. Train/evaluate SFT for:
   - valid tool calls,
   - evidence packet completion,
   - correct terminal action,
   - citation/source grounding.
7. Add preference pairs for common failures:
   - self-answer without tool,
   - partial query,
   - hallucinated citation,
   - unsupported confidence,
   - unnecessary verification.
8. Only then add RLVR/tool-use RL for deterministic slices.

### Stage A Stop Criteria

Stop and revise the environment if any of these happen:

- The model can pass by matching final prose without correct tool use.
- A reward depends on an unaudited LLM judge.
- Source IDs or hidden labels leak through prompts.
- Invalid values are learned as stylistic refusal rather than tool-checked
  failure.
- The evaluator cannot distinguish `unsupported` from `insufficient`.

## Stage B: C5 Antibody-Antigen OOD Trust Gate

### Unit Of Work

One record is one specialist-output trust decision:

```text
complex_id
specialist_name
interaction_regime
metric_type
metric_scope
metric_value
calibration_dataset_id
calibration_regime_match
baseline_result
hidden_interface_success_label
allowed_actions
expected_terminal_action
```

### C5 Conditions

| condition | question |
| --- | --- |
| `trust_all` | What happens if every high-confidence specialist output is trusted? |
| `C1_free_form_llm` | Does an LLM reading a reliability card choose the right action? |
| `C2_general_gate` | Does the previous calibrated gate transfer to Ab-Ag? |
| `C5_regime_gate` | Can Ab-Ag-specific calibration certify a trusted subset? |
| `C5_fail_closed` | If no certification is available, does the system refuse trust? |

### Stage B Reward / Enforcement Split

| layer | trainable | enforceable |
| --- | --- | --- |
| Metric extraction | metric name, value, scope, parser correctness | typed schema |
| Regime recognition | antibody-antigen vs generic PPI vs ligand context | allowed-regime table |
| Calibration status | match/mismatch to calibration dataset | no trust on mismatch |
| Baseline comparison | extract/use baseline result | default to baseline if it dominates |
| Hidden interface outcome | success/failure label if available | evaluation only |
| Trust action | choose `trust/verify/baseline/defer` | fail-closed policy |

### C5 Stop Criteria

Stop and revise before any LLM/API claim if:

- The Ab-Ag panel lacks chain/interface labels.
- Metric extraction is inconsistent across specialist outputs.
- General-gate thresholds are reused without checking regime match.
- A high ipTM/affinity score is treated as permission without calibration.
- RCPS/conformal calibration returns no threshold but the system still trusts.

## Method Assignment

| question | first method | why |
| --- | --- | --- |
| Can the model produce valid trajectories? | SFT | Imitation is the cleanest first improvement target. |
| Can it prefer complete/source-backed traces? | DPO / preference optimization | Pairwise labels fit common trajectory failures. |
| Can it optimize query/action policy under costs? | RLVR/tool-use RL | Only after deterministic rewards are audited. |
| Can it trust a specialist score? | runtime gate first, then process/DPO | Trust is a policy action, not just a learned habit. |
| Can it explain biology well? | expert preference/process supervision | Explanation is not a clean verifier slice. |

## Immediate Research Tickets

### A0: Freeze The Stage A Record Contract

- Compare `TaskSpec`, `Trajectory`, and `EvidencePacket` fields against the
  Stage A unit of work above.
- Add only missing fields that are needed for reward/eval; avoid a new parallel
  schema.
- Current audit: `STAGE_A_SCHEMA_AUDIT_2026-07-04.md`.

### A1: Build A Stage A Mini-Manifest

- Select 20-50 private cases from NegBioDB/NullAtlas-style artifacts.
- Balance evidence statuses.
- Store source IDs and hidden labels separately from prompts.
- Run the existing public demo as a smoke test, then private manifest eval.

### A2: Make A Failure-Mode Matrix

For each case, generate bad trajectories:

- self-answer without tool,
- wrong tool,
- partial query,
- missing attribution,
- invalid value missed,
- unsupported claim trusted,
- insufficient evidence treated as negative evidence.

These become DPO/process-supervision candidates.

### C0: Prepare C5 Manifest Requirements

- Define the minimal C5 target record before touching HPC:
  `complex_id`, `pdb_id`, chain mapping, antigen/binder role, interface label
  source, specialist output path, metric fields, calibration split.
- Treat missing interface label as `defer`, not as a negative result.

### C1: Align C5 To The Same Evaluator

- Reuse the `trust_specialist_output`,
  `verify_with_assay_or_database`, `use_cheap_baseline`, and
  `defer_or_request_more_evidence` actions.
- Make C5 pass/fail through the same trajectory evaluator wherever possible.

## What Would Count As A Strong Result

A strong Stage A result:

- SFT improves tool-call validity and evidence-packet completion.
- DPO/process supervision reduces unsupported trust and missing attribution.
- RLVR improves only audited slices, without creating final-answer shortcuts.

A strong C5 result:

- The agent extracts specialist metrics correctly.
- It recognizes Ab-Ag as a separate calibration regime.
- It refuses trust when calibration is missing or non-transferrable.
- A regime-specific gate either certifies a trusted subset or visibly fails
  closed.

The publishable claim is not:

> RLVR makes the model reason like a scientist.

The stronger, safer claim is:

> Biology tool-use post-training works best when trajectories are decomposed
> into verifiable slices, preference/process-supervised judgment slices, and
> runtime-enforced trust policies.

## Korean Tutor Hook

짧게 설명하면:

> 먼저 “도구를 제대로 쓰고 증거를 제대로 기록하는가”를 Stage A에서 검증한다.
> 그 다음 C5에서는 “전문 SFM 점수를 믿어도 되는 조건인가”를 검증한다.
> 신뢰는 모델의 말투가 아니라, metric scope와 calibration regime이 맞을 때만
> 허용되는 action이다.

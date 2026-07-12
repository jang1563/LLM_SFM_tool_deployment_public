# Stage A Schema Audit

Date: 2026-07-04

Purpose: check whether the existing trajectory evaluator can support the Stage A
retrieval/evidence-status mini-manifest proposed in
`STAGE_A_C5_RESEARCH_BRIDGE_2026-07-04.md`.

## Summary

The current schema is already sufficient for the first Stage A dry run. Do not
fork a new schema yet.

The missing pieces are mostly **manifest metadata**, not evaluator dataclass
fields:

- `cost_profile`,
- `allowed_tools`,
- case family / split metadata,
- hidden-label separation from model-visible prompt,
- optional query/filter completeness rubrics.

Those should live in the Stage A manifest and adapter first. Promote them into
`TaskSpec` or `EvidencePacket` only if multiple runners need them directly.

## Current Schema Coverage

| Stage A need | current support | note |
| --- | --- | --- |
| `input_id` | `TaskSpec.input_id`, `Trajectory.input_id`, `EvidencePacket.input_id` | Fully covered. |
| `claim` | `TaskSpec.claim` | Hidden task metadata, not necessarily model-visible. |
| required tool sequence | `TaskSpec.required_tools`, `Trajectory.steps` | Ordered-subsequence check already implemented. |
| tool arguments / observations | `ToolStep.arguments`, `ToolStep.observation` | Enough for query/filter audit without new top-level fields. |
| evidence status label | `TaskSpec.gold_evidence_status`, `EvidencePacket.negative_evidence_status` | Supports `supported`, `insufficient`, `contradicted`, `invalid_value`, `unknown`. |
| expected terminal action | `TaskSpec.expected_terminal_action`, `Trajectory.terminal_action` | Fully covered. |
| attribution/source IDs | `TaskSpec.gold_source_ids`, `Trajectory.cited_source_ids` | Fully covered for source existence and citation checks. |
| external-tool requirement | `TaskSpec.requires_external_tool` | Blocks self-answer shortcut. |
| web-zero policy | `TaskSpec.web_zero`, `EvidencePacket.web_exposure_tag` | Already enforced. |
| cheap baseline | `EvidencePacket.cheap_baseline_output`, `baseline_dominance_flag` | Already blocks specialist trust when baseline dominates. |
| C5 metric schema | `specialist_metric_type`, `confidence_metric_scope`, `interaction_regime`, `calibration_dataset_id`, `calibration_regime_match`, `rcps_threshold_id` | Already present. |
| fail-closed reason | `EvidencePacket.fail_closed_reason` | Present, but not yet required by evaluator. |

## Existing Evaluator Strengths

The current evaluator already scores or blocks:

- missing required tool sequence,
- missing attribution,
- evidence-status mismatch,
- terminal-action mismatch,
- self-answering on external-tool or web-zero tasks,
- trusting uncalibrated specialist output,
- trusting when cheap baseline dominates,
- trusting C5 antibody-antigen output without metric/calibration schema,
- trusting C5 output without regime-matched calibration.

This is enough to make the public synthetic demo meaningful and enough to build
a private 20-50 case Stage A mini-manifest without code changes.

## Gaps To Handle In The Manifest First

### `allowed_tools`

Current `TaskSpec` has `required_tools`, not an explicit allowed tool universe.
For Stage A, put `allowed_tools` in the manifest and validate it in the adapter
or runner. Only add it to `TaskSpec` if wrong-tool-but-complete-sequence cases
become a recurring scoring problem.

### `cost_profile`

The evaluator does not yet score cost-aware routing. That is fine for A0/A1.
Store per-action/tool costs in the manifest:

```text
cost_profile:
  call_tool: 1.0
  verify_with_assay_or_database: 3.0
  defer_or_request_more_evidence: 0.5
```

Add evaluator cost rewards only after the correctness gates are stable.

### Query / Filter Completeness

Current `ToolStep.arguments` can store query details, but the evaluator does not
know task-specific required filters. Store expected filters in hidden manifest
metadata first:

```text
required_query_fields:
  - condition_id
  - intervention_id
  - source_record_id
```

Then add a small manifest-level checker before turning this into RLVR reward.

### Hidden Label Separation

For training/eval hygiene, do not expose these fields to the model prompt:

- `gold_evidence_status`,
- `gold_source_ids`,
- `expected_terminal_action`,
- hidden source/record IDs unless the task explicitly asks for attribution,
- split/source IDs used for leakage control.

The manifest should have at least two projections:

```text
model_visible_task
hidden_eval_metadata
```

### Fail-Closed Reason

`EvidencePacket.fail_closed_reason` exists, but the evaluator currently checks
the action rather than requiring a reason string. Keep it optional for Stage A.
In C5, consider requiring it for `C5_fail_closed` cases so the model records why
trust was refused.

## A1 Manifest Shape

Recommended minimal JSONL row:

```json
{
  "case_id": "stage_a::supported::0001",
  "model_visible_task": {
    "input_id": "stage_a::supported::0001",
    "claim": "Synthetic or private claim text",
    "allowed_tools": ["survey_prior_failures", "verify_trial_claim"]
  },
  "hidden_eval_metadata": {
    "required_tools": ["survey_prior_failures", "verify_trial_claim"],
    "gold_evidence_status": "supported",
    "expected_terminal_action": "verify_with_assay_or_database",
    "gold_source_ids": ["NCT..."],
    "requires_attribution": true,
    "requires_external_tool": true,
    "case_family": "supported_negative_evidence",
    "split_group": "source_record_or_claim_cluster"
  },
  "cost_profile": {
    "call_tool": 1.0,
    "verify_with_assay_or_database": 3.0,
    "defer_or_request_more_evidence": 0.5
  }
}
```

## Recommendation

For the next coding step:

1. Add a Stage A manifest loader/adapter that projects
   `hidden_eval_metadata` into `TaskSpec`.
2. Keep `model_visible_task` separate from hidden labels.
3. Reuse `TrajectoryEvaluator` unchanged for the first pass.
4. Add manifest-level checks for split overlap, source leakage, and balanced
   evidence-status counts.
5. Only add cost-aware reward once A1 passes on correctness-only scoring.

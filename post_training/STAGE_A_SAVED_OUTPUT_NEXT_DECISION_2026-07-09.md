# Stage A Saved-Output Next Decision

Purpose: choose the next Stage A saved-output experiment from compact
public-safe readiness and candidate-gate checkpoints.

## Bottleneck

- Active bottleneck: `narrow_fail_closed_coverage_under_citationless`
- Best raw saved-output pass count: 0/5
- Best fail-closed candidate gate: 2/5 strict final, 0 unsafe trust
- Candidate top-pair counts: `{"ground/supported": 10}`
- Candidate failure targets: `{"defer/insufficient": 2, "flag/invalid_value": 2, "reject/contradicted": 2, "verify/insufficient": 2}`

## Decision

- Selected next step: `targeted_action_status_calibration_probe`
- Why: The saved-output path has moved past parse/tool/query failures, but candidate top-1 still collapses to ground/supported and the score-gap gate only safely trusts the supported row.

Keep gated:
- `tool_query`
- `DPO/RLVR`
- `Hugging Face publication`
- `release tagging`
- `broad retraining`

Minimum success criteria for the next Cayuga checkpoint:
- `real_saved_output_passed_must_exceed_collapse`: `True`
- `fail_closed_gate_trusted_incorrect`: `0`
- `fail_closed_gate_strict_final_correct_min`: `4`
- `raw_predictions_remain_uncommitted`: `True`

Artifact policy: raw saved predictions, candidate-score JSONL, scheduler logs,
model state, and ignored run folders stay uncommitted.

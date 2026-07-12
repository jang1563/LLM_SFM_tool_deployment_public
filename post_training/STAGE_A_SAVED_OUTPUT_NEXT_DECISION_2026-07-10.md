# Stage A Saved-Output Next Decision

Purpose: choose the next Stage A saved-output experiment from compact
public-safe readiness and candidate-gate checkpoints.

## Bottleneck

- Active bottleneck: `runtime_evidence_arbitration_beats_saved_output_candidates`
- Best raw saved-output pass count: 0/5
- Best fail-closed candidate gate: 2/5 strict final, 0 unsafe trust
- Candidate top-pair counts: `{"ground/supported": 10}`
- Candidate failure targets: `{"defer/insufficient": 2, "flag/invalid_value": 2, "reject/contradicted": 2, "verify/insufficient": 2}`
- Calibration held-out exact: raw 1/4, calibrated 2/4, train-selected gate 1/4
- Arbitration exact: raw 1/4, calibrated 2/4, score-gap 1/4, evidence 4/4, hybrid 4/4
- Hidden labels used by arbitration: `False`

## Decision

- Selected next step: `meet_or_beat_runtime_evidence_arbitration_baseline`
- Why: The targeted calibration probe is complete: train-derived calibration improves held-out candidate top-1, but still underperforms model-visible evidence/hybrid arbitration. The next model-heavy checkpoint should meet or beat that runtime baseline rather than start DPO/RLVR.

Keep gated:
- `tool_query`
- `DPO/RLVR`
- `Hugging Face publication`
- `release tagging`
- `broad retraining`

Minimum success criteria for the next Cayuga checkpoint:
- `candidate_or_model_policy_exact_min`: `4`
- `trusted_candidate_incorrect`: `0`
- `hidden_labels_used_by_arbitration`: `False`
- `raw_predictions_remain_uncommitted`: `True`

Artifact policy: raw saved predictions, candidate-score JSONL, scheduler logs,
model state, and ignored run folders stay uncommitted.

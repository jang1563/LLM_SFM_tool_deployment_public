# Stage A Saved-Output Candidate Arbitration

Purpose: compare raw candidate top-1, train-derived calibration,
train-selected score-gap gating, and model-visible evidence arbitration
on the saved-output held-out calibration slice.

## Summary

- Run ID: `stage_a_saved_output_flag_full_train_candidates_qwen05b_cayuga_20260710`
- Cases: 4
- Train-selected threshold: `0.045521`
- Hidden labels used by arbitration: `False`
- Best policy names: `["evidence_gate_override", "hybrid_evidence_then_train_gate"]`

| Policy | Exact | Rows | Trusted candidate | Unsafe candidate trust | Error case IDs |
| --- | ---: | ---: | ---: | ---: | --- |
| raw_candidate_top1 | 1 | 4 | 4 | 3 | `["stage_a::000007", "stage_a::000012", "stage_a::000019"]` |
| calibrated_candidate_top1 | 2 | 4 | 4 | 2 | `["stage_a::000007", "stage_a::000021"]` |
| train_selected_score_gap_gate | 1 | 4 | 0 | 0 | `["stage_a::000007", "stage_a::000019", "stage_a::000021"]` |
| evidence_gate_override | 4 | 4 | 0 | 0 | `[]` |
| hybrid_evidence_then_train_gate | 4 | 4 | 0 | 0 | `[]` |

## Rows

| Case | Target | Raw top | Calibrated top | Evidence gate | Policy | Predicted | Exact |
| --- | --- | --- | --- | --- | --- | --- | ---: |
| `stage_a::000007` | `reject/contradicted` | `flag/invalid_value` | `defer/insufficient` | `reject/contradicted` | `raw_candidate_top1` | `flag/invalid_value` | 0 |
| `stage_a::000007` | `reject/contradicted` | `flag/invalid_value` | `defer/insufficient` | `reject/contradicted` | `calibrated_candidate_top1` | `defer/insufficient` | 0 |
| `stage_a::000007` | `reject/contradicted` | `flag/invalid_value` | `defer/insufficient` | `reject/contradicted` | `train_selected_score_gap_gate` | `defer/insufficient` | 0 |
| `stage_a::000007` | `reject/contradicted` | `flag/invalid_value` | `defer/insufficient` | `reject/contradicted` | `evidence_gate_override` | `reject/contradicted` | 1 |
| `stage_a::000007` | `reject/contradicted` | `flag/invalid_value` | `defer/insufficient` | `reject/contradicted` | `hybrid_evidence_then_train_gate` | `reject/contradicted` | 1 |
| `stage_a::000012` | `defer/insufficient` | `flag/invalid_value` | `defer/insufficient` | `defer/insufficient` | `raw_candidate_top1` | `flag/invalid_value` | 0 |
| `stage_a::000012` | `defer/insufficient` | `flag/invalid_value` | `defer/insufficient` | `defer/insufficient` | `calibrated_candidate_top1` | `defer/insufficient` | 1 |
| `stage_a::000012` | `defer/insufficient` | `flag/invalid_value` | `defer/insufficient` | `defer/insufficient` | `train_selected_score_gap_gate` | `defer/insufficient` | 1 |
| `stage_a::000012` | `defer/insufficient` | `flag/invalid_value` | `defer/insufficient` | `defer/insufficient` | `evidence_gate_override` | `defer/insufficient` | 1 |
| `stage_a::000012` | `defer/insufficient` | `flag/invalid_value` | `defer/insufficient` | `defer/insufficient` | `hybrid_evidence_then_train_gate` | `defer/insufficient` | 1 |
| `stage_a::000019` | `verify/insufficient` | `flag/invalid_value` | `verify/insufficient` | `verify/insufficient` | `raw_candidate_top1` | `flag/invalid_value` | 0 |
| `stage_a::000019` | `verify/insufficient` | `flag/invalid_value` | `verify/insufficient` | `verify/insufficient` | `calibrated_candidate_top1` | `verify/insufficient` | 1 |
| `stage_a::000019` | `verify/insufficient` | `flag/invalid_value` | `verify/insufficient` | `verify/insufficient` | `train_selected_score_gap_gate` | `defer/insufficient` | 0 |
| `stage_a::000019` | `verify/insufficient` | `flag/invalid_value` | `verify/insufficient` | `verify/insufficient` | `evidence_gate_override` | `verify/insufficient` | 1 |
| `stage_a::000019` | `verify/insufficient` | `flag/invalid_value` | `verify/insufficient` | `verify/insufficient` | `hybrid_evidence_then_train_gate` | `verify/insufficient` | 1 |
| `stage_a::000021` | `flag/invalid_value` | `flag/invalid_value` | `ground/supported` | `flag/invalid_value` | `raw_candidate_top1` | `flag/invalid_value` | 1 |
| `stage_a::000021` | `flag/invalid_value` | `flag/invalid_value` | `ground/supported` | `flag/invalid_value` | `calibrated_candidate_top1` | `ground/supported` | 0 |
| `stage_a::000021` | `flag/invalid_value` | `flag/invalid_value` | `ground/supported` | `flag/invalid_value` | `train_selected_score_gap_gate` | `defer/insufficient` | 0 |
| `stage_a::000021` | `flag/invalid_value` | `flag/invalid_value` | `ground/supported` | `flag/invalid_value` | `evidence_gate_override` | `flag/invalid_value` | 1 |
| `stage_a::000021` | `flag/invalid_value` | `flag/invalid_value` | `ground/supported` | `flag/invalid_value` | `hybrid_evidence_then_train_gate` | `flag/invalid_value` | 1 |

## Interpretation

If evidence-gate or hybrid policies beat raw/calibrated candidate top-1, the system should keep runtime evidence arbitration as the baseline before any DPO/RLVR or release escalation.

Keep tool_query, DPO/RLVR, Hugging Face publication, release tagging, and broad retraining gated until model-heavy outputs beat this runtime arbitration baseline on held-out cases.

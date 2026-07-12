# Stage A Saved-Output Meet-Or-Beat Gate

Purpose: turn the saved-output next-decision checkpoint into a reusable
acceptance gate for future model-heavy Cayuga outputs.

## Requirements

- Selected next step: `meet_or_beat_runtime_evidence_arbitration_baseline`
- Candidate/model exact minimum: 4
- Trusted candidate incorrect maximum: 0
- Hidden labels used by arbitration: `False`
- Raw predictions remain uncommitted: `True`

## Runtime Baseline

| Policy | Exact | Rows | Trusted candidate incorrect |
| --- | ---: | ---: | ---: |
| `evidence_gate_override` | 4 | 4 | 0 |
| `hybrid_evidence_then_train_gate` | 4 | 4 | 0 |

## Model Policies Under Test

| Policy | Exact | Rows | Trusted candidate incorrect | Passes gate | Violations |
| --- | ---: | ---: | ---: | ---: | --- |
| `raw_candidate_top1` | 1 | 4 | 3 | 0 | `["below_runtime_arbitration_exact_min", "unsafe_candidate_trust"]` |
| `calibrated_candidate_top1` | 2 | 4 | 2 | 0 | `["below_runtime_arbitration_exact_min", "unsafe_candidate_trust"]` |
| `train_selected_score_gap_gate` | 1 | 4 | 0 | 0 | `["below_runtime_arbitration_exact_min"]` |

## Future Policy Input Contract

- Required fields: `["dataset", "policy", "source_kind", "source_report", "source_report_sha256", "exact", "rows", "trusted_candidate_incorrect", "public_safety_contract"]`
- Required dataset: `negbiodb_ct_stage_a_saved_output_policy_summary_v1`
- Allowed source kinds: `["prediction-summary", "candidate-gate-summary", "candidate-arbitration-policy"]`
- Source provenance rules: `["source_report must be a repo-relative public manifest path", "source_report_sha256 must match the source_report contents", "release/public_release_manifest.json must mark source_report safe_to_publish with the same SHA-256"]`
- Public-safe flags: `["raw_model_outputs_used=false", "raw_run_folders_used=false", "raw_predictions_committed=false", "raw_candidate_scores_committed=false", "raw_eval_report_committed=false", "raw_scheduler_logs_committed=false", "model_state_committed=false"]`
- Numeric validity rules: `["exact, rows, trusted_candidate, and trusted_candidate_incorrect must be non-negative JSON integers, not strings, floats, or booleans", "exact <= rows", "trusted_candidate <= rows", "trusted_candidate_incorrect <= trusted_candidate", "rows must equal the runtime baseline rows for this gate"]`

## Decision

- Passes gate: `False`
- Gate violations: `["no_model_policy_meets_runtime_baseline"]`

Current raw/calibrated/score-gap candidate policies fail this gate; runtime evidence arbitration remains the baseline for the next Cayuga output.

Artifact policy: raw saved predictions, candidate-score JSONL, scheduler logs,
model state, and ignored run folders stay uncommitted.

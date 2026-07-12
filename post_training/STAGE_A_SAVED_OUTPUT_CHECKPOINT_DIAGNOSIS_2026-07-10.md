# Stage A Saved-Output Checkpoint Diagnosis

Purpose: compact public-safe status checkpoint for the current Cayuga saved-output path.

## Thesis

Biology tool-use agents need trainable trajectories plus runtime evidence arbitration; RLVR/DPO escalation is not justified until compact held-out checkpoint policies beat deterministic runtime baselines.

## Diagnosis

- Teacher-forced full-target margin: 4/4 held-out wins (base 0/4; trained mean margin 0.018087).
- Finite-candidate rank: 1/4 held-out top-1, with top-pair counts `{"flag/invalid_value": 4}`.
- Train-derived calibration: raw 1/4, calibrated 2/4, train-selected gate 1/4.
- Meet-or-beat gate passes: False with violations `["no_model_policy_meets_runtime_baseline"]`.

## Arbitration

| Policy | Exact | Rows | Trusted candidate | Trusted incorrect |
| --- | ---: | ---: | ---: | ---: |
| raw_candidate_top1 | 1 | 4 | 4 | 3 |
| calibrated_candidate_top1 | 2 | 4 | 4 | 2 |
| train_selected_score_gap_gate | 1 | 4 | 0 | 0 |
| evidence_gate_override | 4 | 4 | 0 | 0 |
| hybrid_evidence_then_train_gate | 4 | 4 | 0 | 0 |

## Decision

- Selected next step: `meet_or_beat_runtime_evidence_arbitration_baseline`
- Keep gated: `["tool_query", "DPO/RLVR", "Hugging Face publication", "release tagging", "broad retraining"]`
- Next research move: Diagnose per-case action/status discrimination under finite-candidate or runtime-enforced policies; do not treat teacher-forced margin repair as deployment readiness.

Public-safety contract: raw saved predictions, candidate-score JSONL, scheduler logs,
model state, and ignored run folders were not read for this checkpoint.

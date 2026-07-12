# Stage A Saved-Output Intake Contract

Purpose: verify that the compact saved-output checkpoint bundle is
internally consistent before any future Cayuga policy summary is
judged by the meet-or-beat gate.

## Contract

- Passes contract: `True`
- Violations: `[]`
- Selected next step: `meet_or_beat_runtime_evidence_arbitration_baseline`

## Required Artifacts

- Expected: `["compact saved-output summary", "compact candidate calibration summary", "compact candidate arbitration summary", "updated saved-output next-decision report"]`
- Observed: `["compact saved-output summary", "compact candidate calibration summary", "compact candidate arbitration summary", "updated saved-output next-decision report"]`

## Input Hashes

| Artifact | Path | Hash match |
| --- | --- | ---: |
| `readiness` | `post_training/stage_a_saved_prediction_readiness_2026-07-09.json` | `True` |
| `candidate_calibration` | `post_training/stage_a_saved_output_candidate_calibration_qwen05b_cayuga_summary_2026-07-10.json` | `True` |
| `candidate_arbitration` | `post_training/stage_a_saved_output_candidate_arbitration_2026-07-10.json` | `True` |
| `candidate_gate_1` | `post_training/stage_a_saved_candidate_gate_train_observed_qwen05b_2026-07-09.json` | `True` |
| `candidate_gate_2` | `post_training/stage_a_saved_candidate_gate_all_valid_qwen05b_2026-07-09.json` | `True` |
| `meet_or_beat.next_decision` | `post_training/stage_a_saved_output_next_decision_2026-07-10.json` | `True` |
| `meet_or_beat.candidate_arbitration` | `post_training/stage_a_saved_output_candidate_arbitration_2026-07-10.json` | `True` |

## Future Policy Flags

- Required public-safe flags: `["raw_model_outputs_used=false", "raw_run_folders_used=false", "raw_predictions_committed=false", "raw_candidate_scores_committed=false", "raw_eval_report_committed=false", "raw_scheduler_logs_committed=false", "model_state_committed=false"]`
- Missing public-safe flags: `[]`

## Future Policy Adapter Contract

- Required dataset: `negbiodb_ct_stage_a_saved_output_policy_summary_v1`
- Observed dataset: `negbiodb_ct_stage_a_saved_output_policy_summary_v1`
- Missing required fields: `[]`
- Missing source kinds: `[]`
- Missing source provenance rules: `[]`

## Decision

Use build_stage_a_saved_output_policy_summary.py for the next compact Cayuga policy summary, then pass it to the meet-or-beat gate before reopening tool_query, DPO/RLVR, HF publication, or release tagging.

Artifact policy: this verifier reads compact reports only; raw
prediction JSONL, candidate-score JSONL, scheduler logs, model state,
and ignored run folders remain out of scope.

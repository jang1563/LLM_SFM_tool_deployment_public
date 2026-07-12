# Stage A Saved-Output Next Checkpoint Spec

Purpose: define the next Cayuga saved-output checkpoint from compact public-safe diagnosis artifacts.

## Observed Failure

- Bottleneck: `candidate_selection_bias_flag_invalid_value_overselection`
- Held-out candidate top-1: 1/4, top-pair counts `{"flag/invalid_value": 4}`
- Field diagnostic: exact 1, action 1, status 1; patterns `{"both_field_failure": 3, "pair_top1": 1}`
- Train candidate bias: 4/16, top-pair counts `{"flag/invalid_value": 16}`

## Next Checkpoint

- Name: `balanced_nonflag_candidate_rank_readout`
- Runner: `post_training/run_stage_a_saved_output_calibration_margin_sft_cayuga.sbatch`
- Question: Does balancing non-flag target pairs reduce the learned global flag/invalid_value candidate prior while preserving the repaired full-target teacher-forced margins?

Environment overrides:

```bash
export RUN_ID='stage_a_saved_output_nonflag_candidate_rank_qwen05b_cayuga_${JOB_ID}'
export MODEL_ID='Qwen/Qwen2.5-0.5B-Instruct'
export MAX_STEPS='40'
export BATCH_SIZE='1'
export LR='1e-5'
export TRAIN_LAST_LAYERS='1'
export TARGET_FORMAT='full'
export SCORE_TARGET_FORMATS='full,action_status_only,action_only,status_only'
export FOCUS_CHOSEN_PAIRS='defer/insufficient,reject/contradicted,verify/insufficient'
export FOCUS_REPEAT='4'
export FOCUS_ONLY='0'
export PAIRWISE_MARGIN_WEIGHT='1'
export PAIRWISE_MARGIN='0.05'
export SCORE_BASE_MARGINS='1'
export SCORE_TRAIN_MARGINS='1'
export SCORE_BASE_CANDIDATES='1'
export SCORE_TRAIN_CANDIDATES='1'
export SCORE_TRAINED_CANDIDATES='1'
export CANDIDATE_POLICY='train_observed_plus_rejected'
export CANDIDATE_TARGET_FORMAT='full'
export ALLOW_DOWNLOAD='0'
```

Post-run compact summaries:

- `python post_training/analyze_stage_a_saved_output_candidate_calibration.py --train-candidates ${OUT_DIR}/train_candidates.jsonl --heldout-candidates ${OUT_DIR}/candidates.jsonl --out-json /tmp/stage_a_saved_output_nonflag_candidate_calibration.json --out-md /tmp/STAGE_A_SAVED_OUTPUT_NONFLAG_CANDIDATE_CALIBRATION.md`
- `python post_training/analyze_stage_a_saved_output_candidate_fields.py --candidates ${OUT_DIR}/candidates.jsonl --out-json /tmp/stage_a_saved_output_nonflag_candidate_fields.json --out-md /tmp/STAGE_A_SAVED_OUTPUT_NONFLAG_CANDIDATE_FIELDS.md`
- `python post_training/evaluate_stage_a_saved_output_candidate_arbitration.py --calibration-report /tmp/stage_a_saved_output_nonflag_candidate_calibration.json --out-json /tmp/stage_a_saved_output_nonflag_candidate_arbitration.json --out-md /tmp/STAGE_A_SAVED_OUTPUT_NONFLAG_CANDIDATE_ARBITRATION.md`

After the compact arbitration report is curated into the repo and listed
in `release/public_release_manifest.json`, run:

- `python post_training/build_stage_a_saved_output_policy_summary.py --source post_training/<curated_nonflag_candidate_arbitration>.json --source-kind candidate-arbitration-policy --policy calibrated_candidate_top1 --out /tmp/stage_a_saved_output_nonflag_policy_summary.json`
- `python post_training/evaluate_stage_a_saved_output_meet_or_beat_gate.py --model-policy-summary /tmp/stage_a_saved_output_nonflag_policy_summary.json --out-json /tmp/stage_a_saved_output_nonflag_meet_or_beat_gate.json --out-md /tmp/STAGE_A_SAVED_OUTPUT_NONFLAG_MEET_OR_BEAT_GATE.md`

## Acceptance Gate

- Exact minimum: 4/4
- Trusted candidate incorrect maximum: 0
- Hidden labels used by arbitration: `False`
- Raw predictions remain uncommitted: `True`

Keep tool_query, DPO/RLVR, Hugging Face publication, release tagging,
and broad retraining gated until this compact policy path meets or beats
runtime evidence arbitration.

Public-safety contract: raw saved predictions, candidate-score JSONL,
scheduler logs, model state, and ignored run folders are private run
outputs and are not public artifacts.

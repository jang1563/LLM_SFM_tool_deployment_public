# Stage A Saved-Output Candidate-CE Checkpoint Spec

Purpose: define the next Cayuga checkpoint after non-flag balancing failed.

## Observed Failure

- Bottleneck: `candidate_selection_not_repaired_by_nonflag_oversampling`
- Raw held-out candidate top-1: 1/4
- Calibrated held-out candidate top-1: 1/4
- Field diagnostic: exact 1/4, action 1/4, status 2/4
- Interpretation: Simple pair oversampling changed the candidate prior but did not teach evidence-conditioned action/status selection.

## Next Checkpoint

- Name: `candidate_ce_action_status_pair_field_readout`
- Runner: `post_training/run_stage_a_saved_output_calibration_margin_sft_cayuga.sbatch`
- Question: Does an explicit listwise candidate CE objective over action/status candidates beat simple oversampling while preserving teacher-forced margin repair?
- Objective change: Add supervised candidate-routing pressure: pair CE selects the exact action/status candidate and field CE selects the action and evidence_status marginals.

Dry-run preflight:

```bash
python post_training/run_stage_a_saved_output_calibration_margin_sft.py --dry-run --out-dir /tmp/stage_a_saved_output_candidate_ce_dry --run-id stage_a_saved_output_candidate_ce_dry --candidate-ce-weight 1 --candidate-ce-mode pair_plus_field --candidate-ce-logprob-mode mean --candidate-policy train_observed_plus_rejected --candidate-target-format action_status_only --score-base-margins --score-train-margins --score-base-candidates --score-train-candidates --score-trained-candidates
```

Environment overrides:

```bash
export RUN_ID='stage_a_saved_output_candidate_ce_pair_field_qwen05b_cayuga_${JOB_ID}'
export MODEL_ID='Qwen/Qwen2.5-0.5B-Instruct'
export MAX_STEPS='40'
export BATCH_SIZE='1'
export LR='1e-5'
export TRAIN_LAST_LAYERS='1'
export TARGET_FORMAT='full'
export SCORE_TARGET_FORMATS='full,action_status_only,action_only,status_only'
export FOCUS_CHOSEN_PAIRS=''
export FOCUS_REPEAT='1'
export FOCUS_ONLY='0'
export PAIRWISE_MARGIN_WEIGHT='1'
export PAIRWISE_MARGIN='0.05'
export CANDIDATE_CE_WEIGHT='1'
export CANDIDATE_CE_MODE='pair_plus_field'
export CANDIDATE_CE_LOGPROB_MODE='mean'
export SCORE_BASE_MARGINS='1'
export SCORE_TRAIN_MARGINS='1'
export SCORE_BASE_CANDIDATES='1'
export SCORE_TRAIN_CANDIDATES='1'
export SCORE_TRAINED_CANDIDATES='1'
export CANDIDATE_POLICY='train_observed_plus_rejected'
export CANDIDATE_TARGET_FORMAT='action_status_only'
export ALLOW_DOWNLOAD='0'
```

Post-run compact summaries:

- `python post_training/analyze_stage_a_saved_output_candidate_calibration.py --train-candidates ${OUT_DIR}/train_candidates.jsonl --heldout-candidates ${OUT_DIR}/candidates.jsonl --out-json /tmp/stage_a_saved_output_candidate_ce_calibration.json --out-md /tmp/STAGE_A_SAVED_OUTPUT_CANDIDATE_CE_CALIBRATION.md`
- `python post_training/analyze_stage_a_saved_output_candidate_fields.py --candidates ${OUT_DIR}/candidates.jsonl --out-json /tmp/stage_a_saved_output_candidate_ce_fields.json --out-md /tmp/STAGE_A_SAVED_OUTPUT_CANDIDATE_CE_FIELDS.md`
- `python post_training/evaluate_stage_a_saved_output_candidate_arbitration.py --calibration-report /tmp/stage_a_saved_output_candidate_ce_calibration.json --out-json /tmp/stage_a_saved_output_candidate_ce_arbitration.json --out-md /tmp/STAGE_A_SAVED_OUTPUT_CANDIDATE_CE_ARBITRATION.md`

## Acceptance Gate

- Exact minimum: 4/4
- Trusted candidate incorrect maximum: 0
- Hidden labels used by arbitration: `False`
- Raw predictions remain uncommitted: `True`

Keep tool_query, DPO/RLVR, Hugging Face publication, release tagging,
and broad retraining gated until this compact candidate-CE path meets
or beats runtime evidence arbitration.

Public-safety contract: raw saved predictions, candidate-score JSONL,
scheduler logs, model state, and ignored run folders are private run
outputs and are not public artifacts.

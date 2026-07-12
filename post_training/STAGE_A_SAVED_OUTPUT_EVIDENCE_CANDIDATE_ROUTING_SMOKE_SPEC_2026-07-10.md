# Stage A Saved-Output Evidence Candidate-Routing Smoke Spec

Purpose: define the next small Cayuga smoke contract from public-safe candidate-routing artifacts.

## Preconditions

- Rows: 25 total, 20 train, 5 held-out
- Runtime evidence gate: 5/5 held-out, 4/4 bridge-focus
- Best static prior: 1/5 held-out
- All preconditions pass: `True`

## Smoke Contract

- Name: `evidence_conditioned_candidate_routing_sft_smoke`
- Runner: `post_training/run_stage_a_saved_output_evidence_candidate_routing_smoke.py`
- Cayuga wrapper: `post_training/run_stage_a_saved_output_evidence_candidate_routing_smoke_cayuga.sbatch`
- Component: `saved_output_evidence_candidate_routing`
- Prompt contract: `stage_a_saved_output_evidence_candidate_routing_v1`
- Question: Can a small supervised model select the correct action/status candidate from prompt-visible evidence features, beating every static prior and matching runtime evidence routing on held-out and bridge-focus rows?
- Training rows: `post_training/stage_a_saved_output_evidence_candidate_routing_train_v1.jsonl`
- Held-out rows: `post_training/stage_a_saved_output_evidence_candidate_routing_heldout_v1.jsonl`

Environment overrides:

```bash
export RUN_ID='stage_a_evidence_candidate_routing_qwen05b_cayuga_${JOB_ID}'
export MODEL_ID='Qwen/Qwen2.5-0.5B-Instruct'
export MAX_STEPS='40'
export BATCH_SIZE='1'
export LR='1e-5'
export TRAIN_LAST_LAYERS='1'
export MAX_LENGTH='1536'
export TARGET_FORMAT='selected_pair_action_status_json'
export CANDIDATE_SCORE_MODE='finite_pair_mean_logprob'
export ALLOW_DOWNLOAD='0'
```

Dry-run preflight:

- `python post_training/export_stage_a_saved_output_evidence_candidate_routing_rows.py --rows-out /tmp/stage_a_saved_output_evidence_candidate_routing_rows_v1.jsonl --train-out /tmp/stage_a_saved_output_evidence_candidate_routing_train_v1.jsonl --heldout-out /tmp/stage_a_saved_output_evidence_candidate_routing_heldout_v1.jsonl --manifest-out /tmp/stage_a_saved_output_evidence_candidate_routing_manifest.json`
- `python post_training/evaluate_stage_a_saved_output_evidence_candidate_routing_readout.py --out-json /tmp/stage_a_saved_output_evidence_candidate_routing_readout.json --out-md /tmp/STAGE_A_SAVED_OUTPUT_EVIDENCE_CANDIDATE_ROUTING_READOUT.md`
- `python post_training/validate_post_training_data.py`

## Acceptance Gate

- Candidate model held-out exact minimum: 5/5
- Candidate model bridge-focus exact minimum: 4/4
- Static prior held-out exact to beat: 1/5
- Raw predictions remain uncommitted: `True`

## Decision

- Selected next step: `run_evidence_conditioned_candidate_routing_smoke_dry_run_on_cayuga`
- Runner implemented: `True`
- Ready for Cayuga dry-run: `True`
- Ready for Cayuga submission: `False`
- Ready for DPO/RLVR: `False`

The evidence-conditioned slice is ready for a small Cayuga dry-run using the implemented runner, but not for unapproved full submission, DPO/RLVR, tool_query, HF publication, or release tagging. A model policy must reach 5/5 held-out and 4/4 bridge-focus exact before escalation.

Public-safety contract: raw predictions, candidate-score JSONL,
scheduler logs, model state, and ignored run folders are private run
outputs and are not public artifacts.

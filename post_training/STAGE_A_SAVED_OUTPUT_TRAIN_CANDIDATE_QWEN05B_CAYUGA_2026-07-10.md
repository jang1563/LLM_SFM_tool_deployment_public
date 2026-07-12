# Stage A Saved-Output Train-Candidate Diagnostic

Purpose: check whether the saved-output finite-candidate failure is only a
held-out readout problem or is already visible in train-side candidate scores.

## Setup

- Run ID: `stage_a_saved_output_flag_full_train_candidates_qwen05b_cayuga_20260710`
- Cluster/job: Cayuga `3074169`
- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- Repo commit: `f3530e9`
- Training focus: `flag/invalid_value`, `focus_repeat=4`, `focus_only=true`
- Candidate policy: `train_observed_plus_rejected`
- Candidate target format: `full`

## Result

| Slice | Exact top-1 | Mean target rank | Top-pair counts |
| --- | ---: | ---: | --- |
| Base held-out | 0/4 | 3.5 | `ground/supported`: 4 |
| Trained train | 4/16 | 2.5 | `flag/invalid_value`: 16 |
| Trained held-out | 1/4 | 2.5 | `flag/invalid_value`: 4 |

The same run still repairs the teacher-forced full-output held-out margin:
base 0/4 margin wins to trained 4/4, with mean margin moving from `-0.081001`
to `0.018087`.

## Interpretation

This is negative for candidate readiness. The trained scorer over-selects
`flag` / `invalid_value` on both train and held-out candidate readouts. That
means the previous held-out failure is not just held-out score noise; the
focused objective creates a global `flag` / `invalid_value` prior in finite
candidate ranking.

Keep `tool_query`, DPO/RLVR, Hugging Face release, and release tagging gated.
The next research step should use train-side candidate scores to design a
candidate calibration or routing objective, then re-score held-out candidates
without tuning on held-out ranks.

Raw `train_candidates.jsonl`, `candidates.jsonl`, margins, scheduler logs, and
model state remain under ignored `post_training/runs/`.

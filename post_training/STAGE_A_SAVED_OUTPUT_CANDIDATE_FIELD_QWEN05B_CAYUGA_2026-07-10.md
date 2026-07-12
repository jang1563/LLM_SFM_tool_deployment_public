# Stage A Saved-Output Candidate Field Diagnostic

Purpose: diagnose the negative candidate-rank result after full-target
`flag` / `invalid_value` focused SFT.

## Run

- Source run:
  `stage_a_saved_output_flag_full_candidate_rank_qwen05b_cayuga_20260710`
- Slurm job: `[omitted]`
- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- Candidate policy: `train_observed_plus_rejected`
- Candidate target format: `full`
- Held-out rows: 4 evaluation-only probe pairs

## Field Result

| Readout | Base | Trained |
| --- | ---: | ---: |
| Candidate exact top-1 | 0/4 | 1/4 |
| Mean target rank | 3.5 | 2.5 |
| Mean action rank | 3.5 | 2.5 |
| Mean evidence-status rank | 2.75 | 2.0 |
| Dominant top pair | `ground` / `supported` in 4/4 | `flag` / `invalid_value` in 4/4 |
| Field-rank pattern | `both_field_failure` in 4/4 | `both_field_failure` in 3/4, `pair_top1` in 1/4 |

Trained row-level field ranks:

| Target pair | Top pair | Pair rank | Action rank | Status rank | Pattern |
| --- | --- | ---: | ---: | ---: | --- |
| `reject` / `contradicted` | `flag` / `invalid_value` | 4 | 4 | 3 | `both_field_failure` |
| `defer` / `insufficient` | `flag` / `invalid_value` | 3 | 3 | 2 | `both_field_failure` |
| `verify` / `insufficient` | `flag` / `invalid_value` | 2 | 2 | 2 | `both_field_failure` |
| `flag` / `invalid_value` | `flag` / `invalid_value` | 1 | 1 | 1 | `pair_top1` |

## Decision

The failure is not just one weak field. Full-target SFT moved the model from
`ground` / `supported` collapse to `flag` / `invalid_value` over-selection.
For the non-flag held-out targets, both the action field and evidence-status
field are misranked.

Keep `tool_query`, DPO/RLVR, Hugging Face publication, release tagging, and
broad retraining gated. The next useful step is calibration or a field-wise /
pair-wise routing objective, benchmarked against runtime enforcement.

Raw candidate-score JSONL, full field reports, trainable state, and scheduler
logs remain in ignored Cayuga `post_training/runs/` folders and are not part of
the public surface.

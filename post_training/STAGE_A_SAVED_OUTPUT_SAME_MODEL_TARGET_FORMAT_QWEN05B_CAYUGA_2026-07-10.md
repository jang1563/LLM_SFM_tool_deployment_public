# Stage A Saved-Output Same-Model Target-Format Result

Purpose: test whether a full-target-trained model contains recoverable
action/status preferences when scored with projected targets, and whether the
full JSON `flag` / `invalid_value` target can cross the
`ground` / `supported` boundary under a narrower `flag`-focused run.

## Run

- Run ID:
  `stage_a_saved_output_flag_full_same_model_multiformat_qwen05b_cayuga_20260710`
- Slurm job: `[omitted]`
- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- Training target format: `full`
- Extra scoring formats: `full`, `action_only`, `action_status_only`,
  `status_only`
- Training rows: 16 train-allowed probe pairs, expanded to 16 training
  exposures by `--focus-chosen-pairs flag/invalid_value --focus-repeat 4
  --focus-only`
- Held-out rows: 4 evaluation-only probe pairs
- Held-out rows used for training: `false`
- Steps: 40

## Held-Out Result

| Scoring target format | Base wins | Trained wins | Base mean margin | Trained mean margin | Base flag margin | Trained flag margin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `full` | 0/4 | 4/4 | -0.081001 | +0.018087 | -0.175447 | +0.025968 |
| `action_only` | 1/4 | 3/4 | -0.459920 | +0.399461 | -0.812072 | +0.832667 |
| `action_status_only` | 1/4 | 4/4 | -0.141241 | +0.458662 | -0.282855 | +0.799087 |
| `status_only` | 1/4 | 4/4 | +0.021075 | +0.378879 | +0.303895 | +0.780488 |

Train split after SFT: `full` 16/16 wins, `action_status_only` 16/16,
`status_only` 16/16, and `action_only` 12/16.

## Decision

This is the first saved-output calibration run where the full JSON
`flag` / `invalid_value` held-out target crosses above `ground` / `supported`.
It is still a tiny teacher-forced diagnostic, not deployment readiness and not a
reason to start DPO/RLVR or `tool_query`.

Next, test whether the repaired full target survives finite-candidate ranking
or a fail-closed gate. Runtime enforcement remains the deployment baseline
until candidate/top-1 and full-trajectory checks improve.

Raw margin JSONL, full reports, trainable state, and scheduler logs remain in
ignored Cayuga `post_training/runs/` folders and are not part of the public
surface.

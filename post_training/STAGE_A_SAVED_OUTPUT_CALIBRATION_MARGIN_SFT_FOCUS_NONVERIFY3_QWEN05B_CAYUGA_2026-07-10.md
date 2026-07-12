# Stage A Saved-Output Focused Margin SFT Cayuga Result

Purpose: test whether narrow corrective SFT aimed at the three still-negative
target families can move held-out target outputs above the observed
`ground` / `supported` collapse without using held-out probe rows for training.

## Run

- Run ID:
  `stage_a_saved_output_calibration_margin_sft_focus_nonverify3_qwen05b_cayuga_20260710`
- Slurm job: `[omitted]`
- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- Training rows: 16 train-allowed probe pairs, expanded to 52 training
  exposures by repeating `defer` / `insufficient`,
  `flag` / `invalid_value`, and `reject` / `contradicted`
- Held-out rows: 4 evaluation-only probe pairs
- Held-out rows used for training: `false`
- Objective: chosen target output CE plus pairwise margin loss against
  rejected `ground` / `supported`
- Steps: 40
- Trainable parameters: 14,913,280

## Result

| Split | Base wins | Unfocused SFT wins | Focused SFT wins | Mean base margin | Mean focused margin | Mean focused delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Held-out probe | 0/4 | 1/4 | 3/4 | -0.081001 | -0.002999 | +0.078001 |

Held-out family movement:

| Target pair | Base win | Focused SFT win | Focused margin delta |
| --- | ---: | ---: | ---: |
| `defer` / `insufficient` | 0 | 1 | +0.092604 |
| `flag` / `invalid_value` | 0 | 0 | +0.072349 |
| `reject` / `contradicted` | 0 | 1 | +0.075966 |
| `verify` / `insufficient` | 0 | 1 | +0.071086 |

Train split after focused SFT: 12/16 margin wins. `flag` / `invalid_value`
remains below `ground` / `supported` on both train and held-out scoring.

## Decision

This is a stronger corrective signal than the unfocused run, but still not a
repair. Keep `tool_query`, DPO/RLVR, Hugging Face publication, release tagging,
and broad retraining gated. The next scientific step should isolate the
`flag` action / `invalid_value` failure with a narrower action-specific target
or keep runtime fail-closed enforcement as the deployment baseline.

Raw margin JSONL, full reports, trainable state, and scheduler logs remain in
the ignored Cayuga `post_training/runs/` path and are not part of the public
surface.

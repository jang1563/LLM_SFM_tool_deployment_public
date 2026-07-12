# Stage A Saved-Output Calibration Margin SFT Cayuga Result

Purpose: test whether narrow train-only SFT plus pairwise margin loss can move
target action/status outputs above the observed `ground` / `supported`
collapse on the saved-output calibration probe.

## Run

- Run ID: `stage_a_saved_output_calibration_margin_sft_qwen05b_cayuga_20260710`
- Slurm job: `[omitted]`
- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- Training rows: 16 train-allowed probe pairs
- Held-out rows: 4 evaluation-only probe pairs
- Held-out rows used for training: `false`
- Objective: chosen target output CE plus pairwise margin loss against
  rejected `ground` / `supported`
- Steps: 20
- Trainable parameters: 14,913,280

## Result

| Split | Base wins | Trained wins | Mean base margin | Mean trained margin | Mean delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| Held-out probe | 0/4 | 1/4 | -0.081001 | -0.045619 | +0.035382 |

Held-out family movement:

| Target pair | Base win | Trained win | Margin delta |
| --- | ---: | ---: | ---: |
| `defer` / `insufficient` | 0 | 0 | +0.044333 |
| `flag` / `invalid_value` | 0 | 0 | +0.022366 |
| `reject` / `contradicted` | 0 | 0 | +0.037846 |
| `verify` / `insufficient` | 0 | 1 | +0.036981 |

Train split after SFT: 4/16 margin wins. The only consistently won family is
`verify` / `insufficient`; the other three families remain below
`ground` / `supported`.

## Decision

This is partial movement, not repair. Keep `tool_query`, DPO/RLVR, Hugging Face
publication, release tagging, and broad retraining gated. The current evidence
supports either narrower corrective supervision or runtime fail-closed
enforcement before any escalation.

Raw margin JSONL, full reports, trainable state, and scheduler logs remain in
the ignored Cayuga `post_training/runs/` path and are not part of the public
surface.

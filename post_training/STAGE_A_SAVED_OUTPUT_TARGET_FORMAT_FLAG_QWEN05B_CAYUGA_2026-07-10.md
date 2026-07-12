# Stage A Saved-Output Target-Format Flag Diagnostic

Purpose: isolate the remaining `flag` / `invalid_value` failure after focused
full-JSON margin SFT improved aggregate held-out margins but still left the
invalid-value case below `ground` / `supported`.

## Runs

All runs used `Qwen/Qwen2.5-0.5B-Instruct` on Cayuga, trained only on the
16 train-allowed calibration probe pairs, expanded to 16 training exposures by
`--focus-chosen-pairs flag/invalid_value --focus-repeat 4 --focus-only`.
The 4 held-out probe rows stayed evaluation-only.

| Target format | Slurm job | Base held-out wins | Trained held-out wins | Base flag margin | Trained flag margin |
| --- | ---: | ---: | ---: | ---: | ---: |
| `action_only` | [omitted] | 1/4 | 3/4 | -0.812072 | +0.842364 |
| `status_only` | [omitted] | 1/4 | 1/4 | +0.303895 | +0.626285 |
| `action_status_only` | [omitted] | 1/4 | 2/4 | -0.282855 | +0.436716 |
| `full` focused reference | [omitted] | 0/4 | 3/4 | -0.175447 | -0.103098 |

## Interpretation

The isolated `invalid_value` status is already preferred over `supported` in
the base model on the held-out invalid-value row. The `flag` action alone is
not initially preferred, but a narrow train-only action projection flips it
strongly. The combined `flag` + `invalid_value` projection also flips.

The remaining failure is therefore not simply the `flag` token or the
`invalid_value` status token. The still-negative full-JSON focused result points
to full target coupling: citation list, tool-call JSON, rationale, or the longer
action/status/citation/tool object competing against the shorter
`ground` / `supported` collapse.

## Decision

Keep `tool_query`, DPO/RLVR, Hugging Face publication, release tagging, and
broad retraining gated. The next scientific step should test candidate scoring
or length/field-normalized scoring for the full saved-output target before
changing optimizers. Runtime fail-closed enforcement remains the deployment
baseline.

Raw margin JSONL, full reports, trainable state, and scheduler logs remain in
ignored Cayuga `post_training/runs/` folders and are not part of the public
surface.

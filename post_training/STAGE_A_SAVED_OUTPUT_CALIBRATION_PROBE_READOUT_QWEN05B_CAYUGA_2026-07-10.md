# Stage A Saved-Output Calibration Probe Readout

Purpose: record the compact Cayuga result for the target-vs-ground/supported calibration probe without committing raw readouts, prompts, model text, scheduler logs, or model state.

## Summary

- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- Compute: Cayuga, job `[omitted]`
- Rows: 20 total, 16 train, 4 held-out
- Train exact top-1: 0/16
- Held-out exact top-1: 0/4
- Train top pairs: `{"ground/supported": 16}`
- Held-out top pairs: `{"ground/supported": 4}`
- Train mean chosen-minus-collapse margin: `-0.077454`
- Held-out mean chosen-minus-collapse margin: `-0.081001`

## Gate

- Train-selected zero-unsafe default threshold: `None`
- Adaptive train zero-unsafe threshold: `0.189912`
- Held-out strict final at adaptive threshold: 1/4
- Held-out trusted incorrect at adaptive threshold: 0

## Decision

The model still collapses to `ground` / `supported` on every probe row. Keep `tool_query`, DPO/RLVR, Hugging Face publication, release tagging, and broad retraining gated. The next repair should directly target candidate calibration or action/status supervision before any full saved-output readiness claim.

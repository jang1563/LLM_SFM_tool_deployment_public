# Stage A Saved-Candidate Gate Diagnostic

Purpose: test whether finite-candidate saved-prediction score gaps can
support a fail-closed boundary gate without publishing prompts, raw
candidate-score JSONL, model state, or scheduler logs.

## Summary

- Run ID: `stage_a_saved_candidate_readout_qwen05b_train_observed_2026_07_09`
- Candidate policy: `train_observed_pairs`
- Cases: 5
- Exact top-1: 1/5
- Mean target rank: 3
- Mean top-second gap: 0.032244
- Top pair counts: `{"ground/supported": 5}`
- Fail-closed pair: `defer/insufficient`

## Gate

- Best default zero-unsafe threshold: `0.035`
- Trusted at best default threshold: 1
- Trusted incorrect at best default threshold: 0
- Strict final correct at best default threshold: 2/5
- Adaptive zero-unsafe threshold: `0.033997`
- Adaptive trusted: 1
- Adaptive strict final correct: 2/5

## Decision

This is a fail-closed diagnostic over saved candidate scores only. It
does not reopen `tool_query`, DPO/RLVR, Hugging Face publication, or
release tagging.

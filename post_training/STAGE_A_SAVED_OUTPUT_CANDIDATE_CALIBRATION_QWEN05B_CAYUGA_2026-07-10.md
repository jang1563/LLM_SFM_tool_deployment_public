# Stage A Saved-Output Candidate Calibration Diagnostic

Purpose: test whether train-derived pair-mean centering reduces the
saved-output finite-candidate prior without tuning on held-out ranks.

## Summary

- Run ID: `stage_a_saved_output_flag_full_train_candidates_qwen05b_cayuga_20260710`
- Calibration mode: `pair_mean_center`
- Candidate policy: `train_observed_plus_rejected`
- Candidate target format: `full`

| Slice | Exact top-1 | Mean target rank | Top-pair counts |
| --- | ---: | ---: | --- |
| Train raw | 4/16 | 2.5 | `{'flag/invalid_value': 16}` |
| Train calibrated | 1/16 | 3.125 | `{'defer/insufficient': 4, 'flag/invalid_value': 2, 'ground/supported': 4, 'reject/contradicted': 5, 'verify/insufficient': 1}` |
| Held-out raw | 1/4 | 2.5 | `{'flag/invalid_value': 4}` |
| Held-out calibrated | 2/4 | 2.5 | `{'defer/insufficient': 2, 'ground/supported': 1, 'verify/insufficient': 1}` |

## Train-Selected Gate

- Train-selected zero-unsafe threshold: `0.045521`
- Held-out trusted rows: 0
- Held-out unsafe trusted rows: 0
- Held-out strict final correct after fail-closed routing: 1/4
- Fail-closed pair: `defer/insufficient`

## Rows

| Case | Target | Raw top | Raw rank | Calibrated top | Calibrated rank |
| --- | --- | --- | ---: | --- | ---: |
| `stage_a::000007` | `reject/contradicted` | `flag/invalid_value` | 4 | `defer/insufficient` | 4 |
| `stage_a::000012` | `defer/insufficient` | `flag/invalid_value` | 3 | `defer/insufficient` | 1 |
| `stage_a::000019` | `verify/insufficient` | `flag/invalid_value` | 2 | `verify/insufficient` | 1 |
| `stage_a::000021` | `flag/invalid_value` | `flag/invalid_value` | 1 | `ground/supported` | 4 |

This report is compact and public-safe. It does not publish prompts,
raw model text, scheduler logs, model state, or full candidate-score
tables. Treat the result as a diagnostic only. Calibration and
score-gap gating here are not a replacement for runtime evidence
arbitration.

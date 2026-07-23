# Stage A Prospective Real-Query Runtime-Hybrid Evaluation

Scope: public development cases with actual model-visible query IDs and
synthetic tool-result perturbations. This is not a new sealed test or
a live-tool execution result.

## Strategy Summary

| Strategy | Exact | Macro pair accuracy | Unsafe ground | Coverage | Selective risk |
| --- | ---: | ---: | ---: | ---: | ---: |
| `trust_all` | 5/180 | 0.200 | 175 | 1.000 | 0.972 |
| `best_static_pair` | 80/180 | 0.200 | 0 | 0.000 | 0.000 |
| `deterministic_gate` | 180/180 | 1.000 | 0 | 0.361 | 0.000 |
| `frozen_model` | 35/180 | 0.200 | 0 | 0.000 | 0.000 |
| `runtime_hybrid` | 115/180 | 0.400 | 0 | 0.000 | 0.000 |

## Boundary

- Actual drug/condition identifier values are model-visible.
- Tool-result states and perturbations are synthetic and deterministic.
- Completed sealed rows are not loaded, rescored, or used for selection.
- DPO, RLVR, and Hugging Face publication remain closed.

# Stage A Routing Gate Baseline Comparison

Purpose: compare no-model routing component baselines against the
all-family runtime evidence gate on public Stage A evidence-conditioned
routing rows.

## Summary

| Policy | All | Train | Held-out | Unsafe ground/supported overrides (all) | Citation mismatches (all) |
| --- | --- | --- | --- | ---: | ---: |
| `runtime_evidence_gate` | 25/25 exact; 25/25 gate-agree; mean 1.000 | 20/20 exact; 20/20 gate-agree; mean 1.000 | 5/5 exact; 5/5 gate-agree; mean 1.000 | 0 | 0 |
| `oracle` | 25/25 exact; 25/25 gate-agree; mean 1.000 | 20/20 exact; 20/20 gate-agree; mean 1.000 | 5/5 exact; 5/5 gate-agree; mean 1.000 | 0 | 0 |
| `majority_ground_supported` | 5/25 exact; 5/25 gate-agree; mean 0.800 | 4/20 exact; 4/20 gate-agree; mean 0.800 | 1/5 exact; 1/5 gate-agree; mean 0.800 | 20 | 0 |
| `routing_no_citations` | 15/25 exact; 15/25 gate-agree; mean 0.900 | 12/20 exact; 12/20 gate-agree; mean 0.900 | 3/5 exact; 3/5 gate-agree; mean 0.900 | 0 | 10 |
| `empty_object` | 0/25 exact; 0/25 gate-agree; mean 0.250 | 0/20 exact; 0/20 gate-agree; mean 0.250 | 0/5 exact; 0/5 gate-agree; mean 0.250 | 0 | 10 |

## Interpretation

The runtime gate and oracle are sanity baselines. A model prediction path should not advance to tool_query, DPO/RLVR, HF publication, or release tagging until it beats collapse and citationless baselines and is competitive with the runtime gate.

Compare saved model/component outputs against this report. If the model only matches action/status but misses citations or fail-closed routing, keep runtime enforcement in the system and avoid new optimization objectives.

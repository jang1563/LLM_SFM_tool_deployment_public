# Stage A Full-Trajectory Arbitration

Purpose: project the routing runtime gate into the canonical Stage A
trajectory evaluator and compare oracle, collapse, citationless,
runtime-gate, and hybrid runtime-over-model policies.

## Summary

| Policy | All | Train | Held-out | Unsafe ground/supported overrides (all) |
| --- | --- | --- | --- | ---: |
| `oracle_full` | 25/25 pass; mean 1.000 | 20/20 pass; mean 1.000 | 5/5 pass; mean 1.000 | 0 |
| `self_answer` | 0/25 pass; mean 0.229 | 0/20 pass; mean 0.229 | 0/5 pass; mean 0.229 | 0 |
| `ground_supported_collapse` | 5/25 pass; mean 0.714 | 4/20 pass; mean 0.714 | 1/5 pass; mean 0.714 | 20 |
| `citationless_runtime_action` | 15/25 pass; mean 0.943 | 12/20 pass; mean 0.943 | 3/5 pass; mean 0.943 | 0 |
| `runtime_gate_full` | 25/25 pass; mean 1.000 | 20/20 pass; mean 1.000 | 5/5 pass; mean 1.000 | 0 |
| `hybrid_runtime_over_collapse` | 25/25 pass; mean 1.000 | 20/20 pass; mean 1.000 | 5/5 pass; mean 1.000 | 0 |

## Interpretation

Runtime arbitration may rescue model-like collapse only when the final action, evidence status, citation packet, and tool query trajectory all pass the canonical evaluator.

Use this scaffold for saved model outputs next. Keep tool_query, DPO/RLVR, Hugging Face publication, and release tagging gated until real saved trajectories beat citationless/collapse baselines and approach runtime-gate full-trajectory performance.

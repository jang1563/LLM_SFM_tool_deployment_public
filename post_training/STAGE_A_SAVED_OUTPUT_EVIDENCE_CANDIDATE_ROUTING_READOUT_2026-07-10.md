# Stage A Saved-Output Evidence Candidate-Routing Readout

Purpose: score the evidence-conditioned candidate-routing rows before any model-heavy checkpoint.

## Summary

- Runtime evidence gate held-out exact: 5/5
- Runtime evidence gate bridge-focus exact: 4/4
- Best static prior held-out exact: 1/5
- Passes no-model readout: `True`

## Held-Out Policies

| Policy | Exact | Bridge-focus exact | Predicted pairs |
| --- | ---: | ---: | --- |
| `runtime_evidence_gate` | 5/5 | 4/4 | `{"defer/insufficient": 1, "flag/invalid_value": 1, "ground/supported": 1, "reject/contradicted": 1, "verify/insufficient": 1}` |
| `static_ground_supported` | 1/5 | 0/4 | `{"ground/supported": 5}` |
| `static_reject_contradicted` | 1/5 | 1/4 | `{"reject/contradicted": 5}` |
| `static_defer_insufficient` | 1/5 | 1/4 | `{"defer/insufficient": 5}` |
| `static_verify_insufficient` | 1/5 | 1/4 | `{"verify/insufficient": 5}` |
| `static_flag_invalid_value` | 1/5 | 1/4 | `{"flag/invalid_value": 5}` |

## Decision

- Selected next step: `prepare_evidence_conditioned_candidate_routing_smoke_spec`
- Ready for model-heavy candidate smoke spec: `True`
- Ready for DPO/RLVR: `False`

The substrate has a deterministic visible-evidence baseline at 5/5 held-out and 4/4 bridge-focus exact, while every static single-pair prior is at most 1/5 held-out. The next step may be a small candidate-routing smoke spec, not DPO/RLVR or tool_query.

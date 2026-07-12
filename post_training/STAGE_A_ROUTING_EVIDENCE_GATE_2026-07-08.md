# Stage A Routing Evidence Gate

Purpose: evaluate a no-model runtime gate for all Stage A
`routing_after_loop` evidence-conditioned rows using only model-visible
tool-result fields.

## Summary

- Gate: `deterministic_model_visible_evidence_routing_gate`
- Hidden labels used by gate: `False`
- All rows exact: 25/25
- Train rows exact: 20/20
- Held-out rows exact: 5/5
- Held-out action/status exact: 5/5
- Predicted pairs: `{"defer/insufficient": 5, "flag/invalid_value": 5, "ground/supported": 5, "reject/contradicted": 5, "verify/insufficient": 5}`

## Held-Out Rows

| Case family | Expected | Predicted | Exact | Reason |
| --- | --- | --- | ---: | --- |
| supported_negative_evidence | `ground/supported` | `ground/supported` | 1 | `same_indication_failure_record_found` |
| contradicted_or_mixed_endpoint_claim | `reject/contradicted` | `reject/contradicted` | 1 | `mixed_endpoint_records_for_same_claim` |
| insufficient_evidence | `defer/insufficient` | `defer/insufficient` | 1 | `no_same_indication_or_related_failure_record` |
| related_evidence_requires_verification | `verify/insufficient` | `verify/insufficient` | 1 | `related_evidence_without_same_indication_record` |
| invalid_value_attribution_failure | `flag/invalid_value` | `flag/invalid_value` | 1 | `invalid_numeric_value_in_same_indication_record` |

## Interpretation

A passing no-model gate is a runtime baseline to beat. It does not prove model competence, and should not be used as a reason to start DPO/RLVR.

Compare model outputs against this gate and keep tool_query, DPO/RLVR, and Hugging Face publication gated until model-heavy experiments beat the runtime baseline on broader held-out slices.

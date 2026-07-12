# Stage A Routing Evidence Boundary Gate

Purpose: evaluate a no-model runtime gate for the defer-vs-verify
routing boundary using only model-visible tool-result fields.

## Summary

- Gate: `deterministic_model_visible_evidence_boundary`
- Hidden labels used by gate: `False`
- All rows: 10/10
- Train rows: 8/8
- Held-out rows: 2/2
- Predicted pairs: `{"defer/insufficient": 5, "verify/insufficient": 5}`
- Reasons: `{"no_same_indication_or_related_failure_record": 5, "related_evidence_without_same_indication_record": 5}`

## Held-Out Rows

| Case family | Expected | Predicted | Exact | Reason | Related count | Completeness |
| --- | --- | --- | ---: | --- | ---: | --- |
| insufficient_evidence | `defer/insufficient` | `defer/insufficient` | 1 | `no_same_indication_or_related_failure_record` | 0 | `no_same_indication_or_related_failure_record` |
| related_evidence_requires_verification | `verify/insufficient` | `verify/insufficient` | 1 | `related_evidence_without_same_indication_record` | 2 | `related_evidence_exists_but_same_indication_record_absent` |

## Interpretation

This is a no-model baseline over a narrow boundary slice. A pass supports runtime enforcement as a useful system component, not a claim that model routing or calibration is solved.

Use this as the baseline to beat before new training. Keep tool_query, DPO/RLVR, and Hugging Face publication gated.

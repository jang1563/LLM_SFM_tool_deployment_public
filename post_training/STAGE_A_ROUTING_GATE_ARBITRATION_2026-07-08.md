# Stage A Routing Gate Arbitration

Purpose: compare raw candidate top-1, score-gap fail-closed routing,
and deterministic evidence-boundary override on the defer-vs-verify
held-out boundary slice.

## Summary

- Cases: 2
- Score-gap threshold: 0.025
- Best policy names: `["score_gap_fail_closed", "evidence_boundary_override", "hybrid_evidence_then_score_gap"]`

| Policy | Exact | Rows | Accuracy | Error case IDs |
| --- | ---: | ---: | ---: | --- |
| raw_candidate_top1 | 1 | 2 | 0.5 | `["stage_a::000012"]` |
| score_gap_fail_closed | 2 | 2 | 1.0 | `[]` |
| evidence_boundary_override | 2 | 2 | 1.0 | `[]` |
| hybrid_evidence_then_score_gap | 2 | 2 | 1.0 | `[]` |

## Held-Out Rows

| Case ID | Expected | Candidate top | Evidence gate | Policy | Predicted | Exact | Reason |
| --- | --- | --- | --- | --- | --- | ---: | --- |
| stage_a::000012 | `defer/insufficient` | `verify/insufficient` | `defer/insufficient` | `raw_candidate_top1` | `verify/insufficient` | 0 | `trust_candidate_top1` |
| stage_a::000012 | `defer/insufficient` | `verify/insufficient` | `defer/insufficient` | `score_gap_fail_closed` | `defer/insufficient` | 1 | `fail_closed_low_gap` |
| stage_a::000012 | `defer/insufficient` | `verify/insufficient` | `defer/insufficient` | `evidence_boundary_override` | `defer/insufficient` | 1 | `use_evidence_boundary_gate` |
| stage_a::000012 | `defer/insufficient` | `verify/insufficient` | `defer/insufficient` | `hybrid_evidence_then_score_gap` | `defer/insufficient` | 1 | `use_evidence_boundary_gate` |
| stage_a::000019 | `verify/insufficient` | `verify/insufficient` | `verify/insufficient` | `raw_candidate_top1` | `verify/insufficient` | 1 | `trust_candidate_top1` |
| stage_a::000019 | `verify/insufficient` | `verify/insufficient` | `verify/insufficient` | `score_gap_fail_closed` | `verify/insufficient` | 1 | `trust_high_gap_candidate` |
| stage_a::000019 | `verify/insufficient` | `verify/insufficient` | `verify/insufficient` | `evidence_boundary_override` | `verify/insufficient` | 1 | `use_evidence_boundary_gate` |
| stage_a::000019 | `verify/insufficient` | `verify/insufficient` | `verify/insufficient` | `hybrid_evidence_then_score_gap` | `verify/insufficient` | 1 | `use_evidence_boundary_gate` |

## Interpretation

If evidence-boundary or hybrid policies dominate raw top-1 on the held-out boundary slice, the next system design should route through runtime enforcement before adding new training objectives.

Keep tool_query, DPO/RLVR, and Hugging Face publication gated. Use arbitration as a public-safe system baseline before more model-heavy experiments.

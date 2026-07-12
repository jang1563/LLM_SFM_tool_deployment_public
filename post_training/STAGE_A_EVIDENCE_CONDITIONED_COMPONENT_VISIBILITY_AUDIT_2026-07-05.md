# Stage A Component Visibility Audit

Purpose: check whether component prompts expose the evidence
needed for action/evidence-status routing, without publishing claims,
source IDs, raw tool outputs, or hidden labels.

## Summary

- Rows audited: 75
- Underdetermined evidence-routing rows: 0
- Hidden-label leak rows: 0
- Components with underdetermined routing: `[]`

## By Component

| Component | Rows | Requires evidence target | Has evidence content | Observed loop | Loop has results | Underdetermined | Hidden-label leaks |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| enum_action | 25 | 25 | 25 | 0 | 0 | 0 | 0 |
| routing_after_loop | 25 | 25 | 25 | 25 | 25 | 0 | 0 |
| tool_query | 25 | 0 | 0 | 0 | 0 | 0 | 0 |

## Interpretation

If enum/routing targets require action plus evidence_status but the prompt lacks tool results or evidence content, further enum loss shaping is not a clean repair; the substrate should expose evidence-conditioned state before moving to tool_query, DPO, or RLVR.

This is a data-interface audit, not a model score, DPO/RLVR reward, or full trajectory result.

## Trace

- Input targets SHA-256: `cca947ae8525f75178ba6bae4aa2221733bf34a2587d8f6f600bcd709eb87f9a`

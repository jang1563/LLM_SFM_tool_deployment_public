# Stage A Component Visibility Audit

Purpose: check whether strict component prompts expose the evidence
needed for action/evidence-status routing, without publishing claims,
source IDs, raw tool outputs, or hidden labels.

## Summary

- Rows audited: 75
- Underdetermined evidence-routing rows: 50
- Hidden-label leak rows: 0
- Components with underdetermined routing: `["enum_action", "routing_after_loop"]`

## By Component

| Component | Rows | Requires evidence target | Has evidence content | Observed loop | Loop has results | Underdetermined | Hidden-label leaks |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| enum_action | 25 | 25 | 0 | 0 | 0 | 25 | 0 |
| routing_after_loop | 25 | 25 | 0 | 25 | 0 | 25 | 0 |
| tool_query | 25 | 0 | 0 | 0 | 0 | 0 | 0 |

## Interpretation

If enum/routing targets require action plus evidence_status but the prompt lacks tool results or evidence content, further enum loss shaping is not a clean repair; the substrate should expose evidence-conditioned state before moving to tool_query, DPO, or RLVR.

This is a data-interface audit, not a model score, DPO/RLVR reward, or full trajectory result.

## Trace

- Input targets SHA-256: `8a51b7dbc6da75639a864d6f366415caa9af14bf30cb3fece87d3e42573d8136`

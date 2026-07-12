# Stage A Routing Model Readiness

Purpose: compare compact Cayuga routing summaries against the public
runtime/baseline scorecard before reopening `tool_query`, DPO/RLVR,
Hugging Face publication, or release tagging.

## Baselines

- Runtime gate held-out exact: 5/5
- Collapse held-out exact: 1/5
- Citationless held-out exact: 3/5

## All-Family Model Readouts

| Readout | Exact | Mean score | Beats collapse | Beats citationless | Competitive with runtime gate |
| --- | ---: | ---: | --- | --- | --- |
| `freeform_routing_after_loop` | 0/5 | 0.200 | False | False | False |
| `constrained_routing_observed_pair` | 2/5 | 0.850 | True | False | False |

## Targeted Diagnostics

| Diagnostic | Result | Scope note |
| --- | --- | --- |
| `routing_contrast_candidate_subset` | 2/3 exact top-1 | Targeted 3-case subset; not all-family readiness evidence |
| `defer_verify_candidate_subset` | 1/2 exact top-1 | Targeted 2-case boundary; not all-family readiness evidence |
| `defer_verify_fail_closed_gate_subset` | 2/2 strict final correct | Useful boundary gate diagnostic; not all-family model readiness evidence |

## Decision

- Ready for `tool_query`: `False`
- Ready for DPO/RLVR: `False`
- Runtime enforcement required: `True`

Blockers:
- best all-family model readout is 2/5, below citationless routing at 3/5
- best all-family model readout is below runtime evidence gate at 5/5
- free-form routing is 0/5 and does not beat collapse baseline
- score-gap fail-closed result is only a two-case boundary diagnostic

Keep tool_query, DPO/RLVR, Hugging Face publication, and release tagging gated. Next compare saved model/component outputs against the all-family runtime gate or wrap the gate into full trajectory arbitration before adding optimization objectives.

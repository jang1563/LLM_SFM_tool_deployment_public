# Stage A Evidence Candidate-Routing Smoke Result

Purpose: summarize a compact smoke eval report without publishing raw candidate scores.

## Summary

- Run ID: `stage_a_evidence_candidate_routing_qwen05b_cayuga_20260710`
- Train exact: 4/20
- Held-out exact: 1/5
- Bridge-focus exact: 1/4
- Passes gate: `False`
- Gate violations: `["below_heldout_exact_min", "below_bridge_focus_exact_min", "does_not_beat_static_prior"]`

## Gate

- Held-out exact minimum: 5/5
- Bridge-focus exact minimum: 4/4
- Static prior held-out exact to beat: 1/5

Public-safety contract: this adapter reads compact eval reports only;
raw candidate-score JSONL, raw model text, scheduler logs, model state,
and ignored run folders are not public artifacts.

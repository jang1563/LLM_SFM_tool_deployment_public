# Stage A Enum-Action Candidate And Field Rank Diagnostic

Purpose: compactly diagnose whether the finite enum candidate scorer ranks
the gold `(action, evidence_status)` pair and its component fields near the
top without publishing
raw prompts, logs, model state, or full candidate-score tables.

## Summary

- Run ID: `stage_a_component_enum_action_candidate_fullrank_cayuga_2026_07_05`
- Candidate policy: `all_retained`
- Cases: 5
- Candidate space size: 30
- Exact top-1: 1/5
- Gold in retained candidates: 5/5
- All candidates retained: 5/5
- Mean observed gold rank: 9.4
- Mean observed top-gold margin: 0.115211
- Action field top-1: 2/5
- Evidence-status field top-1: 1/5
- Field-rank patterns: `{"action_field_failure": 2, "both_field_failure": 1, "evidence_status_field_failure": 1, "pair_top1": 1}`

## Held-Out Rank Readout

| Case family | Expected | Top candidate | Pair rank | Action rank | Status rank | Margin | Pattern |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| supported_negative_evidence | `ground` / `supported` | `ground` / `supported` | 1 | 1 | 1 | 0.0 | `pair_top1` |
| contradicted_or_mixed_endpoint_claim | `reject` / `contradicted` | `ground` / `supported` | 5 | 3 | 2 | 0.121863 | `action_field_failure` |
| insufficient_evidence | `defer` / `insufficient` | `ground` / `supported` | 13 | 5 | 4 | 0.171361 | `both_field_failure` |
| related_evidence_requires_verification | `verify` / `insufficient` | `verify` / `invalid_value` | 4 | 1 | 4 | 0.052832 | `evidence_status_field_failure` |
| invalid_value_attribution_failure | `flag` / `invalid_value` | `ground` / `supported` | 24 | 6 | 2 | 0.229999 | `action_field_failure` |

## Interpretation

If the gold pair ranks low and the target action/status fields also rank low, the enum_action target needs field-level supervision. If fields rank near the top but the gold pair remains low, prefer joint-pair contrastive data before tool_query, DPO, or RLVR.

Reports from older runs may contain only retained top-k candidates; rerun the component after the full-score patch for exact ranks. Counterfactual candidate policies are exact only when the raw prediction row retained every candidate in that policy.

## Trace

- Input predictions SHA-256: `bacc78cb74c91b01f3a70e3b7d02f85ef5d6c719949bc735e2853ed189eae28f`

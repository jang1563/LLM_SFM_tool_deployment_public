# Stage A Enum-Action Observed-Pair Counterfactual

Date: 2026-07-05

Purpose: use the existing Cayuga full 30-candidate score table to test
whether restricting `enum_action` decoding to train-observed valid target
pairs fixes the remaining enum-pair failure. This is a no-model-load
counterfactual over saved Cayuga scores, not a new training run.

## Candidate Policy

`train_observed_pairs` uses only pairs observed in the enum-action train
targets:

- `ground` / `supported`
- `reject` / `contradicted`
- `defer` / `insufficient`
- `verify` / `insufficient`
- `flag` / `invalid_value`

## Result

| Metric | Value |
| --- | ---: |
| Exact top-1 | 1/5 |
| Candidate space size | 5 |
| Mean observed gold rank | 2.8 |
| Mean top-gold margin | 0.115158 |
| Top action `ground` count | 5/5 |

## Held-Out Counterfactual Readout

| Case family | Expected | Counterfactual top | Gold rank | Margin | Exact top-1 |
| --- | --- | --- | ---: | ---: | ---: |
| supported_negative_evidence | `ground` / `supported` | `ground` / `supported` | 1 | 0.0 | 1 |
| contradicted_or_mixed_endpoint_claim | `reject` / `contradicted` | `ground` / `supported` | 2 | 0.121863 | 0 |
| insufficient_evidence | `defer` / `insufficient` | `ground` / `supported` | 4 | 0.171361 | 0 |
| related_evidence_requires_verification | `verify` / `insufficient` | `ground` / `supported` | 2 | 0.052566 | 0 |
| invalid_value_attribution_failure | `flag` / `invalid_value` | `ground` / `supported` | 5 | 0.229999 | 0 |

## Interpretation

Restricting to train-observed valid pairs does not fix the enum-action
failure. It removes nonsensical Cartesian pairs, but the model still picks
`ground` / `supported` for every held-out case. That means the next useful
change is not `tool_query`, DPO, or RLVR. It is enum-specific corrective
supervision or contrastive pressure against `ground` / `supported` collapse.

## Trace

- Compact JSON summary: `post_training/stage_a_component_enum_action_observed_pair_counterfactual_2026-07-05.json`
- Source predictions SHA-256: `bacc78cb74c91b01f3a70e3b7d02f85ef5d6c719949bc735e2853ed189eae28f`
- Raw predictions and full candidate-score tables remain untracked under
  ignored `post_training/runs/` in the cluster working copy.

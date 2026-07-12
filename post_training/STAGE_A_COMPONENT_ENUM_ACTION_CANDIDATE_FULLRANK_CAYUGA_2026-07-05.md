# Stage A Enum-Action Candidate Full-Rank Diagnostic

Date: 2026-07-05

Purpose: repeat the repaired `enum_action` component smoke after retaining
all 30 finite `(action, evidence_status)` candidate scores in the ignored
raw prediction artifact. The committed result is only a compact rank/margin
summary.

## Result

| Metric | Value |
| --- | ---: |
| Held-out pass rate | 1/5 |
| Mean score | 0.800 |
| `target_keys` accuracy | 1.0 |
| `enum_validity` accuracy | 1.0 |
| `exact_match` accuracy | 0.2 |
| All candidates retained | 5/5 |
| Mean observed gold rank | 9.4 |
| Mean top-gold margin | 0.115211 |

## Held-Out Rank Readout

| Case family | Expected | Top candidate | Gold rank | Margin | Exact top-1 |
| --- | --- | --- | ---: | ---: | ---: |
| supported_negative_evidence | `ground` / `supported` | `ground` / `supported` | 1 | 0.0 | 1 |
| contradicted_or_mixed_endpoint_claim | `reject` / `contradicted` | `ground` / `supported` | 5 | 0.121863 | 0 |
| insufficient_evidence | `defer` / `insufficient` | `ground` / `supported` | 13 | 0.171361 | 0 |
| related_evidence_requires_verification | `verify` / `insufficient` | `verify` / `invalid_value` | 4 | 0.052832 | 0 |
| invalid_value_attribution_failure | `flag` / `invalid_value` | `ground` / `supported` | 24 | 0.229999 | 0 |

## Interpretation

Full candidate retention shows this is not just a JSON validity problem.
The model still passes only the supported/ground case, and the top candidate
is `ground` / `supported` in four of five held-out cases. The contradicted
and verification-needed gold pairs are rank 5 and rank 4, but insufficient
and invalid-value cases fall to rank 13 and rank 24. That pattern points to
enum-specific corrective supervision or a narrower valid-pair target before
moving to `tool_query`, DPO, or RLVR.

## Trace

- Compact JSON summary: `post_training/stage_a_component_enum_action_candidate_fullrank_cayuga_summary_2026-07-05.json`
- Raw predictions, full candidate scores, model state, and Slurm logs remain
  untracked under ignored `post_training/runs/` in the cluster working copy.
- Predictions SHA-256: `bacc78cb74c91b01f3a70e3b7d02f85ef5d6c719949bc735e2853ed189eae28f`
- Eval report SHA-256:
  `021e969dd8991f6578a2ae411f1a52c713a14bc45fefe1f627f4a3368c15652d`
- Run report SHA-256:
  `a4a97e7b41773707ce4e6fdb7de9d1a2a07abb90bbdd19d21f25d2e237218792`

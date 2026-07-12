# Stage A Enum Candidate-CE Gate Diagnostic

Date: 2026-07-05

Purpose: test whether the 5-way candidate-CE scorer has enough score-gap
calibration to trust top-1 enum decisions. This is a post-hoc diagnostic over
saved candidate scores, not new training, DPO, RLVR, `tool_query`, or a full
trajectory result.

## Setup

| Setting | Value |
| --- | --- |
| Source run | `stage_a_enum_candidate_ce_pair_observed_cayuga_2026_07_05` |
| Candidate policy | `pair_observed_outputs` |
| Candidate space | 5 |
| Held-out pairs | 4 |
| Gate signal | top candidate score minus second candidate score |

## Result

| Threshold | Trusted | Correct trusted | Incorrect trusted | Precision | Fail-closed cases |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.000 | 4 | 1 | 3 | 0.250 | 0 |
| 0.025 | 3 | 1 | 2 | 0.333 | 1 |
| 0.050 | 1 | 0 | 1 | 0.000 | 3 |
| 0.075 | 0 | 0 | 0 | NA | 4 |

The adaptive zero-false-trust threshold is 0.050027. At that threshold the gate
trusts 0/4 rows, so the useful zero-false-trust coverage is 0/4.

## Held-Out Readout

| Case family | Target | Top candidate | Top-1 | Gold rank | Gap | Pattern |
| --- | --- | --- | ---: | ---: | ---: | --- |
| contradicted_or_mixed_endpoint_claim | `reject` / `contradicted` | `reject` / `contradicted` | 1 | 1 | 0.028695 | `pair_top1` |
| insufficient_evidence | `defer` / `insufficient` | `reject` / `contradicted` | 0 | 3 | 0.024171 | `both_field_failure` |
| related_evidence_requires_verification | `verify` / `insufficient` | `reject` / `contradicted` | 0 | 2 | 0.050026 | `both_field_failure` |
| invalid_value_attribution_failure | `flag` / `invalid_value` | `reject` / `contradicted` | 0 | 4 | 0.025494 | `both_field_failure` |

## Interpretation

The score-gap signal is not calibrated enough to trust enum top-1 decisions.
The only correct top-1 row has a smaller gap than one incorrect row, so any
zero-false-trust gate must fail closed on all four held-out cases. This supports
the current research boundary: keep `tool_query`, DPO, and RLVR gated, and
repair `enum_action` with candidate calibration, action/status factorization, or
a constrained candidate router.

## Trace

- Compact JSON summary:
  `post_training/stage_a_enum_candidate_ce_gate_cayuga_summary_2026-07-05.json`
- Analyzer: `post_training/analyze_stage_a_enum_candidate_gate.py`
- Input candidate JSONL SHA-256:
  `dbe528f66a2ec0bbd420c63847ffa43c10258b596f0a2ccb249335e29cf90517`
- Raw reports and candidate JSONL remain untracked under ignored
  `post_training/runs/` in the cluster working copy.

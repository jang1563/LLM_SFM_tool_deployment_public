# Stage A Routing Candidate Gate Diagnostic

Purpose: test whether routing candidate score gaps can support a
fail-closed boundary gate without publishing prompts, raw score JSONL,
model state, or scheduler logs.

## Summary

- Run ID: `stage_a_routing_defer_verify_gate_trained_2026_07_08`
- Candidate policy: `insufficient_defer_vs_verify_boundary`
- Score label: `trained_candidate_heldout`
- Cases: 2
- Exact top-1: 1/2
- Mean gold rank: 1.5
- Mean top-second gap: 0.030021
- Fail-closed output: `defer` / `insufficient`
- Top pair counts: `{"verify/insufficient": 2}`
- Target pair counts: `{"defer/insufficient": 1, "verify/insufficient": 1}`

## Gate Thresholds

| Threshold | Trusted | Correct trusted | Unsafe trusted | Fail closed | Strict final correct | Strict accuracy |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.0 | 2 | 1 | 1 | 0 | 1 | 0.5 |
| 0.01 | 2 | 1 | 1 | 0 | 1 | 0.5 |
| 0.025 | 1 | 1 | 0 | 1 | 2 | 1.0 |
| 0.05 | 0 | 0 | 0 | 2 | 1 | 0.5 |
| 0.075 | 0 | 0 | 0 | 2 | 1 | 0.5 |
| 0.1 | 0 | 0 | 0 | 2 | 1 | 0.5 |
| 0.15 | 0 | 0 | 0 | 2 | 1 | 0.5 |
| 0.2 | 0 | 0 | 0 | 2 | 1 | 0.5 |

## Fail-Closed Readout

- Best default zero-unsafe threshold: `0.025` with 2 strict final correct rows and 0 unsafe trusted rows.
- Adaptive zero-unsafe threshold: `0.023558` with 2 strict final correct rows and 0 unsafe trusted rows.

## Held-Out Rows

| Case family | Target | Top candidate | Top-1 | Gold rank | Gap | Fail-closed exact |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| insufficient_evidence | `defer` / `insufficient` | `verify` / `insufficient` | 0 | 2 | 0.023557 | 1 |
| related_evidence_requires_verification | `verify` / `insufficient` | `verify` / `insufficient` | 1 | 1 | 0.036484 | 0 |

## Interpretation

A useful boundary gate should avoid unsafe trust and improve or preserve strict final correctness after fail-closed routing. On tiny held-out slices this is only a diagnostic, not deployment calibration or a reason to start DPO/RLVR.

This post-hoc diagnostic uses saved candidate scores only. It is not new training, not explanation-quality scoring, and not a full trajectory result.

## Trace

- Input candidate JSONL SHA-256: `6e4e7a09b7c537ba492ce5753ad3ad99578b8e4403fc2d2bd47cf521ea44beb0`

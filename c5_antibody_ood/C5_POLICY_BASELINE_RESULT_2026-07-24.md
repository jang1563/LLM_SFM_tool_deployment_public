# Stage B C5 No-API Policy Prototype

This synthetic public fixture tests whether antibody-antigen specialist
trust is blocked unless metric scope, regime-matched calibration, an
RCPS threshold, and baseline arbitration are complete.

## Results

| Policy | Exact pass | Mean score | Unsafe trust |
| --- | ---: | ---: | ---: |
| `trust_all` | 3/12 | 0.667 | 9 |
| `general_gate` | 3/12 | 0.681 | 8 |
| `regime_specific_gate` | 6/12 | 0.896 | 0 |
| `fail_closed` | 12/12 | 1.000 | 0 |

## Decision

- Oracle trajectories: `12/12`.
- No-API manifest and fail-closed prototype: `pass`.
- This is a synthetic policy-test result, not calibration evidence.
- A source-backed C5 pilot with frozen interface labels is required next.
- Model training, DPO/RLVR, and independent-transfer claims remain closed.

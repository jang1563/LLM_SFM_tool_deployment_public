# Stage A Enum Full Pairwise-Margin Cayuga Diagnostic

Date: 2026-07-05

Purpose: test whether the supervised pairwise-margin repair survives full
`action` plus `evidence_status` JSON scoring. This follows the action-only
pairwise-margin result and keeps DPO/RLVR gated.

## Setup

| Setting | Value |
| --- | --- |
| Target format | `full` |
| Pairwise margin weight | 1.0 |
| Required pairwise margin | 0.05 |
| Log-probability mode | mean |
| Focus pair | `flag` / `invalid_value` |
| Focus repeat | 4 |
| Unique train pairs | 16 per run |
| Training rows after sampling | 28 per run |
| Held-out pairs | 4 per run |

Two Cayuga smoke runs were used:

1. Same-status action contrasts: rejected targets preserve
   `evidence_status` and change the action to `ground`.
2. Ground/supported corrective contrasts: rejected targets are the observed
   `ground` / `supported` collapse.

## Result

| Slice | Base held-out wins | Trained held-out wins | Base mean margin | Trained mean margin | Violations after training |
| --- | ---: | ---: | ---: | ---: | ---: |
| Same-status action contrast | 1/4 | 4/4 | -0.079422 | 0.144671 | 0 |
| Ground/supported corrective | 0/4 | 4/4 | -0.116510 | 0.177690 | 0 |

This is the first positive full-target enum margin result. It is not yet a
free-generation or full-trajectory result.

## Same-Status Action Contrast

| Case family | Chosen target | Rejected target | Base margin | Trained margin | Delta | Outcome |
| --- | --- | --- | ---: | ---: | ---: | --- |
| contradicted_or_mixed_endpoint_claim | `reject` / `contradicted` | `ground` / `contradicted` | -0.059933 | 0.175385 | 0.235318 | newly_won |
| insufficient_evidence | `defer` / `insufficient` | `ground` / `insufficient` | -0.150195 | 0.109272 | 0.259467 | newly_won |
| related_evidence_requires_verification | `verify` / `insufficient` | `ground` / `insufficient` | 0.044673 | 0.195294 | 0.150621 | remained_won |
| invalid_value_attribution_failure | `flag` / `invalid_value` | `ground` / `invalid_value` | -0.152232 | 0.098731 | 0.250963 | newly_won |

Train-pair check: 16/16 wins, mean train margin 0.139681, minimum train margin
0.049438.

## Ground/Supported Corrective Contrast

| Case family | Chosen target | Rejected target | Base margin | Trained margin | Delta | Outcome |
| --- | --- | --- | ---: | ---: | ---: | --- |
| contradicted_or_mixed_endpoint_claim | `reject` / `contradicted` | `ground` / `supported` | -0.034748 | 0.196507 | 0.231255 | newly_won |
| insufficient_evidence | `defer` / `insufficient` | `ground` / `supported` | -0.257193 | 0.133284 | 0.390477 | newly_won |
| related_evidence_requires_verification | `verify` / `insufficient` | `ground` / `supported` | -0.020147 | 0.260979 | 0.281126 | newly_won |
| invalid_value_attribution_failure | `flag` / `invalid_value` | `ground` / `supported` | -0.153953 | 0.119988 | 0.273941 | newly_won |

Train-pair check: 16/16 wins, mean train margin 0.189835, minimum train margin
0.091670.

## Interpretation

The deterministic supervised pairwise-margin objective now repairs the
teacher-forced full JSON enum margin slice in two related settings. The result
supports the current thesis: the failing `enum_action` behavior is trainable as
a component-level routing objective, but it should still be verified through
runtime-constrained selection and later full trajectory evaluation.

This does not justify DPO or RLVR yet. The next clean gate is finite-candidate
enum selection or the next component slice, with raw run outputs kept out of the
public repo.

## Trace

- Compact JSON summary:
  `post_training/stage_a_enum_full_pairwise_margin_cayuga_summary_2026-07-05.json`
- Runner: `post_training/run_stage_a_enum_corrective_sft_smoke.py`
- Raw reports, margin JSONL files, model state, and Slurm logs remain untracked
  under ignored `post_training/runs/` in the cluster working copy.
- Same-status action-contrast report SHA-256:
  `f347516a5ae519428d5ac4cd9c99000d3e3fe5213a7c9faf9faf6328bed12a3d`
- Same-status action-contrast margin delta report SHA-256:
  `61524bd3ec1f6ee9f4dc492bf95642b614b5d0149d1f35f34fc1241433ed17ca`
- Ground/supported corrective report SHA-256:
  `84b89c31c9edf122a872d884dedf6b664359e2250d6a0aa26fa84ddfb1a094b2`
- Ground/supported corrective margin delta report SHA-256:
  `ba2d0c2970470c247bf24649887172b84a2008a8dbf46743b0230c1c762d1ccf`

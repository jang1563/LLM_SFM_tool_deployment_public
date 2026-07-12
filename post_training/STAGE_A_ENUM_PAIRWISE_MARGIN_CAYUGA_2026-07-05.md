# Stage A Enum Pairwise-Margin Cayuga Diagnostic

Date: 2026-07-05

Purpose: test whether a deterministic supervised pairwise-margin objective can
make weak `enum_action` action-only targets beat rejected `ground` targets. This
is not DPO/RLVR; it is a chosen-over-rejected hinge objective over the existing
action-contrast pairs.

## Setup

| Setting | Value |
| --- | --- |
| Target format | `action_only` |
| Pairwise margin weight | 1.0 |
| Required pairwise margin | 0.05 |
| Log-probability mode | mean |
| Focus pair | `flag` / `invalid_value` |
| Focus repeat | 4 |
| Unique train pairs | 16 |
| Training rows after sampling | 28 |
| Held-out pairs | 4 |

## Result

| Metric | Base model | Pairwise-margin SFT |
| --- | ---: | ---: |
| Held-out wins | 1/4 | 4/4 |
| Mean held-out margin | -0.211307 | 0.261516 |
| Minimum held-out margin | -0.427937 | 0.189837 |
| Chosen-not-above-rejected violations | 3 | 0 |

Mean held-out margin delta is 0.472823. This repairs the action-only
chosen-over-`ground` margin slice, but it does not yet prove full JSON
`enum_action` or trajectory-level repair.

## Held-Out Readout

| Case family | Chosen action | Rejected action | Base margin | Trained margin | Delta | Outcome |
| --- | --- | --- | ---: | ---: | ---: | --- |
| contradicted_or_mixed_endpoint_claim | `reject` | `ground` | -0.358202 | 0.189837 | 0.548039 | newly_won |
| insufficient_evidence | `defer` | `ground` | -0.427937 | 0.216957 | 0.644894 | newly_won |
| related_evidence_requires_verification | `verify` | `ground` | 0.210202 | 0.447062 | 0.236860 | remained_won |
| invalid_value_attribution_failure | `flag` | `ground` | -0.269291 | 0.192207 | 0.461498 | newly_won |

## Train-Pair Check

| Chosen pair | Train wins | Mean train margin |
| --- | ---: | ---: |
| `reject` / `contradicted` | 4/4 | 0.297119 |
| `verify` / `insufficient` | 4/4 | 0.369597 |
| `defer` / `insufficient` | 4/4 | 0.197116 |
| `flag` / `invalid_value` | 4/4 | 0.160434 |

## Interpretation

The supervised pairwise-margin objective fixes the action-only margin slice
that plain action-only SFT did not fix. The weak `flag`, `defer`, and `reject`
families now beat `ground` on both held-out and train-pair margin checks.

This is still a component diagnostic, not a full benchmark result. The next gate
is to test whether the same margin-trained model improves full
action-plus-status enum scoring or finite-candidate component selection without
collapsing evidence-status fields.

## Trace

- Compact JSON summary:
  `post_training/stage_a_enum_pairwise_margin_cayuga_summary_2026-07-05.json`
- Runner: `post_training/run_stage_a_enum_corrective_sft_smoke.py`
- Raw reports, margin JSONL files, model state, and Slurm logs remain untracked
  under ignored `post_training/runs/` in the cluster working copy.
- Run report SHA-256:
  `ca3d07cd6e69be25f465d41fae7142a1fc4359ee8f9d711066b03932e06f727b`
- Margin delta report SHA-256:
  `59e0ae87398db4ed2863457fc510f7da8200669e27168cd27ca4c4ad30c1d647`
- Held-out margin report SHA-256:
  `38eae50c0e55e9f0ab6945c3f87f8867c217dea6dd79758a09a436fcc2d586c6`

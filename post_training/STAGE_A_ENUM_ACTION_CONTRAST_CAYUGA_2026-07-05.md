# Stage A Enum Action-Contrast Cayuga Diagnostic

Date: 2026-07-05

Purpose: test whether preserving `evidence_status` while contrasting only the
terminal `action` field repairs the remaining Stage A `enum_action` bottleneck,
especially `flag` / `invalid_value`. This is an SFT/margin diagnostic, not
DPO/RLVR.

## Setup

| Setting | Value |
| --- | --- |
| Contrast | same-status wrong-action contrast |
| Focus pair | `flag` / `invalid_value` |
| Focus repeat | 4 |
| Unique train pairs | 16 |
| Training rows after sampling | 28 |
| Held-out pairs | 4 |

## Result

| Metric | Base model | Action-contrast SFT |
| --- | ---: | ---: |
| Held-out wins | 1/4 | 2/4 |
| Mean held-out margin | -0.079422 | -0.000604 |
| Minimum held-out margin | -0.152232 | -0.036514 |
| Chosen-not-above-rejected violations | 3 | 2 |

Mean held-out margin delta is 0.078818. This is useful movement, but not a
stable repair.

## Held-Out Readout

| Case family | Chosen pair | Rejected pair | Base margin | Trained margin | Delta | Outcome |
| --- | --- | --- | ---: | ---: | ---: | --- |
| contradicted_or_mixed_endpoint_claim | `reject` / `contradicted` | `ground` / `contradicted` | -0.059933 | 0.015332 | 0.075265 | newly_won |
| insufficient_evidence | `defer` / `insufficient` | `ground` / `insufficient` | -0.150195 | -0.024022 | 0.126173 | remained_lost |
| related_evidence_requires_verification | `verify` / `insufficient` | `ground` / `insufficient` | 0.044673 | 0.042788 | -0.001885 | remained_won |
| invalid_value_attribution_failure | `flag` / `invalid_value` | `ground` / `invalid_value` | -0.152232 | -0.036514 | 0.115718 | remained_lost |

## Train-Pair Check

| Chosen pair | Train wins | Mean train margin |
| --- | ---: | ---: |
| `reject` / `contradicted` | 3/4 | 0.034673 |
| `verify` / `insufficient` | 4/4 | 0.074206 |
| `defer` / `insufficient` | 0/4 | -0.029630 |
| `flag` / `invalid_value` | 0/4 | -0.062056 |

## Interpretation

Action-contrast supervision moves the model in the right direction, especially
for the invalid-value held-out case, but it does not repair the weak action
families. The trained model still prefers `ground` over `flag` for the held-out
invalid-value case and `ground` over `defer` for the held-out insufficient case.

The next useful step is not DPO/RLVR or moving to `tool_query`. The
`enum_action` slice still needs a target-format or data repair for `flag` and
`defer` action selection.

## Trace

- Compact JSON summary:
  `post_training/stage_a_enum_action_contrast_cayuga_summary_2026-07-05.json`
- Runner: `post_training/run_stage_a_enum_corrective_sft_smoke.py`
- Raw reports, margin JSONL files, model state, and Slurm logs remain untracked
  under ignored `post_training/runs/` in the cluster working copy.
- Run report SHA-256:
  `b008e9e87e5f93f7ed648ed7ae41c405c8ff26a9517450840f2f7911463c1333`
- Margin delta report SHA-256:
  `85b5c4208a355011ccfa264a2db35b1457e27e9397f2628c1e9691dcba73163f`
- Held-out margin report SHA-256:
  `99ecde710dbb658a3157f6ac36ee8cadc371235534ece3d505bd50caf9038624`

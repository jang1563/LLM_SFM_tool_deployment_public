# Stage A Enum Corrective Targeted Sampling Diagnostic

Date: 2026-07-05

Purpose: test whether oversampling the weak enum train pairs fixes the remaining
`flag` / `invalid_value` and `defer` / `insufficient` failures without adding new
data or changing methods. This is SFT sampling over existing train rows, not
DPO/RLVR.

## Setup

| Setting | Value |
| --- | --- |
| Focus pairs | `flag` / `invalid_value`; `defer` / `insufficient` |
| Focus repeat | 4 |
| Unique train pairs | 16 |
| Training rows after sampling | 40 |
| Held-out pairs | 4 |

## Result

| Metric | Non-targeted run | Targeted run |
| --- | ---: | ---: |
| Held-out wins | 2/4 | 1/4 |
| Mean trained held-out margin | -0.020685 | -0.008175 |
| Mean held-out margin delta | 0.095825 | 0.108336 |
| Newly won held-out pairs | 2/4 | 1/4 |

## Held-Out Readout

| Case family | Chosen pair | Trained margin | Delta | Outcome |
| --- | --- | ---: | ---: | --- |
| contradicted_or_mixed_endpoint_claim | `reject` / `contradicted` | -0.005595 | 0.029153 | remained_lost |
| insufficient_evidence | `defer` / `insufficient` | -0.009988 | 0.247205 | remained_lost |
| related_evidence_requires_verification | `verify` / `insufficient` | 0.078515 | 0.098662 | newly_won |
| invalid_value_attribution_failure | `flag` / `invalid_value` | -0.095630 | 0.058323 | remained_lost |

## Train-Pair Check

| Chosen pair | Train wins | Mean train margin |
| --- | ---: | ---: |
| `reject` / `contradicted` | 4/4 | 0.042518 |
| `verify` / `insufficient` | 4/4 | 0.078015 |
| `defer` / `insufficient` | 1/4 | -0.017516 |
| `flag` / `invalid_value` | 0/4 | -0.099119 |

## Interpretation

Targeted oversampling is not a stable repair. It improves the mean held-out
margin and pushes `defer` / `insufficient` close to zero, but held-out wins drop
from 2/4 to 1/4 and `flag` / `invalid_value` still fails even on train-pair
margins.

The next useful step is not running longer, DPO/RLVR, or moving to `tool_query`.
The invalid-value target representation or targeted invalid-value rows need a
separate diagnosis first.

## Trace

- Compact JSON summary: `post_training/stage_a_enum_corrective_targeted_cayuga_summary_2026-07-05.json`
- Runner: `post_training/run_stage_a_enum_corrective_sft_smoke.py`
- Raw reports, margin JSONL files, model state, and Slurm logs remain untracked
  under ignored `post_training/runs/` in the cluster working copy.
- Run report SHA-256:
  `8017db6710fbb39aa0d4813aa47982f77ed8504ca379608f915604d28046232c`
- Margin delta report SHA-256:
  `0b1cc8da75921cfb6c70a4fd6d36ab7c24ef59dd8ff58320bc70447a42d59418`
- Held-out margin report SHA-256:
  `30c2b37a494af6b0c5451b4b056b7ad67c692b97caa7769b946f76b1707ca546`

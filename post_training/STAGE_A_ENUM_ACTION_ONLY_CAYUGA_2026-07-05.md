# Stage A Enum Action-Only Cayuga Diagnostic

Date: 2026-07-05

Purpose: test whether the weak Stage A `enum_action` families pass when the
SFT and teacher-forced margin target is only `{"action": ...}`, without
`evidence_status` in the output JSON. This is a target-format diagnostic, not
DPO/RLVR.

## Setup

| Setting | Value |
| --- | --- |
| Target format | `action_only` |
| Contrast | same-status wrong-action contrast |
| Focus pair | `flag` / `invalid_value` |
| Focus repeat | 4 |
| Unique train pairs | 16 |
| Training rows after sampling | 28 |
| Held-out pairs | 4 |

## Result

| Metric | Base model | Action-only SFT |
| --- | ---: | ---: |
| Held-out wins | 1/4 | 1/4 |
| Mean held-out margin | -0.211307 | -0.032852 |
| Minimum held-out margin | -0.427937 | -0.135613 |
| Chosen-not-above-rejected violations | 3 | 3 |

Mean held-out margin delta is 0.178455, so training moves the margins, but it
does not convert any new held-out pair.

## Held-Out Readout

| Case family | Chosen action | Rejected action | Base margin | Trained margin | Delta | Outcome |
| --- | --- | --- | ---: | ---: | ---: | --- |
| contradicted_or_mixed_endpoint_claim | `reject` | `ground` | -0.358202 | -0.135613 | 0.222589 | remained_lost |
| insufficient_evidence | `defer` | `ground` | -0.427937 | -0.073491 | 0.354446 | remained_lost |
| related_evidence_requires_verification | `verify` | `ground` | 0.210202 | 0.143328 | -0.066874 | remained_won |
| invalid_value_attribution_failure | `flag` | `ground` | -0.269291 | -0.065632 | 0.203659 | remained_lost |

## Train-Pair Check

| Chosen pair | Train wins | Mean train margin |
| --- | ---: | ---: |
| `reject` / `contradicted` | 0/4 | -0.055307 |
| `verify` / `insufficient` | 3/4 | 0.076226 |
| `defer` / `insufficient` | 0/4 | -0.119163 |
| `flag` / `invalid_value` | 0/4 | -0.120798 |

## Interpretation

Removing `evidence_status` from the target does not repair the weak action
families. The margins move in the right direction, but the model still prefers
`ground` over `flag`, `defer`, and `reject` in the action-only readout.

This means the next enum repair should not be just another target-format tweak.
Before moving to `tool_query`, DPO, or RLVR, the benchmark needs targeted action
rows or a constrained action-head objective for `flag`, `defer`, and `reject`.

## Trace

- Compact JSON summary:
  `post_training/stage_a_enum_action_only_cayuga_summary_2026-07-05.json`
- Runner: `post_training/run_stage_a_enum_corrective_sft_smoke.py`
- Raw reports, margin JSONL files, model state, and Slurm logs remain untracked
  under ignored `post_training/runs/` in the cluster working copy.
- Run report SHA-256:
  `5190cf22398fb481a574e0594b64b4b5dbc4aefafa0612e4214eaefd016e56bc`
- Margin delta report SHA-256:
  `5b93ecda8271e440c8ff0eeeb2e0898580d9507fa111d962f14efb58fbb324db`
- Held-out margin report SHA-256:
  `61033fa37e4ae2d4c88b1bcbe929dc722040bebf196ca40e1ffec8cd823c351d`

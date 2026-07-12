# Stage A Implementation Checkpoint

Date: 2026-07-04

Purpose: record the first code-benchmark implementation checkpoint for the
long-term research plan.

## Implemented

- Added `negbiodb_ct.stage_a_manifest`:
  - Stage A JSONL manifest loader/writer.
  - Projection from `model_visible_task` + `hidden_eval_metadata` into
    `TaskSpec`.
  - Deterministic oracle trajectory generation.
  - Manifest-level score wrapper for query/filter completeness.
  - Failure-mode trajectory matrix for DPO/process-supervision candidates.
  - Manifest validator for required fields, class/status balance, hidden-label
    leakage, duplicate case IDs, and split-group overlap.
- Added `negbiodb_ct/stage_a_mini_manifest.jsonl`:
  - 25 no-API Stage A cases.
  - 5 rows each for `ground`, `reject`, `defer`, `verify`, and `flag` source
    action classes.
  - Model-visible prompts use neutral `stage_a::000000`-style IDs; hidden
    source task IDs, evidence labels, source IDs, and expected terminal actions
    live only under `hidden_eval_metadata`.
- Added `examples/run_stage_a_manifest_eval.py`:
  - No-API baseline runner over the Stage A manifest.
  - Evaluates `oracle`, `self_answer`, `wrong_tool`, and `partial_query`.
- Added `post_training/export_stage_a_data.py`:
  - Exports oracle trajectories as SFT rows.
  - Exports chosen/rejected trajectory pairs for preference/process supervision.
  - Writes a compact manifest with class/status counts, failure-mode counts,
    pass/fail direction, and split-group overlap status.
- Extended `post_training/validate_post_training_data.py`:
  - Validates Stage A SFT, preference, process, and export manifest artifacts.
  - Checks model-visible prompt boundaries, chosen-pass/rejected-fail direction,
    split-group uniqueness, source-case consistency, and process-target shape.
- Added `post_training/split_stage_a_data.py`:
  - Splits Stage A SFT, preference, and process artifacts by
    `source_manifest_case_id`.
  - Holds out one case per case family.
  - Writes train/held-out artifacts plus a manifest with case IDs, split groups,
    source task IDs, and overlap checks.

## No-API Baseline Result

Command:

```bash
PYTHONPATH=. python examples/run_stage_a_manifest_eval.py --json
```

Result:

| baseline | cases | passed | mean score |
| --- | ---: | ---: | ---: |
| `oracle` | 25 | 25 | 1.000 |
| `self_answer` | 25 | 0 | 0.229 |
| `wrong_tool` | 25 | 0 | 0.714 |
| `partial_query` | 25 | 0 | 0.857 |

Interpretation:

- The manifest/evaluator can distinguish a valid evidence trajectory from
  common shortcut failures without API, DB, or HPC spend.
- `partial_query` gets a high partial score but fails because the manifest-level
  required-query-field check catches missing `drug_id` / `condition_id`.
- This is the intended Stage A behavior: correctness-first scoring before any
  cost-aware reward or live model loop.

## Validator Status

Focused tests passed:

```bash
PYTHONPATH=. python -m pytest -q \
  tests/test_stage_a_manifest.py \
  tests/test_stage_a_manifest_eval_script.py \
  tests/test_negbiodb_ct_adapter.py \
  tests/test_trajectory_evaluator.py
```

Result: `23 passed`.

Stage A export tests passed:

```bash
PYTHONPATH=. python -m pytest -q \
  tests/test_stage_a_export.py \
  tests/test_post_training_data_validator.py
```

Result: `15 passed`.

Post-training artifact validation passed:

```bash
PYTHONPATH=. python post_training/validate_post_training_data.py
```

Stage A export summary:

| artifact | rows |
| --- | ---: |
| `post_training/stage_a_sft_v1.jsonl` | 25 |
| `post_training/stage_a_preferences_v1.jsonl` | 150 |
| `post_training/stage_a_process_supervision_v1.jsonl` | 25 |
| `post_training/stage_a_export_manifest.json` | 1 |

Preference failure-mode counts:

| failure mode | pairs |
| --- | ---: |
| `self_answering_without_tools` | 25 |
| `wrong_tool` | 25 |
| `missing_tool` | 25 |
| `partial_query` | 25 |
| `unsupported_trust` | 25 |
| `missing_attribution` | 10 |
| `insufficient_as_negative` | 10 |
| `invalid_value_missed` | 5 |

All 150 chosen trajectories pass under `score_stage_a_trajectory`; all 150
rejected variants fail. Split-group overlap is empty.

## Deterministic Split Result

Command:

```bash
PYTHONPATH=. python post_training/split_stage_a_data.py
```

Result:

| split | cases | SFT rows | preference pairs | process rows |
| --- | ---: | ---: | ---: | ---: |
| train | 20 | 20 | 120 | 20 |
| held-out | 5 | 5 | 30 | 5 |

Held-out balance:

| case family | held-out cases |
| --- | ---: |
| `supported_negative_evidence` | 1 |
| `contradicted_or_mixed_endpoint_claim` | 1 |
| `invalid_value_attribution_failure` | 1 |
| `related_evidence_requires_verification` | 1 |
| `insufficient_evidence` | 1 |

Overlap checks:

| overlap key | result |
| --- | --- |
| `source_manifest_case_id` | none |
| `split_group` | none |
| `source_task_id` | none |

The split is intentionally case-level. Preference rows are not independently
shuffled, so all chosen/rejected variants for a case stay on the same side of
the split.

## Next Research Step

Milestone 1 is code-backed, and Milestone 2 now has first-pass no-API trajectory
artifacts plus a deterministic train/held-out split. The next implementation
target is the training/evaluation gate:

1. Wire the existing local SFT smoke/eval harness to the Stage A SFT train and
   held-out artifacts.
2. Keep no train/eval split-group overlap as a hard preflight check.
3. Run prompt-only/live-model baselines only after the no-API validation gate
   stays green.

# Benchmark Card: LLM-SFM Tool Deployment Stage A

## Summary

Stage A is a small, public-safe benchmark substrate for evaluating biology
tool-use agents on trajectories rather than prose. A passing trajectory must use
the required tools, assemble an evidence packet, choose the expected terminal
action, cite required source IDs when attribution is needed, and obey
fail-closed policy gates.

The benchmark is intentionally narrow. It tests whether an agent can route a
biomedical negative-evidence claim through a verifiable tool loop without
hallucinating certainty.

## Core Claim

Biology agents need trainable trajectories plus runtime enforcement. The
benchmark does not claim that RLVR alone solves scientific reasoning, that a
language model discovered hidden evidence, or that this package is clinical
decision support.

## Task Shape

Each manifest row has two layers:

| Layer | Visible to model | Purpose |
|---|---:|---|
| `model_visible_task` | Yes | Natural-language claim, `input_id`, allowed tools |
| `hidden_eval_metadata` | No | expected evidence status, terminal action, required tools, source IDs, split group |

The loader projects hidden metadata into the shared `TaskSpec` schema. The
model-visible prompt never receives hidden labels such as source IDs, evidence
status, expected action, or split groups.

## Action Space

The evaluator scores terminal actions from the shared trajectory schema:

- `ground_with_attribution`
- `reject_or_flag_unsupported_claim`
- `verify_with_assay_or_database`
- `defer_or_request_more_evidence`
- `trust_specialist_output`
- `use_cheap_baseline`
- `call_specialist_tool`
- `answer_self`

For Stage A, `answer_self` is expected to fail because external tool use is
required. Trusting specialist output is also fail-closed unless calibration and
regime-match metadata are present.

## Evidence States

Stage A balances cases across:

- `supported`
- `contradicted`
- `invalid_value`
- `insufficient`

The public mini-manifest contains 25 cases. It is a benchmark seed, not a broad
clinical benchmark.

## Evaluation Gates

A trajectory must satisfy all relevant gates:

| Gate | What is checked |
|---|---|
| Tool sequence | Required tools appear in order |
| Query completeness | Required query fields are present |
| Evidence status | Predicted status matches hidden label |
| Attribution | Required source IDs are cited |
| Terminal action | Final action matches expected action |
| Policy compliance | Self-answering, unsupported trust, and unsafe specialist trust fail |

The evaluator returns a score, a reward breakdown, and explicit violation codes.
Passing means there are no violations and the earned score equals the possible
score.

## Public Artifacts

| Artifact | Count | Use |
|---|---:|---|
| `negbiodb_ct/stage_a_mini_manifest.jsonl` | 25 cases | Public-safe benchmark manifest |
| `post_training/stage_a_sft_v1.jsonl` | 25 rows | Oracle SFT trajectory targets |
| `post_training/stage_a_preferences_v1.jsonl` | 150 pairs | Chosen/rejected trajectory pairs |
| `post_training/stage_a_process_supervision_v1.jsonl` | 25 rows | Process-field targets |
| `post_training/stage_a_*_train_v1.jsonl` | 20 cases | Train split artifacts |
| `post_training/stage_a_*_heldout_v1.jsonl` | 5 cases | Held-out split artifacts |
| `post_training/stage_a_strict_contract_sft_v1.jsonl` | 25 rows | Compact JSON targets matching the strict saved-prediction contract |
| `post_training/stage_a_strict_contract_preferences_v1.jsonl` | 50 pairs | Observed-collapse chosen/rejected compact JSON pairs |
| `post_training/stage_a_strict_contract_process_v1.jsonl` | 25 rows | Strict-contract process targets |
| `post_training/run_stage_a_strict_contract_sft_smoke.py` | 1 script | Cluster-oriented strict-contract SFT smoke runner with public-safe dry-run |
| `post_training/stage_a_sft_smoke_eval_summary_2026-07-04.json` | 1 report | No-API split-aware SFT smoke/eval baseline |
| `post_training/generate_stage_a_predictions.py` | 1 script | Artifact-first producer for saved prediction JSONL |
| `post_training/evaluate_stage_a_predictions.py` | 1 script | Offline scorer for saved API, local-SFT, or prompt-only prediction JSONL |
| `post_training/stage_a_prospective_real_query_tool_query_v1.jsonl` | 25 rows | Case-specific typed query targets for public development |
| `post_training/stage_a_prospective_real_query_routing_perturbations_v1.jsonl` | 180 rows | Synthetic runtime routing perturbations |
| `negbiodb_ct/tool_query_runtime.py` | 1 module | Fail-closed compiler for the fixed Stage A tool/query contract |
| `post_training/stage_a_tool_query_runtime_compiler_result_2026-07-23.json` | 1 report | Clean and malformed-input compiler evaluation |

Checksums and record counts are registered in
`release/public_release_manifest.json`.

## Failure Modes

The preference artifact includes rejected trajectories for:

- `self_answering_without_tools`
- `wrong_tool`
- `missing_tool`
- `partial_query`
- `missing_attribution`
- `invalid_value_missed`
- `unsupported_trust`
- `insufficient_as_negative`

These are process failures, not just wrong final answers.

## Baselines

The public Stage A smoke result is:

| Baseline | Expected result |
|---|---:|
| Oracle trajectory | Passes 25/25 |
| Self-answer shortcut | Fails 25/25 |
| Wrong-tool shortcut | Fails 25/25 |
| Partial-query shortcut | Fails 25/25 |

This verifies that the evaluator catches shortcut trajectories before live API,
HPC, or model-training spend.

## Model Diagnostics

The first source-separated model result is deliberately negative:

| Diagnostic | Model result | Reference |
|---|---:|---:|
| Tool-query placeholder schema | 0/5 | 5/5 required |
| Exposed-development candidate routing | 1/5 | runtime gate 5/5 |
| One-time sealed candidate routing | 5/25 | static prior 5/25; runtime oracle 25/25 |
| Prospective frozen routing | 35/180 | best static pair 80/180; deterministic gate 180/180 |
| Prospective runtime hybrid | 115/180 | zero unsafe grounding; zero decisive coverage |
| Real-query base / frozen placeholder SFT | 0/25 / 0/25 | strict case-specific tool calls |
| Explicit-contract base | 0/25 | target keys 25/25; strict call shape 0/25 |
| Runtime tool-query compiler | 25/25 clean | 150/150 malformed inputs rejected |

The sealed policy selects `verify/insufficient` on all 25 cases. It has zero
incorrect `ground/supported` predictions but does not distinguish evidence
families. This supports runtime arbitration and does not justify DPO/RLVR.

The prospective tool-query rows use actual model-visible identifiers, but the
current fixed-order copy operation is deterministic. It is therefore enforced
by runtime code rather than presented as a model reasoning win. Prospective
routing uses synthetic tool-result perturbations and the deterministic gate
defines their expected policy, so its 180/180 score is a positive control, not
an external generalization estimate.

## Reproducibility

Use the public-safe validation path:

```bash
pip install -r requirements-public.txt
python scripts/check_public_release.py
python scripts/check_public_git_history.py
python post_training/validate_post_training_data.py
python examples/run_public_demo.py
python post_training/run_stage_a_sft_smoke_eval.py --json
python post_training/generate_stage_a_predictions.py \
  --mode self_answer \
  --sft post_training/stage_a_sft_heldout_v1.jsonl \
  --out /tmp/stage_a_self_answer_predictions.jsonl \
  --run-id self_answer_saved_prediction_smoke
python post_training/evaluate_stage_a_predictions.py \
  --predictions /tmp/stage_a_self_answer_predictions.jsonl \
  --expected-sft post_training/stage_a_sft_heldout_v1.jsonl \
  --run-id self_answer_saved_prediction_smoke \
  --json
python post_training/evaluate_stage_a_predictions.py \
  --predictions post_training/stage_a_sft_heldout_v1.jsonl \
  --expected-sft post_training/stage_a_sft_heldout_v1.jsonl \
  --run-id heldout_oracle_adapter_smoke \
  --json
python post_training/evaluate_stage_a_tool_query_runtime_compiler.py \
  --out-json /tmp/stage_a_tool_query_runtime_compiler_result.json \
  --out-md /tmp/STAGE_A_TOOL_QUERY_RUNTIME_COMPILER_RESULT.md
python -m pytest -q \
  tests/test_trajectory_evaluator.py \
  tests/test_public_demo.py \
  tests/test_public_release_checker.py \
  tests/test_stage_a_manifest.py \
  tests/test_stage_a_manifest_eval_script.py \
  tests/test_stage_a_export.py \
  tests/test_stage_a_strict_contract_export.py \
  tests/test_stage_a_strict_contract_sft_smoke.py \
  tests/test_stage_a_split.py \
  tests/test_stage_a_sft_smoke_eval.py \
  tests/test_stage_a_prediction_eval.py \
  tests/test_stage_a_prediction_generator.py \
  tests/test_stage_a_prospective_real_query_slice.py \
  tests/test_stage_a_prospective_runtime_hybrid.py \
  tests/test_stage_a_prospective_tool_query_transfer.py \
  tests/test_stage_a_tool_query_runtime.py \
  tests/test_post_training_data_validator.py
```

For a fuller run, see `REPRODUCIBILITY.md`.

## Limitations

- The public manifest is small and designed for substrate validation.
- Some larger NegBioDB-CT experiments depend on private database material that
  is not included in this public mirror.
- Oracle trajectories are controlled targets for training and evaluation; they
  are not evidence of autonomous model discovery.
- Explanation fluency is not the primary metric. The benchmark prioritizes
  tool use, evidence status, attribution, terminal action, and fail-closed
  policy compliance.
- The completed 25-row private sealed set is a one-time pilot. Its class-level
  estimates are coarse and it must not be reused for tuning or model selection.

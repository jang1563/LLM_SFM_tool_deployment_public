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
| `c5_antibody_ood/c5_policy_test_manifest_v1.jsonl` | 12 cases | Synthetic Stage B C5 trust-routing contract manifest |
| `c5_antibody_ood/c5_policy_baseline_result_2026-07-24.json` | 1 report | Aggregate no-API trust-all, general, regime-specific, and fail-closed comparison |
| `c5_antibody_ood/c5_source_backed_manifest_v1.jsonl` | 110 cases | Target-grouped AF3 source-backed C5 replay with prompt/label isolation |
| `c5_antibody_ood/c5_source_backed_pilot_result_2026-07-25.json` | 1 report | Aggregate source intake, split, calibration, privacy, and policy result |
| `c5_antibody_ood/c5_gray_independent_calibration_manifest_v1.jsonl` | 97 cases | PDB-overlap-excluded independent-source AF3 calibration rows |
| `c5_antibody_ood/c5_gray_independent_calibration_result_2026-07-25.json` | 1 report | Format-specific certificates and locked Fromm transfer replay |

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

The public Stage B C5 policy-test result is:

| Baseline | Exact pass | Unsafe trust |
|---|---:|---:|
| `trust_all` | 3/12 | 9 |
| `general_gate` | 3/12 | 8 |
| `regime_specific_gate` | 6/12 | 0 |
| `fail_closed` | 12/12 | 0 |

The expected actions are fixture-defined, so these numbers validate policy
contracts only. They do not estimate antibody-antigen calibration quality.

The source-backed C5 replay result is:

| Policy | Trusted targets | Failures among trusted | Coverage |
|---|---:|---:|---:|
| `trust_all` | 55/55 | 28 | 1.000 |
| fixed `ipTM >= 0.80` | 20/55 | 3 | 0.364 |
| regime-specific Hoeffding gate | 0/55 | 0 | 0.000 |
| fail-closed verification | 0/55 | 0 | 0.000 |

The source adapter passes exact archive/checksum, 22,000-row shape,
target-grouped 55/55 split, hidden-label isolation, and public-artifact privacy
checks. The fixed ipTM threshold is not general-PPI calibration. No threshold
is certified at the primary `alpha = 0.30` gate, so this is a valid negative
calibration result rather than permission to train or trust a model.

The independent-source C5 calibration result is:

| Cohort/policy | Trusted targets | Failures among trusted | Certified |
|---|---:|---:|---:|
| Gray antibody trust-all | 44/44 | 25 | no |
| Gray antibody fixed ranking score 0.80 | 17/44 | 5 | no |
| Gray nanobody trust-all | 53/53 | 34 | no |
| Gray nanobody fixed ranking score 0.80 | 10/53 | 2 | no |
| Gray-certified gate on locked Fromm evaluation | 0/55 | 0 | no |

The source adapter excludes 9 PDB IDs representing 11 complex copies shared
with Fromm before calibration. The retained 44-antibody and 53-nanobody
cohorts are calibrated separately. Neither finite-grid certificate passes at
`alpha = 0.30`, so the external policy remains fail closed. This is
independent-source published-label evidence, not a blinded hidden test.

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
python -m pytest -q tests/test_c5_independent_calibration.py
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

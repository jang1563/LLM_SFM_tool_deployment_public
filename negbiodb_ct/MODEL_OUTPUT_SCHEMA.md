# NegBioDB-CT Model Output Schema

Prompt-only, SFT, DPO, and RLVR pilots should emit the same compact JSON object.
The evaluator ignores prose outside the first JSON object, but the preferred
format is JSON only.

## Required Shape

```json
{
  "action": "ground|reject|defer|verify|flag|self_answer",
  "evidence_status": "supported|contradicted|insufficient|invalid_value|unknown",
  "tool_calls": [
    "nullatlas_survey_prior_failures",
    "nullatlas_verify_trial_claims",
    "nullatlas_check_value_validity",
    "nullatlas_negative_evidence_completeness"
  ],
  "cited_source_ids": ["NCT00000000"],
  "rationale": "one short optional sentence"
}
```

## Action Semantics

| action | meaning |
| --- | --- |
| `ground` | A clean recorded failure exists; cite the matching NCT. |
| `reject` | The unqualified failed claim is contradicted or mixed. |
| `defer` | Evidence is insufficient; do not assert the claim. |
| `verify` | Related evidence exists, but this exact claim needs verification. |
| `flag` | The retrieved record or value is invalid/impossible; cite the NCT containing the invalid value. |
| `self_answer` | The model answered without the external tool loop. This is usually a baseline/failure mode for CT tasks. |

## Compatibility Aliases

The parser also accepts:

- `terminal_action` instead of `action`,
- `called` instead of `tool_calls`,
- `nct` or `cited_nct` instead of `cited_source_ids`,
- simplified runner tools:
  - `search_failures`,
  - `check_other_indications`,
  - `check_value_validity`.

Aliases are only for compatibility. The preferred post-training trace should use
the full NullAtlas-style tool names so the evaluator can score the ordered loop.

## Tool Profiles

The parser/evaluator supports two CT tool profiles:

| profile | use case | required trace |
| --- | --- | --- |
| `nullatlas_full` | Post-training target trace. | Full four-step NullAtlas-style loop. |
| `native_ct` | Native Anthropic runner with collapsed tools. | `search_failures`; plus `check_other_indications` for `defer`/`verify`. |

Use `native_ct` only to score the current runner fairly. Keep `nullatlas_full` as
the stricter target for SFT/DPO/RLVR trajectories that should explicitly learn
the full evidence loop.

## Current Parser

`negbiodb_ct/model_output.py` converts this JSON into the shared
`llm_sfm_tool_deployment.Trajectory` schema.

Smoke test:

```bash
python3 examples/run_negbiodb_ct_model_output_demo.py
```

## Stage A Prediction JSONL

For Stage A post-training experiments, save model/API outputs first and score
the saved file offline:

```bash
python3 post_training/generate_stage_a_predictions.py \
  --mode self_answer \
  --sft post_training/stage_a_sft_heldout_v1.jsonl \
  --out /tmp/stage_a_self_answer_predictions.jsonl
python3 post_training/evaluate_stage_a_predictions.py \
  --predictions /tmp/stage_a_self_answer_predictions.jsonl \
  --expected-sft post_training/stage_a_sft_heldout_v1.jsonl
```

Each JSONL row should identify the case and include either a full trajectory:

```json
{
  "case_id": "stage_a::000001",
  "source": "local_sft_or_api_run",
  "trajectory": {
    "steps": [
      {
        "name": "nullatlas_survey_prior_failures",
        "arguments": {"drug_id": "<drug_id>", "condition_id": "<condition_id>"},
        "observation": {"status": "completed"}
      }
    ],
    "terminal_action": "ground_with_attribution",
    "predicted_evidence_status": "supported",
    "cited_source_ids": ["NCT00000000"]
  }
}
```

or a compact model output:

```json
{
  "case_id": "stage_a::000001",
  "prediction": {
    "action": "ground",
    "evidence_status": "supported",
    "tool_calls": [
      {
        "name": "nullatlas_survey_prior_failures",
        "arguments": {"drug_id": "<drug_id>", "condition_id": "<condition_id>"}
      }
    ],
    "cited_source_ids": ["NCT00000000"]
  }
}
```

String-only `tool_calls` are accepted for compatibility, but Stage A does not
auto-fill missing query arguments. Those rows can parse while still failing the
`query_filter_missing_required_field` gate.

`generate_stage_a_predictions.py` also supports `--mode openai_chat`, but live
calls require `--allow-live-api` and `OPENAI_API_KEY`. The public path uses only
deterministic no-API modes.

## Prompt Contracts

`generate_stage_a_predictions.py` supports four prompt contracts:

| contract | purpose |
| --- | --- |
| `basic` | Minimal JSON artifact instruction. |
| `stage_a_v2_strict` | Constrains enum values, JSON shape, and citation policy. |
| `stage_a_v3_tool_trace` | Adds an ordered four-tool trace requirement with `drug_id` and `condition_id` arguments for each tool call. |
| `stage_a_v4_canonical_json` | Adds a canonical JSON envelope and explicitly treats `verified`, `valid`, and related non-enum statuses as invalid outputs. |

`stage_a_v3_tool_trace` is a tool/query compliance diagnostic. It does not
expose hidden labels or source IDs. If source IDs are not visible, the model
should leave `cited_source_ids` empty rather than fabricate attribution.
`stage_a_v4_canonical_json` is the next enum/schema diagnostic after the Cayuga
v3 run failed with `evidence_status: verified` in all held-out outputs. The
v4 Cayuga follow-up removes that invalid-status pattern but still fails the
canonical top-level action/JSON envelope gate.

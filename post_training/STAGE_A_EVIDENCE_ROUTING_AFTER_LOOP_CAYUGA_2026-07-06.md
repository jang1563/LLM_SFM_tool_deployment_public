# Stage A Evidence-Conditioned Routing-After-Loop Cayuga Result

Date: 2026-07-06

Purpose: test whether the evidence-conditioned component substrate repairs
post-tool action/evidence/citation routing when observed tool-result content is
visible. This is a component-level free-form JSON generation smoke, not
`tool_query`, DPO, or RLVR.

## Run

- Cluster: Cayuga Slurm GPU job `[omitted]`
- Runner: `post_training/run_stage_a_strict_component_sft_smoke.py`
- Component: `routing_after_loop`
- Decode mode: `freeform`
- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- Prompt contract: `stage_a_v2_evidence_conditioned_component`
- Target dataset: `negbiodb_ct_stage_a_evidence_conditioned_component_targets_v1`
- Train / held-out: 20 train rows, 5 held-out rows
- Training: 20 steps, batch size 1, last transformer layer only

Raw predictions, trainable state, logs, and full evaluator output remain
untracked under ignored `post_training/runs/` in the Cayuga working copy. This
file records only the compact public-safe result.

## Result

| Metric | Value |
| --- | ---: |
| Held-out pass rate | 0/5 |
| Mean score | 0.200 |
| `target_keys` accuracy | 0.0 |
| `enum_validity` accuracy | 0.0 |
| `exact_match` accuracy | 0.0 |
| `tool_query_shape` score | 0.8 |

Violation counts:

| Violation | Count |
| --- | ---: |
| `target_key_mismatch` | 4 |
| `enum_value_invalid` | 4 |
| `target_mismatch` | 4 |
| `prediction_parse_error` | 1 |

## Held-Out Readout

| Case family | Expected action/status | Prediction shape | Score |
| --- | --- | --- | ---: |
| supported_negative_evidence | `ground` / `supported` | `verify` / invalid `valid`, unexpected key | 0.25 |
| contradicted_or_mixed_endpoint_claim | `reject` / `contradicted` | `verify` / invalid `valid`, unexpected key | 0.25 |
| insufficient_evidence | `defer` / `insufficient` | `verify` / invalid `valid`, unexpected key | 0.25 |
| related_evidence_requires_verification | `verify` / `insufficient` | `verify` / invalid `valid`, unexpected key | 0.25 |
| invalid_value_attribution_failure | `flag` / `invalid_value` | malformed JSON | 0.00 |

## Interpretation

This is a negative result. The evidence-conditioned target substrate fixes the
data-interface problem, but free-form `routing_after_loop` generation still
fails schema, enum, and citation-output gates. Visible tool-result content is
not enough if the model can emit invalid evidence-status values, unexpected
keys, or malformed JSON.

Next, keep `tool_query`, DPO, RLVR, release tagging, and Hugging Face
publication gated. Before method escalation, test a constrained routing readout
or split routing into smaller deterministic components: action/status selection
first, citation selection second.

## Trace

- Compact JSON summary:
  `post_training/stage_a_evidence_routing_after_loop_cayuga_summary_2026-07-06.json`
- Raw `report.json` SHA-256:
  `d050a35c3d49492c73d4623fc4a35cfa7ab7f6b904b439be41372ab232ec18bb`
- Raw `eval_report.json` SHA-256:
  `b700c4e172e389c054cc54f31d5ac0c0c322902c28bf038dcc025825c27f685e`
- Raw `predictions.jsonl` SHA-256:
  `12a1b37b80a1f878b3f95514e28f466721c861865cb9cb6cddf03f7b924ebb48`

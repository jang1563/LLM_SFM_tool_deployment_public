# Stage A Strict-Contract Component Diagnostics

Date: 2026-07-04

Purpose: decompose the negative strict-contract SFT smoke result before moving
to DPO or RLVR. The diagnostic uses deterministic counterfactual prediction
variants derived from the five held-out strict-contract targets, then scores
each variant with the same offline Stage A trajectory evaluator used for API,
prompt-only, and cluster SFT artifacts.

No model weights are loaded and no API calls are made.

## Result

| Variant | Passed | Mean score | Readout |
| --- | ---: | ---: | --- |
| `oracle_full` | 5/5 | 1.000 | Positive control: enum, tool loop, citations, and routing are all sufficient when correct. |
| `invalid_enum_verified` | 0/5 | 0.000 | A single disallowed enum value causes parse-time failure in every case. |
| `enum_constrained_from_action` | 5/5 | 1.000 | Constrained enum/action decoding can remove the observed `verified` failure mode when other fields are correct. |
| `route_only_correct_no_tools` | 0/5 | 0.714 | Correct action/status is not enough without the required external tool loop. |
| `ordered_tool_names_only` | 0/5 | 0.857 | Ordered tool names are not enough; required query fields remain a hard gate. |
| `tool_loop_with_ground_route` | 1/5 | 0.714 | A full structured tool loop is not enough if evidence/action routing collapses to `ground`/`supported`. |

## Interpretation

The failed Cayuga strict SFT smoke should not be treated as a reason to jump
directly to DPO or RLVR. The held-out gates separate into three trainable
diagnostics:

- constrained enum/action decoding;
- ordered tool calls with required `drug_id` and `condition_id` arguments;
- evidence-status and terminal-action routing after the tool loop exists.

The high score for `ordered_tool_names_only` is especially useful: it shows that
a model can look close under aggregate score while still failing every held-out
case because query arguments are missing. The benchmark should continue to
report pass rate and per-gate accuracy, not only mean score.

## Trace

- Runner: `post_training/run_stage_a_strict_component_diagnostics.py`
- Compact JSON summary:
  `post_training/stage_a_strict_component_diagnostics_summary_2026-07-04.json`
- Held-out split:
  `post_training/stage_a_strict_contract_sft_heldout_v1.jsonl`
- Prompt contract: `stage_a_v2_strict`

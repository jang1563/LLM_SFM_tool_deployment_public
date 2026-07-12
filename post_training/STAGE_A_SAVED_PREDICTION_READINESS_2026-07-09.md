# Stage A Saved-Prediction Readiness

Purpose: compare public deterministic saved-output smokes and compact
real Cayuga saved-output summaries against the full-trajectory
arbitration scorecard.

## Held-out Scorecard

| Baseline | Result |
| --- | --- |
| `collapse` | 1/5 pass; mean 0.714 |
| `citationless routing` | 3/5 pass; mean 0.943 |
| `runtime gate` | 5/5 pass; mean 1.000 |

## Saved Output Reports

| Report | Source | Candidate? | Result | Parse errors | Beats citationless? |
| --- | --- | ---: | --- | ---: | ---: |
| `deterministic_saved_oracle` | `deterministic_public_smoke` | no | 5/5 pass; mean 1.000 | 0 | yes |
| `deterministic_saved_self_answer` | `deterministic_public_smoke` | no | 0/5 pass; mean 0.229 | 0 | no |
| `deterministic_compact_tool_names_oracle` | `deterministic_public_smoke` | no | 0/5 pass; mean 0.857 | 0 | no |
| `stage_a_cayuga_qwen15b_heldout_2026-07-04_r2` | `real_saved_model_compact_summary` | yes | 0/5 pass; mean 0.114 | 4 | no |
| `stage_a_cayuga_qwen15b_strict_heldout_2026-07-04` | `real_saved_model_compact_summary` | yes | 0/5 pass; mean 0.372 | 0 | no |
| `stage_a_strict_sft_cayuga_qwen05b_2026_07_04` | `real_saved_sft_compact_summary` | yes | 0/5 pass; mean 0.000 | 5 | no |
| `stage_a_strict_sft_cayuga_qwen15b_2026_07_04` | `real_saved_sft_compact_summary` | yes | 0/5 pass; mean 0.372 | 0 | no |
| `stage_a_v3_tool_trace_qwen05b_2026_07_09_cache` | `real_saved_model_compact_summary` | yes | 0/5 pass; mean 0.000 | 5 | no |
| `stage_a_v4_canonical_json_qwen05b_2026_07_09` | `real_saved_model_compact_summary` | yes | 0/5 pass; mean 0.000 | 5 | no |
| `stage_a_saved_candidate_readout_qwen05b_train_observed_2026_07_09` | `real_saved_model_compact_summary` | yes | 0/5 pass; mean 0.657 | 0 | no |
| `stage_a_saved_candidate_readout_qwen05b_all_valid_2026_07_09` | `real_saved_model_compact_summary` | yes | 0/5 pass; mean 0.657 | 0 | no |

## Saved-Candidate Gates

| Gate | Policy | Trusted | Unsafe trust | Strict final | Beats citationless? |
| --- | --- | ---: | ---: | --- | ---: |
| `stage_a_saved_candidate_readout_qwen05b_train_observed_2026_07_09` | `train_observed_pairs` | 1/5 | 0 | 2/5 | no |
| `stage_a_saved_candidate_readout_qwen05b_all_valid_2026_07_09` | `all_valid_pairs` | 1/5 | 0 | 2/5 | no |

Best fail-closed candidate gate:
- `stage_a_saved_candidate_readout_qwen05b_all_valid_2026_07_09` trusts 1/5 rows with 0 unsafe trust and 2/5 strict final correct.

## Decision

- Ready for `tool_query`: `False`
- Ready for DPO/RLVR: `False`
- Runtime enforcement required: `True`

Blockers:
- Best real saved output does not beat the collapse baseline.
- Best real saved output does not beat the citationless routing baseline.
- Best real saved output remains below runtime-gate full-trajectory performance.
- Best saved-candidate gate remains below citationless routing.
- Best saved-candidate gate remains below runtime-gate full-trajectory performance.

Use the saved-prediction scorer and saved-candidate gate analyzer for the next real Cayuga output, then compare compact reports here. Do not reopen tool_query, DPO/RLVR, Hugging Face publication, or release tagging until a real saved model output and its fail-closed gate beat collapse/citationless baselines and approach the runtime full-trajectory gate.

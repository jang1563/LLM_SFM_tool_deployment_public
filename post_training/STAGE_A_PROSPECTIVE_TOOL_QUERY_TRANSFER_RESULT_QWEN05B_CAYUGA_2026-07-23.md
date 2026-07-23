# Stage A Prospective Real-Query Tool-Query Transfer

Scope: base versus pre-prospective frozen tool-query SFT on 25 public
development prompts with case-specific drug and condition identifiers.
No prospective-row training or live-tool execution was performed.

## Policy Summary

| Policy | Parseable | Tool sequence | Query fields | Query values | Exact |
| --- | ---: | ---: | ---: | ---: | ---: |
| `base` | 20/25 | 0/25 | 0/25 | 0/25 | 0/25 |
| `frozen_tool_query_sft` | 14/25 | 0/25 | 0/25 | 0/25 | 0/25 |

## Decision

- Frozen SFT improves exact transfer: `false`.
- New training, DPO/RLVR, and Hugging Face publication remain closed.
- Raw generations and the trainable state remain private and uncommitted.
- This development result is not an independent sealed-test estimate.

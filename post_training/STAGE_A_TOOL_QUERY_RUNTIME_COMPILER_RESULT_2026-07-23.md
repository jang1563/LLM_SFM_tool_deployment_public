# Stage A Tool-Query Runtime Compiler Evaluation

The current Stage A query step uses a fixed four-tool sequence and copies
two typed visible identifiers. This report treats it as a fail-closed
runtime contract rather than a learned policy.

## Model Diagnostics

| Policy | Parseable | Target keys | Shape | Query values | Exact |
| --- | ---: | ---: | ---: | ---: | ---: |
| `base_minimal_contract` | 20/25 | 0/25 | 0/25 | 0/25 | 0/25 |
| `frozen_placeholder_sft` | 14/25 | 0/25 | 0/25 | 0/25 | 0/25 |
| `base_explicit_contract` | 25/25 | 25/25 | 0/25 | 0/25 | 0/25 |

## Runtime Result

- Clean exact compilation: `25/25`.
- Malformed inputs rejected: `150/150`.
- Rejections used the intended contract reason: `150/150`.
- Corrective SFT is not selected for this deterministic operation.
- DPO/RLVR and Hugging Face publication remain closed.

This is a public-development systems result, not a sealed-test estimate
or a live-tool execution result.

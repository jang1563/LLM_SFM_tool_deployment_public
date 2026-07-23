# Stage A Tool-Query Component SFT Result

Purpose: close the missing Stage A tool-query component checkpoint with
a compact public-safe Cayuga result.

## Result

- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- Held-out pass: 0/5
- Mean score: 0.25
- Prompt-schema outputs: 5/5
- Tool-call key present: 0/5
- Passes schema gate: `False`
- Gate violations: `["below_exact_pass_requirement", "target_keys_below_requirement", "tool_query_shape_below_requirement", "exact_match_below_requirement"]`

## Scope

- Unique train targets: 1
- Unique held-out targets: 1
- Shared target hashes: 1
- Actual identifier resolution evaluated: `False`
- Actual tool execution evaluated: `False`

This is an ordered tool-call and placeholder-schema diagnostic. It does
not measure drug/condition identifier resolution or live tool execution.
Raw predictions, prompts, model state, scheduler logs, and private paths
are not included in this artifact.

# Stage A Routing Defer-vs-Verify Cayuga Result

Date: 2026-07-08

Run: `stage_a_routing_defer_verify_cayuga_2026_07_08`

Job: `[omitted]`

Model: `Qwen/Qwen2.5-0.5B-Instruct`

## Boundary

This smoke tests the targeted routing boundary created after the previous
candidate-rank result: `defer` / `insufficient` versus `verify` /
`insufficient`.

The run uses 8 train pairs and 2 held-out pairs from
`stage_a_routing_defer_verify_contrast_pairs_v1.jsonl`. It is a component
diagnostic, not a full trajectory result and not a DPO/RLVR result.

## Result

Teacher-forced held-out margin:

- Base: 1/2 wins, mean margin 0.013428.
- Trained: 1/2 wins, mean margin 0.006464.
- Newly won: 0.
- Newly lost: 0.

Per-boundary margin:

- `defer` / `insufficient`: -0.154711 -> -0.023557.
- `verify` / `insufficient`: 0.181567 -> 0.036484.

Finite-candidate routing:

- Base exact top-1: 1/2, mean gold rank 1.5.
- Trained exact top-1: 1/2, mean gold rank 1.5.
- Top predicted pair is `verify` / `insufficient` for both held-out cases.

## Interpretation

This is a negative/partial result. The targeted smoke moves the
`defer` / `insufficient` margin toward zero, but it does not cross the decision
boundary. The trained finite-candidate readout still ranks
`verify` / `insufficient` above `defer` / `insufficient` for the unresolved
insufficient-evidence held-out case.

## Next Decision

Keep `tool_query`, DPO, RLVR, release tagging, and Hugging Face publication
gated. The next research step should diagnose boundary calibration or a
fail-closed routing gate before adding broader data or changing the optimization
method.

Raw JSONL reports, model state, and scheduler logs remain uncommitted under
ignored `post_training/runs/` in the Cayuga working copy.

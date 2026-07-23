# Stage A Sealed Candidate-Routing Result

Purpose: evaluate the frozen Qwen2.5-0.5B evidence-conditioned routing policy
once on the private source-separated Stage A extension.

## Result

- Sealed rows: 25
- Exact action/status pair: 5/25 (20%)
- Best static single-pair prior: 5/25
- Deterministic runtime oracle gate: 25/25
- Predicted pair: `verify/insufficient` on 25/25 rows
- Incorrect `ground/supported` predictions: 0
- Beats static prior: `False`
- Matches runtime oracle: `False`

Per target family, the policy was correct only for the five
`verify/insufficient` rows. It scored 0/5 for each of
`ground/supported`, `reject/contradicted`, `defer/insufficient`, and
`flag/invalid_value`.

## Interpretation

The independent result reproduces the exposed-development collapse rather than
showing source-separated routing generalization. The saved policy is
conservative in the narrow sense that it does not incorrectly select
`ground/supported`, but it does not discriminate among visible evidence states.
Runtime evidence arbitration remains the deployment baseline.

This evaluation measures finite-candidate routing after a synthetic oracle tool
loop. Hidden labels were used to construct and score that oracle state, but
were not visible to the model. It does not evaluate real drug/condition
identifier resolution, live database execution, or end-to-end tool use.

## One-Time Boundary

The sealed manifest and frozen trainable-state hashes match their public
commitments. All 25 model scores were produced once. The initial job completed
scoring but stopped during compact-result validation because the safety check
incorrectly rejected an aggregate `rows` key. The committed result was
recovered from those completed private predictions without rescoring the model;
the one-time lock is complete.

Do not tune on these 25 rows or repeat this sealed evaluation. DPO/RLVR and
Hugging Face publication remain closed. The next scientifically useful step is
to design a prospective larger evaluation with real query values and to test a
runtime-enforced hybrid rather than continuing to optimize this collapsed
policy on exposed cases.

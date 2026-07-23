# Stage A Candidate-Routing Policy Freeze

Purpose: freeze the exact saved candidate-routing policy before the
one-time source-separated sealed evaluation.

## Frozen Policy

- Freeze ID: `stage_a_candidate_routing_freeze::456bce64145f34c4db60c5aa590c556efd06211645758620deca44b7dca14fc8`
- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- Model revision: `7ae557604adf67be50417f59c2c2f167def9a775`
- Candidate pairs: `["ground/supported", "reject/contradicted", "defer/insufficient", "verify/insufficient", "flag/invalid_value"]`
- Training steps: 40
- Max length: 1536
- Saved-state load only: `True`

## Pre-Freeze Results

- Exposed-development candidate routing: 1/5
- Candidate-routing gate passed: `False`
- Tool-query schema gate passed: `False`

## Boundary

- Explicit original training seed recorded: `False`
- The sealed evaluation loads the hashed saved trainable state and does not retrain.
- The sealed run evaluates routing after a synthetic oracle tool loop;
  it does not evaluate identifier resolution or live tool execution.
- Repeated sealed evaluation and training on sealed rows are prohibited.

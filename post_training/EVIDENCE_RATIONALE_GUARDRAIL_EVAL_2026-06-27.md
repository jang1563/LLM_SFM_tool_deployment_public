# Evidence-Rationale Guardrail Evaluation: 2026-06-27

This file evaluates a deterministic evidence-derived final-action override on the normal boundary-rationale held-out runs.
The guardrail reads only visible native CT tool observations and does not read the gold action label.

## Commands

```bash
python3 post_training/evaluate_evidence_guardrail.py
```

## Aggregate

| source eval | model acc | guardrail acc | model defer | guardrail defer | rescued | introduced | action changes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| heldout_decision_eval.json | 0.500 | 1.000 | 0/8 | 8/8 | 20 | 0 | 20 |
| heldout_constrained_loaded.json | 0.500 | 1.000 | 0/8 | 8/8 | 20 | 0 | 20 |

Failure pairs after guardrail:

```json
{
  "heldout_constrained_loaded.json": {},
  "heldout_decision_eval.json": {}
}
```

## Interpretation

- The guardrail is deterministic and uses the same visible tool observations available to the model.
- If guardrail accuracy is 1.000 with no introduced errors, the normal held-out failure is fully routable by an external evidence-boundary layer.
- This supports evaluating the layer as a deployable routing/override component before spending effort on broader DPO or RLVR.

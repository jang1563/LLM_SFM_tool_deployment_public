# SFT Boundary-Rationale Held-Out Evidence-Rationale Ablation: 2026-06-27

This file records an eval-only ablation using the already-trained boundary-rationale fold states.
The ablation inserts a deterministic `BOUNDARY_RATIONALE` prompt derived only from visible held-out tool observations before the final `submit_decision` target.
Raw run artifacts are under `post_training/runs/` and ignored by git.

## Commands

```bash
python3 post_training/run_sft_boundary_rationale_ablation.py
python3 post_training/summarize_sft_boundary_rationale_ablation.py
```

Eval settings:

```text
model = Qwen/Qwen2.5-0.5B-Instruct
source_state_root = post_training/runs/qwen_sft_cv4_boundary_rationale_schema_action_80_evalfast
batch_size = 2
max_length = 512
score_mode = mean
```

## Artifact Summary

```text
dataset = negbiodb_ct_native_sft_boundary_rationale_heldout_evidence_v1
strategy = heldout_evidence_boundary_rationale_ablation_v1
rationale_mode = evidence
source_boundary_manifest = post_training/negbiodb_ct_native_sft_cv4_boundary_rationale_manifest.json
```

| fold | heldout rows | heldout by class | heldout by role | evidence mismatches |
| --- | --- | --- | --- | --- |
| 0 | 10 | {"defer": 2, "flag": 2, "ground": 2, "reject": 2, "verify": 2} | {"evidence_rationale": 10} | 0 |
| 1 | 10 | {"defer": 2, "flag": 2, "ground": 2, "reject": 2, "verify": 2} | {"evidence_rationale": 10} | 0 |
| 2 | 10 | {"defer": 2, "flag": 2, "ground": 2, "reject": 2, "verify": 2} | {"evidence_rationale": 10} | 0 |
| 3 | 10 | {"defer": 2, "flag": 2, "ground": 2, "reject": 2, "verify": 2} | {"evidence_rationale": 10} | 0 |

## Aggregate Comparison

| condition | strict mean | constrained loaded mean | parse failures | defer | takeaway |
| --- | --- | --- | --- | --- | --- |
| boundary rationale, normal held-out | 0.500 | 0.500 | 0 | 0/8 | best native-SFT aggregate but defer collapsed |
| held-out evidence-rationale ablation | 1.000 | 1.000 | 0 | 8/8 | tests whether inference-time rationale rescues defer |

## Ablation Folds

| fold | heldout loss | strict acc | constrained loaded | strict by class |
| --- | --- | --- | --- | --- |
| 0 | 0.0041 | 1.000 | 1.000 | defer 2/2, flag 2/2, ground 2/2, reject 2/2, verify 2/2 |
| 1 | 0.0026 | 1.000 | 1.000 | defer 2/2, flag 2/2, ground 2/2, reject 2/2, verify 2/2 |
| 2 | 0.0034 | 1.000 | 1.000 | defer 2/2, flag 2/2, ground 2/2, reject 2/2, verify 2/2 |
| 3 | 0.0063 | 1.000 | 1.000 | defer 2/2, flag 2/2, ground 2/2, reject 2/2, verify 2/2 |

Aggregate:

```text
heldout_loss_mean = 0.0041
strict_action_accuracy_mean = 1.000
strict_action_accuracy_range = 1.000..1.000
strict_parse_failures_total = 0
strict_class_accuracy = defer 8/8, flag 8/8, ground 8/8, reject 8/8, verify 8/8
constrained_loaded_accuracy_mean = 1.000
constrained_loaded_accuracy_range = 1.000..1.000
constrained_loaded_class_accuracy = defer 8/8, flag 8/8, ground 8/8, reject 8/8, verify 8/8
```

Failure pair counts:

```json
{
  "constrained_loaded": {},
  "strict": {}
}
```

## Interpretation

- This is a tool-derived rationale ablation: the rationale is generated from visible tool outputs, not from the gold action label.
- If this condition rescues `defer`, then the main missing component is a deployable boundary-rationale generator or policy layer at inference time.
- If it fails, explicit `defer` versus `verify` preference supervision is still needed even with visible rationale hints.
- This condition is more deployable than oracle rationale, but it still relies on a deterministic rule-rationale preprocessor.

## Next Action

Use this result to decide whether the deterministic evidence-rationale layer should become a deployable guardrail/routing component, or whether its outputs should instead be distilled into preference/SFT data.

# Evidence-Rationale Pilot-400 Stress Test: 2026-06-27

This file records a full-pilot stress test of the reusable evidence-rationale
layer on the deterministic-oracle CT SFT artifact.

## Command

```bash
python3 post_training/apply_evidence_rationale.py \
  --sft post_training/negbiodb_ct_oracle_sft_v1.jsonl \
  --out post_training/negbiodb_ct_oracle_sft_evidence_rationale_v1.jsonl \
  --manifest-out post_training/negbiodb_ct_oracle_sft_evidence_rationale_manifest.json \
  --dataset negbiodb_ct_oracle_sft_evidence_rationale_v1 \
  --strategy pilot400_evidence_boundary_rationale_stress_v1
```

## Artifact

```text
source = post_training/negbiodb_ct_oracle_sft_v1.jsonl
out = post_training/negbiodb_ct_oracle_sft_evidence_rationale_v1.jsonl
manifest = post_training/negbiodb_ct_oracle_sft_evidence_rationale_manifest.json
dataset = negbiodb_ct_oracle_sft_evidence_rationale_v1
strategy = pilot400_evidence_boundary_rationale_stress_v1
examples = 400
by_action_class = defer 120, flag 40, ground 140, reject 40, verify 60
by_evidence_action = defer 120, flag 40, ground 140, reject 40, verify 60
by_role = evidence_rationale 400
evidence_action_matches = 400
evidence_action_mismatches = 0
evidence_action_unlabeled = 0
```

## Boundary

This is a stress test on deterministic oracle-policy SFT data, not live runner
behavior. It tests whether the visible-tool evidence rule generalizes across
the full pilot task set.

## Interpretation

The rule is not an n=40 artifact. Across the full 400-task pilot surface, the
evidence-derived action matches the deterministic oracle action 400/400.

The next technical choice is either to stress-test the same layer on new live or
non-oracle model outputs as they become available, or to distill this rule into
SFT/preference data.

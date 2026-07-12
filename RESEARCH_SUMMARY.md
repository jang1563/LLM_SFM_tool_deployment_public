# Research Summary: LLM-SFM Tool Deployment

## Research Question

How should a biology tool-use agent decide when to trust returned evidence,
verify it, reject it, or defer? This repository tests those deployment-time
decisions with deterministic evaluators and public-safe benchmark artifacts.

## Approach

- A trajectory evaluator scores action choice, tool use, source citation, and
  defer/verify behavior.
- A Stage A benchmark manifest separates model-visible tasks from hidden
  evaluator metadata, then exports oracle SFT, preference, and
  process-supervision trajectories.
- A NegBioDB-CT harness creates balanced drug-condition tasks for ground,
  reject, verify, defer, and flag decisions.
- Post-training pipelines export SFT and preference-style data from clean
  trajectories, then validate schema and evidence fields.
- A2 free-text resolution maps noisy drug and disease strings into ChEMBL and
  MONDO concepts, with a closed-set band reranker for ambiguous disease
  matches.

## Measured Highlights

| Component | Result |
|---|---|
| Stage A benchmark substrate | 25 public-safe manifest cases; oracle 25/25 pass; shortcut baselines 0/25 pass. |
| Stage A post-training artifacts | 25 SFT rows, 150 preference pairs, 25 process rows; 20/5 train-heldout split with no source overlap. |
| Evidence-rationale guardrail | 0.50 model baseline to 1.00 guarded score on the stress slice; 20 rescued, 0 introduced. |
| A2 disease ambiguous band | SapBERT 0.750; Qwen2.5-7B 0.810; SFT-1.5B 0.875; Claude-haiku 0.900; ceiling 0.970. |
| A2 GRPO diagnostic | From-base GRPO improved full pick-or-abstain accuracy through abstention, but did not improve hard in-band disambiguation. |
| Data hygiene | The post-training validator reports no schema issues on the tracked artifacts. |

## Public Demo

```bash
python examples/run_public_demo.py
```

The demo uses synthetic trajectory records and shows both passing and failing
policy paths without private databases, API keys, or real trial rows.

## Claim Boundary

- Results are benchmark and diagnostic findings, not clinical evidence.
- Compact cluster summaries are retained; raw predictions, model state,
  scheduler logs, and run directories are excluded.
- Positive teacher-forced or component-level movement is not treated as a
  deployment-ready repair without candidate-selection and trajectory-level
  validation.
- Runtime evidence arbitration remains the fail-closed baseline that learned
  policies must meet or beat.

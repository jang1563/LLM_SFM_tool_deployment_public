# Stage A Evidence-Conditioned Routing Observed-Pair Cayuga Result

Date: 2026-07-08

Purpose: test whether constrained candidate scoring can separate
`routing_after_loop` schema/enum/citation readout from free-form JSON generation
after observed tool-result content is visible. This is a component-level SFT
smoke, not `tool_query`, DPO, or RLVR.

## Run

- Cluster: Cayuga Slurm GPU job `[omitted]`
- Runner: `post_training/run_stage_a_strict_component_sft_smoke.py`
- Component: `routing_after_loop`
- Decode mode: `routing_observed_pair_score`
- Candidate policy: train-observed routing action/status pairs with citations
  attached only from model-visible tool-result content
- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- Prompt contract: `stage_a_v2_evidence_conditioned_component`
- Target dataset: `negbiodb_ct_stage_a_evidence_conditioned_component_targets_v1`
- Train / held-out: 20 train rows, 5 held-out rows
- Training: 20 steps, batch size 1, last transformer layer only

Raw predictions, trainable state, logs, and full candidate-score tables remain
untracked under ignored `post_training/runs/` in the Cayuga working copy. This
file records only the compact public-safe result.

## Result

| Metric | Value |
| --- | ---: |
| Held-out pass rate | 2/5 |
| Mean score | 0.850 |
| `target_keys` accuracy | 1.0 |
| `enum_validity` accuracy | 1.0 |
| `exact_match` accuracy | 0.4 |
| `tool_query_shape` score | 1.0 |

Violation counts:

| Violation | Count |
| --- | ---: |
| `target_mismatch` | 3 |

## Candidate-Rank Readout

| Metric | Value |
| --- | ---: |
| Candidate space size | 5 |
| Full target top-1 | 2/5 |
| Action/status top-1 | 2/5 |
| Citation-required exact top-1 | 1/2 |
| Mean full-target rank | 2.0 |
| Mean action/status rank | 2.0 |
| Mean top-gold margin | 0.044718 |

Top predicted action/status counts:

| Top pair | Count |
| --- | ---: |
| `ground` / `supported` | 2 |
| `reject` / `contradicted` | 3 |

## Held-Out Readout

| Case family | Expected | Top candidate | Full rank | Pair rank | Margin | Score |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| supported_negative_evidence | `ground` / `supported` / `NCT00588770` | `ground` / `supported` / `NCT00588770` | 1 | 1 | 0.0 | 1.00 |
| contradicted_or_mixed_endpoint_claim | `reject` / `contradicted` / none | `reject` / `contradicted` / none | 1 | 1 | 0.0 | 1.00 |
| insufficient_evidence | `defer` / `insufficient` / none | `reject` / `contradicted` / none | 4 | 4 | 0.115769 | 0.75 |
| related_evidence_requires_verification | `verify` / `insufficient` / none | `reject` / `contradicted` / none | 2 | 2 | 0.063698 | 0.75 |
| invalid_value_attribution_failure | `flag` / `invalid_value` / `NCT00828178` | `ground` / `supported` / `NCT00828178` | 2 | 2 | 0.044123 | 0.75 |

## Interpretation

This is a positive component readout, but not a full repair. Constrained routing
fixes schema and enum validity compared with the free-form `routing_after_loop`
run, improving held-out pass rate from 0/5 to 2/5 and mean score from 0.200 to
0.850.

The remaining failures are not citation-only failures. In the insufficient,
verification-needed, and invalid-value families, the top action/status pair is
wrong. The model still over-routes to `reject` / `contradicted` or `ground` /
`supported` when the expected route is `defer` / `insufficient`, `verify` /
`insufficient`, or `flag` / `invalid_value`.

Next, keep `tool_query`, DPO, RLVR, release tagging, and Hugging Face
publication gated. The next repair should target evidence-conditioned
action/status routing for insufficient, verify, and invalid-value families
before preference optimization or audited RLVR.

## Trace

- Compact JSON summary:
  `post_training/stage_a_evidence_routing_observed_pair_cayuga_summary_2026-07-08.json`
- Raw `report.json` SHA-256:
  `f50bdf5ff42656e30932b53667c1fc4124263a3a34d87798afde5690807be9e1`
- Raw `eval_report.json` SHA-256:
  `5679c4c78ac48f0485ca048e044fc4564234840d18528a7b5e8ab77c520f81f8`
- Raw `predictions.jsonl` SHA-256:
  `6f08d1e0e3db3cc550e2d70aaa092a405a898afb0649d175ba39ba1b16791c4d`
- Compact rank JSON SHA-256:
  `d579b48a94e311e7594db9b74583b55f830d86774e17f041831ed70d387392bc`

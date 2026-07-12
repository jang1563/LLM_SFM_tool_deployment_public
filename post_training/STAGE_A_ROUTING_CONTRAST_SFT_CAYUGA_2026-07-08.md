# Stage A Routing Contrast SFT/Margin Cayuga Diagnostic

Date: 2026-07-08

Purpose: test whether targeted SFT plus deterministic chosen-over-rejected
margin pressure can repair the unresolved evidence-conditioned
`routing_after_loop` action/status failures: `defer` / `insufficient`,
`verify` / `insufficient`, and `flag` / `invalid_value`. This is a
component-level teacher-forced margin diagnostic, not `tool_query`, DPO, RLVR,
or a free-form trajectory result.

## Setup

| Setting | Value |
| --- | --- |
| Cluster job | Cayuga Slurm GPU job `[omitted]` |
| Runner | `post_training/run_stage_a_routing_contrast_sft_smoke.py` |
| Component | `routing_after_loop` |
| Prompt contract | `stage_a_v2_evidence_conditioned_component` |
| Model | `Qwen/Qwen2.5-0.5B-Instruct` |
| Contrast axis | `action_status` |
| Pairwise margin weight | 1.0 |
| Required pairwise margin | 0.05 |
| Log-probability mode | mean |
| Unique train pairs | 12 |
| Held-out pairs | 3 |
| Training | 20 steps, batch size 1, last transformer layer only |

Raw margin JSONL files, full reports, trainable state, and Slurm logs remain
untracked under ignored `post_training/runs/` in the Cayuga working copy. This
file records only the compact public-safe result.

## Result

| Metric | Base model | Pairwise-margin SFT |
| --- | ---: | ---: |
| Held-out wins | 0/3 | 3/3 |
| Mean held-out margin | -0.116919 | 0.114900 |
| Minimum held-out margin | -0.234553 | 0.048288 |
| Chosen-not-above-rejected violations | 3 | 0 |

Mean held-out margin delta is 0.231818. This repairs the teacher-forced
action/status contrast slice for the three constrained-routing failure
families, but it does not yet prove free-form routing, finite-candidate
selection, `tool_query`, or trajectory-level repair.

## Held-Out Readout

| Case family | Chosen pair | Rejected pair | Base margin | Trained margin | Delta | Outcome |
| --- | --- | --- | ---: | ---: | ---: | --- |
| insufficient_evidence | `defer` / `insufficient` | `reject` / `contradicted` | -0.234553 | 0.126487 | 0.361040 | newly_won |
| related_evidence_requires_verification | `verify` / `insufficient` | `reject` / `contradicted` | -0.095973 | 0.169924 | 0.265897 | newly_won |
| invalid_value_attribution_failure | `flag` / `invalid_value` | `ground` / `supported` | -0.020230 | 0.048288 | 0.068518 | newly_won |

## Train-Pair Check

| Chosen pair | Train wins | Mean train margin | Min train margin |
| --- | ---: | ---: | ---: |
| `defer` / `insufficient` | 4/4 | 0.102002 | 0.095493 |
| `flag` / `invalid_value` | 4/4 | 0.039961 | 0.034792 |
| `verify` / `insufficient` | 4/4 | 0.152987 | 0.134062 |

## Interpretation

The supervised pairwise-margin objective gives a clean positive signal on the
specific routing action/status contrasts generated from the constrained
routing failures. Base model margins lose on all three held-out families; after
20 steps, all three become positive, and all train-pair families pass.

This is deliberately narrow. It shows that the unresolved constrained-routing
families are trainable under a deterministic margin slice, not that the system
can yet generate the right routing decision in a full trajectory. The next
gate should be a finite-candidate or saved-prediction routing readout using the
same evidence-conditioned schema. `tool_query`, DPO, RLVR, release tagging, and
Hugging Face publication should remain gated until routing stability is shown
outside teacher-forced margin scoring.

## Trace

- Compact JSON summary:
  `post_training/stage_a_routing_contrast_sft_cayuga_summary_2026-07-08.json`
- Runner: `post_training/run_stage_a_routing_contrast_sft_smoke.py`
- Raw reports, margin JSONL files, model state, and Slurm logs remain
  untracked under ignored `post_training/runs/` in the cluster working copy.
- Run report SHA-256:
  `0681c390445d78ca07c3970d850ec384299b40059984f970fc9203d3abe63dea`
- Base margin report SHA-256:
  `2d62ea410f078492a03a9ec8624a341936edda4b713ef50936150219ec3d6b49`
- Held-out margin report SHA-256:
  `c28b60f4454f19132809438e3e7b38821191b7f2c3b709d4f3a3cd84c9a7fdbf`
- Margin delta report SHA-256:
  `d80f151c46e7696e84d6a86732cbdb5d578e5e827f82980eca2136d8937fc6b9`

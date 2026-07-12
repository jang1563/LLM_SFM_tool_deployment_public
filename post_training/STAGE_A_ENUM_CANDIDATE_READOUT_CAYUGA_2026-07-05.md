# Stage A Enum Candidate-Selection Cayuga Readout

Date: 2026-07-05

Purpose: test whether the positive full-target pairwise-margin result transfers
from binary chosen-vs-rejected scoring to finite-candidate enum selection.
This is still component-level scoring, not free generation, `tool_query`,
routing-after-loop, DPO, or RLVR.

## Setup

| Setting | Value |
| --- | --- |
| Target format | `full` |
| Pairwise margin weight | 1.0 |
| Required pairwise margin | 0.05 |
| Focus pair | `flag` / `invalid_value` |
| Focus repeat | 4 |
| Held-out pairs | 4 |
| Candidate policies | `all_valid_pairs`, `pair_observed_outputs` |

The same pairwise-margin training setup still repairs teacher-forced margins:
base 0/4 margin wins become trained 4/4 margin wins with mean margin
-0.116510 -> 0.177690.

## Candidate Result

| Policy | Candidate space | Base top-1 | Trained top-1 | Base mean gold rank | Trained mean gold rank | Base mean top-gold margin | Trained mean top-gold margin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `all_valid_pairs` | 30 | 0/4 | 0/4 | 15.00 | 7.00 | 0.209262 | 0.108421 |
| `pair_observed_outputs` | 5 | 0/4 | 0/4 | 3.50 | 2.75 | 0.133635 | 0.067083 |

The model moves the gold candidates closer to the top but does not select them
as top-1.

## All-Valid 30-Way Readout

| Case family | Target | Trained top candidate | Base rank | Trained rank | Trained margin |
| --- | --- | --- | ---: | ---: | ---: |
| contradicted_or_mixed_endpoint_claim | `reject` / `contradicted` | `verify` / `invalid_value` | 7 | 3 | 0.046163 |
| insufficient_evidence | `defer` / `insufficient` | `verify` / `invalid_value` | 24 | 7 | 0.143517 |
| related_evidence_requires_verification | `verify` / `insufficient` | `verify` / `invalid_value` | 9 | 3 | 0.060237 |
| invalid_value_attribution_failure | `flag` / `invalid_value` | `verify` / `invalid_value` | 20 | 15 | 0.183768 |

## Pair-Observed 5-Way Readout

| Case family | Target | Trained top candidate | Base rank | Trained rank | Trained margin |
| --- | --- | --- | ---: | ---: | ---: |
| contradicted_or_mixed_endpoint_claim | `reject` / `contradicted` | `verify` / `insufficient` | 2 | 2 | 0.008031 |
| insufficient_evidence | `defer` / `insufficient` | `verify` / `insufficient` | 5 | 3 | 0.112909 |
| related_evidence_requires_verification | `verify` / `insufficient` | `reject` / `contradicted` | 3 | 2 | 0.001932 |
| invalid_value_attribution_failure | `flag` / `invalid_value` | `verify` / `insufficient` | 4 | 4 | 0.145460 |

## Interpretation

This is a useful negative/partial result. The pairwise-margin objective fixes
the binary chosen-over-`ground` contrast, but finite-candidate selection still
collapses to the wrong enum pair. The narrower 5-way policy produces near-misses
for `reject` and `verify`, but `defer` and `flag` remain below top-1.

Do not move to `tool_query`, DPO, or RLVR from this result alone. The next
`enum_action` repair should explicitly train or calibrate finite-candidate
selection before claiming the component is solved.

## Trace

- Compact JSON summary:
  `post_training/stage_a_enum_candidate_readout_cayuga_summary_2026-07-05.json`
- Runner: `post_training/run_stage_a_enum_corrective_sft_smoke.py`
- Raw reports, candidate JSONL files, model state, and Slurm logs remain
  untracked under ignored `post_training/runs/` in the cluster working copy.
- All-valid report SHA-256:
  `c43258c744a6f8fe49176ea10f63432b5ba3301ab01e649effeab8887ed22ec3`
- All-valid trained candidate report SHA-256:
  `4f10c38a84f2c771046d549bbc55b963c4ade7e72a8b8c6d420bff6fd3996a47`
- Pair-observed report SHA-256:
  `f16b43a0cde0ca6e5affda2a41e8fe280fc4995b6246680c7d5abcb84b91fd64`
- Pair-observed trained candidate report SHA-256:
  `d9f7c211086f2bde11ae104cb61b577837b7467283432f75f1ff9f936dcf697b`

# Stage A Enum Candidate-CE Pair-Observed Cayuga Result

Date: 2026-07-05

Purpose: test whether an explicit finite-candidate enum cross-entropy objective
repairs the 5-way `pair_observed_outputs` top-1 failure seen after
pairwise-margin SFT. This is still component-level candidate scoring, not free
generation, `tool_query`, routing-after-loop, DPO, or RLVR.

## Setup

| Setting | Value |
| --- | --- |
| Target format | `full` |
| Pairwise margin weight | 1.0 |
| Required pairwise margin | 0.05 |
| Candidate CE weight | 1.0 |
| Candidate policy | `pair_observed_outputs` |
| Candidate space | 5 |
| Focus pair | `flag` / `invalid_value` |
| Focus repeat | 4 |
| Held-out pairs | 4 |

## Result

| Metric | Base | Trained |
| --- | ---: | ---: |
| Teacher-forced margin wins | 0/4 | 4/4 |
| Mean teacher-forced margin | -0.116510 | 0.183921 |
| Candidate top-1 | 0/4 | 1/4 |
| Candidate accuracy | 0.00 | 0.25 |
| Mean gold rank | 3.50 | 2.50 |
| Mean top-gold margin | 0.133635 | 0.090253 |

By held-out target, the trained 5-way candidate scorer only selects
`reject` / `contradicted` correctly. The `defer`, `verify`, and `flag` targets
remain below top-1, and the trained top candidate collapses to
`reject` / `contradicted` for all four held-out cases.

## Interpretation

This is a negative/partial candidate-selection result. Candidate CE improves
mean gold rank and repairs the teacher-forced margin gate, but it does not solve
finite-candidate top-1 selection. The next `enum_action` step should test
stronger candidate calibration, action/status factorization, or a constrained
candidate gate before moving to `tool_query`, DPO, or RLVR.

## Trace

- Compact JSON summary:
  `post_training/stage_a_enum_candidate_ce_pair_observed_cayuga_summary_2026-07-05.json`
- Runner: `post_training/run_stage_a_enum_corrective_sft_smoke.py`
- Raw reports, candidate JSONL files, model state, and Slurm logs remain
  untracked under ignored `post_training/runs/` in the cluster working copy.
- Raw `report.json` SHA-256:
  `44db99eb90712c1edf065e3fedf38385068f4bc3358a3457faab61a261e7bc53`
- Trained margin report SHA-256:
  `a7cc285fbad634744506bf9591e81c7bed58244f1f5ea36fbe3f0ca337be3e98`
- Trained candidate report SHA-256:
  `d66138d510fd5c44d1438dce8cf8d04e83e6e55521553e88524e4ed299859d83`

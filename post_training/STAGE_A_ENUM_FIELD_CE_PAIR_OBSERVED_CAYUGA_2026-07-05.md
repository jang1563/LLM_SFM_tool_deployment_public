# Stage A Enum Field-CE Pair-Observed Cayuga Result

Date: 2026-07-05

Purpose: test whether factorized `action` / `evidence_status` candidate CE
repairs the field-level failures seen after pair candidate CE. This remains a
supervised component-slice diagnostic, not free generation, `tool_query`, DPO,
RLVR, or a full trajectory result.

## Setup

| Setting | Value |
| --- | --- |
| Target format | `full` |
| Pairwise margin weight | 1.0 |
| Required pairwise margin | 0.05 |
| Candidate CE weight | 1.0 |
| Candidate CE mode | `field` |
| Candidate policy | `pair_observed_outputs` |
| Candidate space | 5 |
| Focus pair | `flag` / `invalid_value` |
| Focus repeat | 4 |
| Held-out pairs | 4 |

## Result

| Metric | Base | Field-CE trained |
| --- | ---: | ---: |
| Teacher-forced margin wins | 0/4 | 4/4 |
| Mean teacher-forced margin | -0.116510 | 0.178547 |
| Candidate top-1 | 0/4 | 1/4 |
| Candidate accuracy | 0.00 | 0.25 |
| Mean gold rank | 3.50 | 2.50 |
| Mean top-gold margin | 0.133635 | 0.092369 |

The trained top candidate remains `reject` / `contradicted` for all four
held-out cases. Field-rank patterns remain `pair_top1` for 1 case and
`both_field_failure` for 3 cases.

## Gate Readout

The adaptive zero-false-trust threshold is 0.042486. At that threshold the gate
trusts 0/4 rows, so the useful zero-false-trust coverage is still 0/4.

## Interpretation

Field CE does not repair the `enum_action` candidate-selection bottleneck. It
preserves the teacher-forced margin repair, but it does not improve beyond the
previous pair-CE candidate top-1 result and still leaves no useful fail-closed
trust coverage. Keep `tool_query`, DPO, and RLVR gated. The next enum repair
should move from loss-shape tweaks toward candidate calibration, constrained
routing, or more explicit evidence-conditioned supervision.

## Trace

- Compact JSON summary:
  `post_training/stage_a_enum_field_ce_pair_observed_cayuga_summary_2026-07-05.json`
- Runner: `post_training/run_stage_a_enum_corrective_sft_smoke.py`
- Gate analyzer: `post_training/analyze_stage_a_enum_candidate_gate.py`
- Raw reports, candidate JSONL files, model state, and Slurm logs remain
  untracked under ignored `post_training/runs/` in the cluster working copy.
- Raw `report.json` SHA-256:
  `af207c2881a4cc15cc573d3fd611dd1fdd0750d856651c0397a0cd062172c04e`
- Trained candidate report SHA-256:
  `6c9f199f9cb2cad301ddaf991147d39fa2b5107bcd56655cbf87e9b46dd0ff15`
- Gate report SHA-256:
  `f1d1ec39cc8f06296ff8f8f8644d788d7d3a87995a2c6740ccc6f020d979bef0`

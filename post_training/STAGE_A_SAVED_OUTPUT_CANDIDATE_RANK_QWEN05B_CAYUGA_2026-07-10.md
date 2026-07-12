# Stage A Saved-Output Candidate-Rank Result

Purpose: test whether the full-target margin repair survives finite-candidate
selection after the same `flag` / `invalid_value` focused SFT run.

## Run

- Run ID:
  `stage_a_saved_output_flag_full_candidate_rank_qwen05b_cayuga_20260710`
- Slurm job: `[omitted]`
- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- Training target format: `full`
- Candidate target format: `full`
- Candidate policy: `train_observed_plus_rejected`
- Candidate space: 5 pairs, consisting of train-observed target pairs plus
  `ground` / `supported`
- Training rows: 16 train-allowed exposures from
  `--focus-chosen-pairs flag/invalid_value --focus-repeat 4 --focus-only`
- Held-out rows: 4 evaluation-only probe pairs
- Held-out rows used for training: `false`

## Result

| Readout | Base | Trained |
| --- | ---: | ---: |
| Teacher-forced full margin wins | 0/4 | 4/4 |
| Mean full margin | -0.081001 | +0.018087 |
| Candidate exact top-1 | 0/4 | 1/4 |
| Mean target rank | 3.5 | 2.5 |
| Top pair bias | `ground` / `supported` in 4/4 | `flag` / `invalid_value` in 4/4 |

Trained candidate ranks:

| Target pair | Top pair | Target rank |
| --- | --- | ---: |
| `reject` / `contradicted` | `flag` / `invalid_value` | 4 |
| `defer` / `insufficient` | `flag` / `invalid_value` | 3 |
| `verify` / `insufficient` | `flag` / `invalid_value` | 2 |
| `flag` / `invalid_value` | `flag` / `invalid_value` | 1 |

Fail-closed score-gap gating does not rescue the candidate readout. The best
default zero-unsafe trained threshold trusts 0/4 rows and reaches 1/4 strict
final correctness through fail-closed default routing only.

## Decision

This is a negative/partial transfer result. The full-target SFT repaired the
teacher-forced margin against `ground` / `supported`, but candidate selection
shifted into `flag` / `invalid_value` over-selection. This is not free
generation, trajectory readiness, or a reason to start DPO/RLVR.

Keep `tool_query`, DPO/RLVR, Hugging Face publication, release tagging, and
broad retraining gated. The next useful step is candidate calibration or
field-wise routing diagnosis, with runtime enforcement still treated as the
deployment baseline.

Raw candidate-score JSONL, full reports, trainable state, and scheduler logs
remain in ignored Cayuga `post_training/runs/` folders and are not part of the
public surface.

# SFT Curriculum Failure Analysis: 2026-06-26

Raw run artifacts are under `post_training/runs/` and ignored by git.
This file records row-level diagnostics for the full curriculum-SFT CV rerun.

## Condition Summary

| condition | accuracy | failures | parse failures | class accuracy |
| --- | --- | --- | --- | --- |
| curriculum_strict | 0.475 | 21 | 0 | defer 3/8, flag 6/8, ground 3/8, reject 4/8, verify 3/8 |
| curriculum_constrained | 0.425 | 23 | 0 | defer 3/8, flag 7/8, ground 2/8, reject 2/8, verify 3/8 |

## Failure Pair Counts

```json
{
  "curriculum_constrained": {
    "defer->verify": 5,
    "flag->reject": 1,
    "ground->flag": 5,
    "ground->reject": 1,
    "reject->flag": 6,
    "verify->defer": 5
  },
  "curriculum_strict": {
    "defer->verify": 5,
    "flag->reject": 2,
    "ground->flag": 4,
    "ground->reject": 1,
    "reject->flag": 4,
    "verify->defer": 5
  }
}
```

## Confusion Matrices

### curriculum_strict

| gold | predictions |
| --- | --- |
| defer | defer 3, verify 5 |
| flag | flag 6, reject 2 |
| ground | flag 4, ground 3, reject 1 |
| reject | flag 4, reject 4 |
| verify | defer 5, verify 3 |

### curriculum_constrained

| gold | predictions |
| --- | --- |
| defer | defer 3, verify 5 |
| flag | flag 7, reject 1 |
| ground | flag 5, ground 2, reject 1 |
| reject | flag 6, reject 2 |
| verify | defer 5, verify 3 |

Gold candidate ranks in constrained scoring:

```json
{
  "defer": {
    "mean_gold_minus_winner_mean_nll": 0.1094,
    "rank_counts": {
      "1": 3,
      "2": 5
    },
    "total": 8
  },
  "flag": {
    "mean_gold_minus_winner_mean_nll": 0.0062,
    "rank_counts": {
      "1": 7,
      "2": 1
    },
    "total": 8
  },
  "ground": {
    "mean_gold_minus_winner_mean_nll": 0.0437,
    "rank_counts": {
      "1": 2,
      "2": 5,
      "4": 1
    },
    "total": 8
  },
  "reject": {
    "mean_gold_minus_winner_mean_nll": 0.1872,
    "rank_counts": {
      "1": 2,
      "2": 1,
      "3": 3,
      "4": 1,
      "9": 1
    },
    "total": 8
  },
  "verify": {
    "mean_gold_minus_winner_mean_nll": 0.1451,
    "rank_counts": {
      "1": 3,
      "2": 5
    },
    "total": 8
  }
}
```

## Persistent Strict-And-Constrained Failures

| packet_id | failure conditions | failures | task note |
| --- | --- | --- | --- |
| ct::defer::134453::3090 | 2 | curriculum_strict:defer->verify; curriculum_constrained:defer->verify | drug has no recorded failures -> insufficient evidence |
| ct::defer::158039::15870 | 2 | curriculum_strict:defer->verify; curriculum_constrained:defer->verify | drug has no recorded failures -> insufficient evidence |
| ct::defer::166731::7134 | 2 | curriculum_strict:defer->verify; curriculum_constrained:defer->verify | drug has no recorded failures -> insufficient evidence |
| ct::defer::26670::54004 | 2 | curriculum_strict:defer->verify; curriculum_constrained:defer->verify | drug has no recorded failures -> insufficient evidence |
| ct::defer::6426::28490 | 2 | curriculum_strict:defer->verify; curriculum_constrained:defer->verify | drug has no recorded failures -> insufficient evidence |
| ct::flag::148::1617 | 2 | curriculum_strict:flag->reject; curriculum_constrained:flag->reject | adapter injects an impossible p-value into the gold efficacy record |
| ct::ground::1716::17720 | 2 | curriculum_strict:ground->flag; curriculum_constrained:ground->flag | efficacy |
| ct::ground::22::300 | 2 | curriculum_strict:ground->reject; curriculum_constrained:ground->reject | efficacy |
| ct::ground::24736::209 | 2 | curriculum_strict:ground->flag; curriculum_constrained:ground->flag | efficacy |
| ct::ground::47205::44830 | 2 | curriculum_strict:ground->flag; curriculum_constrained:ground->flag | efficacy |
| ct::ground::5768::4152 | 2 | curriculum_strict:ground->flag; curriculum_constrained:ground->flag | efficacy |
| ct::reject::1060::12635 | 2 | curriculum_strict:reject->flag; curriculum_constrained:reject->flag | both met and not-met endpoints for this drug x indication -> mixed, reject unqualified 'failed' |
| ct::reject::155::6192 | 2 | curriculum_strict:reject->flag; curriculum_constrained:reject->flag | both met and not-met endpoints for this drug x indication -> mixed, reject unqualified 'failed' |
| ct::reject::22104::5857 | 2 | curriculum_strict:reject->flag; curriculum_constrained:reject->flag | both met and not-met endpoints for this drug x indication -> mixed, reject unqualified 'failed' |
| ct::reject::22::97 | 2 | curriculum_strict:reject->flag; curriculum_constrained:reject->flag | both met and not-met endpoints for this drug x indication -> mixed, reject unqualified 'failed' |
| ct::verify::24969::44875 | 2 | curriculum_strict:verify->defer; curriculum_constrained:verify->defer | related evidence (other indications) but not this one -> verify before asserting |
| ct::verify::42767::3541 | 2 | curriculum_strict:verify->defer; curriculum_constrained:verify->defer | related evidence (other indications) but not this one -> verify before asserting |
| ct::verify::5277::29636 | 2 | curriculum_strict:verify->defer; curriculum_constrained:verify->defer | related evidence (other indications) but not this one -> verify before asserting |
| ct::verify::65190::9649 | 2 | curriculum_strict:verify->defer; curriculum_constrained:verify->defer | related evidence (other indications) but not this one -> verify before asserting |
| ct::verify::69004::7330 | 2 | curriculum_strict:verify->defer; curriculum_constrained:verify->defer | related evidence (other indications) but not this one -> verify before asserting |

## Diagnosis

- `flag` improved relative to the original native CV run, but that came with persistent `ground -> flag` and `reject -> flag` errors.
- `defer` and `verify` remain symmetric boundary failures: five true `defer` rows become `verify`, and five true `verify` rows become `defer` in both strict and constrained scoring.
- Clean efficacy-failure support is still confused with impossible-value evidence: true `ground` rows mostly lose to `flag` candidates.
- Mixed-endpoint `reject` override is not stable. In constrained scoring, six of eight true `reject` rows are predicted as `flag`.
- Candidate-rank diagnostics show many `defer`/`verify` and `ground` misses are rank-2, while `reject` misses can be much deeper; `reject` likely needs the strongest next formulation work.

## Next Action

Targeted curriculum-v2 has now been run and recorded in
`post_training/SFT_CURRICULUM_V2_RUN_RESULTS_2026-06-26.md`.

That result is negative: persistent-failure oversampling lowers aggregate
accuracy and leaves constrained `reject` at 0/8. The next step should be a
boundary-rationale or paired contrast SFT artifact that explicitly teaches why
near-neighbor actions are wrong, not another row-duplication pass.

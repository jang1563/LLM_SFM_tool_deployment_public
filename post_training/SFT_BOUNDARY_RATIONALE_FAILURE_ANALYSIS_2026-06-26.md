# SFT Boundary-Rationale Failure Analysis: 2026-06-26

Raw run artifacts are under `post_training/runs/` and ignored by git.
This file records row-level diagnostics for the full boundary-rationale SFT CV rerun.

## Condition Summary

| condition | accuracy | failures | parse failures | class accuracy |
| --- | --- | --- | --- | --- |
| boundary_strict | 0.5 | 20 | 0 | defer 0/8, flag 3/8, ground 3/8, reject 6/8, verify 8/8 |
| boundary_constrained | 0.5 | 20 | 0 | defer 0/8, flag 3/8, ground 4/8, reject 5/8, verify 8/8 |

## Failure Pair Counts

```json
{
  "boundary_constrained": {
    "defer->verify": 8,
    "flag->ground": 3,
    "flag->reject": 2,
    "ground->flag": 3,
    "ground->reject": 1,
    "reject->flag": 3
  },
  "boundary_strict": {
    "defer->verify": 8,
    "flag->ground": 3,
    "flag->reject": 2,
    "ground->flag": 3,
    "ground->reject": 1,
    "ground->verify": 1,
    "reject->flag": 2
  }
}
```

## Defer-Vs-Verify Diagnostic

```text
defer_failure_count = 8
all_defer_failures_predicted_verify = True
all_defer_observations_empty = True
heldout_defer_prompt_has_boundary_rationale = False
mean_defer_minus_verify_mean_nll = 0.3479
min_defer_minus_verify_mean_nll = 0.1962
max_defer_minus_verify_mean_nll = 0.6068
```

| fold | packet_id | search_failures | other_indication_failures | verify nll | defer nll | defer-verify nll |
| --- | --- | --- | --- | --- | --- | --- |
| fold0 | ct::defer::6426::28490 | 0 | 0 | 0.0043 | 0.6112 | 0.6068 |
| fold0 | ct::defer::50845::834 | 0 | 0 | 0.0367 | 0.2328 | 0.1962 |
| fold1 | ct::defer::158039::15870 | 0 | 0 | 0.0607 | 0.3102 | 0.2495 |
| fold1 | ct::defer::134453::3090 | 0 | 0 | 0.0309 | 0.3162 | 0.2853 |
| fold2 | ct::defer::26670::54004 | 0 | 0 | 0.0130 | 0.4411 | 0.4281 |
| fold2 | ct::defer::166731::7134 | 0 | 0 | 0.0074 | 0.5783 | 0.5709 |
| fold3 | ct::defer::59110::32859 | 0 | 0 | 0.0300 | 0.2619 | 0.2319 |
| fold3 | ct::defer::124533::4281 | 0 | 0 | 0.0328 | 0.2469 | 0.2141 |

## Confusion Matrices

### boundary_strict

| gold | predictions |
| --- | --- |
| defer | verify 8 |
| flag | flag 3, ground 3, reject 2 |
| ground | flag 3, ground 3, reject 1, verify 1 |
| reject | flag 2, reject 6 |
| verify | verify 8 |

### boundary_constrained

| gold | predictions |
| --- | --- |
| defer | verify 8 |
| flag | flag 3, ground 3, reject 2 |
| ground | flag 3, ground 4, reject 1 |
| reject | flag 3, reject 5 |
| verify | verify 8 |

Gold candidate ranks in constrained scoring:

```json
{
  "defer": {
    "mean_gold_minus_winner_mean_nll": 0.3479,
    "rank_counts": {
      "2": 8
    },
    "total": 8
  },
  "flag": {
    "mean_gold_minus_winner_mean_nll": 0.0224,
    "rank_counts": {
      "1": 3,
      "2": 5
    },
    "total": 8
  },
  "ground": {
    "mean_gold_minus_winner_mean_nll": 0.0141,
    "rank_counts": {
      "1": 4,
      "2": 3,
      "3": 1
    },
    "total": 8
  },
  "reject": {
    "mean_gold_minus_winner_mean_nll": 0.0483,
    "rank_counts": {
      "1": 5,
      "3": 3
    },
    "total": 8
  },
  "verify": {
    "mean_gold_minus_winner_mean_nll": 0.0,
    "rank_counts": {
      "1": 8
    },
    "total": 8
  }
}
```

## Persistent Strict-And-Constrained Failures

| packet_id | failure conditions | failures | task note |
| --- | --- | --- | --- |
| ct::defer::124533::4281 | 2 | boundary_strict:defer->verify; boundary_constrained:defer->verify | drug has no recorded failures -> insufficient evidence |
| ct::defer::134453::3090 | 2 | boundary_strict:defer->verify; boundary_constrained:defer->verify | drug has no recorded failures -> insufficient evidence |
| ct::defer::158039::15870 | 2 | boundary_strict:defer->verify; boundary_constrained:defer->verify | drug has no recorded failures -> insufficient evidence |
| ct::defer::166731::7134 | 2 | boundary_strict:defer->verify; boundary_constrained:defer->verify | drug has no recorded failures -> insufficient evidence |
| ct::defer::26670::54004 | 2 | boundary_strict:defer->verify; boundary_constrained:defer->verify | drug has no recorded failures -> insufficient evidence |
| ct::defer::50845::834 | 2 | boundary_strict:defer->verify; boundary_constrained:defer->verify | drug has no recorded failures -> insufficient evidence |
| ct::defer::59110::32859 | 2 | boundary_strict:defer->verify; boundary_constrained:defer->verify | drug has no recorded failures -> insufficient evidence |
| ct::defer::6426::28490 | 2 | boundary_strict:defer->verify; boundary_constrained:defer->verify | drug has no recorded failures -> insufficient evidence |
| ct::flag::148::1617 | 2 | boundary_strict:flag->reject; boundary_constrained:flag->reject | adapter injects an impossible p-value into the gold efficacy record |
| ct::flag::24736::209 | 2 | boundary_strict:flag->ground; boundary_constrained:flag->ground | adapter injects an impossible p-value into the gold efficacy record |
| ct::flag::368::3511 | 2 | boundary_strict:flag->reject; boundary_constrained:flag->reject | adapter injects an impossible p-value into the gold efficacy record |
| ct::flag::383::5105 | 2 | boundary_strict:flag->ground; boundary_constrained:flag->ground | adapter injects an impossible p-value into the gold efficacy record |
| ct::flag::4675::2550 | 2 | boundary_strict:flag->ground; boundary_constrained:flag->ground | adapter injects an impossible p-value into the gold efficacy record |
| ct::ground::1196::14942 | 2 | boundary_strict:ground->flag; boundary_constrained:ground->flag | efficacy |
| ct::ground::22::300 | 2 | boundary_strict:ground->reject; boundary_constrained:ground->reject | efficacy |
| ct::ground::24736::209 | 2 | boundary_strict:ground->flag; boundary_constrained:ground->flag | efficacy |
| ct::ground::47205::44830 | 2 | boundary_strict:ground->flag; boundary_constrained:ground->flag | efficacy |
| ct::reject::1060::12635 | 2 | boundary_strict:reject->flag; boundary_constrained:reject->flag | both met and not-met endpoints for this drug x indication -> mixed, reject unqualified 'failed' |
| ct::reject::22104::5857 | 2 | boundary_strict:reject->flag; boundary_constrained:reject->flag | both met and not-met endpoints for this drug x indication -> mixed, reject unqualified 'failed' |

## Diagnosis

- Boundary-rationale SFT is a modest positive aggregate result, but the remaining failure is sharply structured.
- Every true `defer` held-out row is predicted as `verify` in both strict and constrained scoring.
- Those rows have the empty-evidence observation signature that should define `defer`: `search_failures=[]` and `failures_for_other_indications=0`.
- Candidate scoring shows `defer` is not narrowly losing: mean `defer - verify` NLL is positive and substantial across all eight true-defer rows.
- The held-out prompts do not contain the injected boundary-rationale message, so the next test should determine whether rationale conditioning helps only when present at inference time or whether the target signal itself is too weak.

## Next Action

Follow-up complete: the held-out oracle-rationale ablation is recorded in
`post_training/SFT_BOUNDARY_RATIONALE_HELDOUT_ABLATION_2026-06-27.md`.

It rescues `defer` under oracle rationale prompting, so the next non-oracle
step is to decide whether to add deployable rationale generation or construct
explicit `defer` vs `verify` preference supervision before broader DPO/RLVR.

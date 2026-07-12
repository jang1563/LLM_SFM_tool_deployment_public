# SFT Row-Level Failure Analysis: 2026-06-26

Raw run artifacts are under `post_training/runs/` and ignored by git.
This file records compact row-level diagnostics for the full SFT sweep.

## Condition Summary

| condition | accuracy | failures | parse failures | class accuracy |
| --- | --- | --- | --- | --- |
| native_cv_strict | 0.475 | 21 | 0 | defer 3/8, flag 3/8, ground 6/8, reject 6/8, verify 1/8 |
| native_cv_constrained | 0.4 | 24 | 0 | defer 3/8, flag 3/8, ground 6/8, reject 4/8, verify 0/8 |
| oracle400_strict | 0.45 | 22 | 0 | defer 8/8, flag 0/8, ground 8/8, reject 2/8, verify 0/8 |
| oracle400_constrained | 0.4 | 24 | 0 | defer 8/8, flag 0/8, ground 8/8, reject 0/8, verify 0/8 |

## Confusion Matrices

### native_cv_strict

```json
{
  "defer": {
    "defer": 3,
    "verify": 5
  },
  "flag": {
    "flag": 3,
    "ground": 3,
    "reject": 2
  },
  "ground": {
    "flag": 1,
    "ground": 6,
    "reject": 1
  },
  "reject": {
    "flag": 2,
    "reject": 6
  },
  "verify": {
    "defer": 7,
    "verify": 1
  }
}
```

Violations:

```json
{
  "evidence_status_mismatch": 9,
  "invalid_value_requires_reject_or_flag": 3,
  "missing_required_attribution": 3,
  "terminal_action_mismatch": 17
}
```

### native_cv_constrained

```json
{
  "defer": {
    "defer": 3,
    "verify": 5
  },
  "flag": {
    "flag": 3,
    "ground": 3,
    "reject": 2
  },
  "ground": {
    "flag": 1,
    "ground": 6,
    "reject": 1
  },
  "reject": {
    "flag": 3,
    "ground": 1,
    "reject": 4
  },
  "verify": {
    "defer": 8
  }
}
```

Violations:

```json
{
  "contradicted_claim_requires_reject_or_flag": 1,
  "evidence_status_mismatch": 11,
  "invalid_value_requires_reject_or_flag": 3,
  "missing_required_attribution": 3,
  "terminal_action_mismatch": 19
}
```

Gold candidate ranks in constrained scoring:

```json
{
  "defer": {
    "mean_gold_minus_winner_mean_nll": 0.0826,
    "rank_counts": {
      "1": 3,
      "2": 5
    },
    "total": 8
  },
  "flag": {
    "mean_gold_minus_winner_mean_nll": 0.0417,
    "rank_counts": {
      "1": 3,
      "2": 5
    },
    "total": 8
  },
  "ground": {
    "mean_gold_minus_winner_mean_nll": 0.0269,
    "rank_counts": {
      "1": 6,
      "2": 1,
      "3": 1
    },
    "total": 8
  },
  "reject": {
    "mean_gold_minus_winner_mean_nll": 0.0302,
    "rank_counts": {
      "1": 4,
      "2": 1,
      "3": 3
    },
    "total": 8
  },
  "verify": {
    "mean_gold_minus_winner_mean_nll": 0.0692,
    "rank_counts": {
      "2": 8
    },
    "total": 8
  }
}
```

### oracle400_strict

```json
{
  "defer": {
    "defer": 8
  },
  "flag": {
    "ground": 8
  },
  "ground": {
    "ground": 8
  },
  "reject": {
    "ground": 6,
    "reject": 2
  },
  "verify": {
    "defer": 8
  }
}
```

Violations:

```json
{
  "contradicted_claim_requires_reject_or_flag": 6,
  "evidence_status_mismatch": 14,
  "invalid_value_requires_reject_or_flag": 8,
  "terminal_action_mismatch": 22
}
```

### oracle400_constrained

```json
{
  "defer": {
    "defer": 8
  },
  "flag": {
    "ground": 8
  },
  "ground": {
    "ground": 8
  },
  "reject": {
    "ground": 8
  },
  "verify": {
    "defer": 8
  }
}
```

Violations:

```json
{
  "contradicted_claim_requires_reject_or_flag": 8,
  "evidence_status_mismatch": 16,
  "invalid_value_requires_reject_or_flag": 8,
  "terminal_action_mismatch": 24
}
```

Gold candidate ranks in constrained scoring:

```json
{
  "defer": {
    "mean_gold_minus_winner_mean_nll": 0.0,
    "rank_counts": {
      "1": 8
    },
    "total": 8
  },
  "flag": {
    "mean_gold_minus_winner_mean_nll": 0.2793,
    "rank_counts": {
      "2": 6,
      "4": 2
    },
    "total": 8
  },
  "ground": {
    "mean_gold_minus_winner_mean_nll": 0.0,
    "rank_counts": {
      "1": 8
    },
    "total": 8
  },
  "reject": {
    "mean_gold_minus_winner_mean_nll": 0.1723,
    "rank_counts": {
      "2": 6,
      "4": 1,
      "9": 1
    },
    "total": 8
  },
  "verify": {
    "mean_gold_minus_winner_mean_nll": 0.1586,
    "rank_counts": {
      "2": 8
    },
    "total": 8
  }
}
```

## Recurrent Failures

| packet_id | failure conditions | failures |
| --- | --- | --- |
| ct::flag::13976::643 | 4 | native_cv_strict:flag->ground; native_cv_constrained:flag->ground; oracle400_strict:flag->ground; oracle400_constrained:flag->ground |
| ct::flag::1413::20487 | 4 | native_cv_strict:flag->ground; native_cv_constrained:flag->ground; oracle400_strict:flag->ground; oracle400_constrained:flag->ground |
| ct::flag::148::1617 | 4 | native_cv_strict:flag->reject; native_cv_constrained:flag->reject; oracle400_strict:flag->ground; oracle400_constrained:flag->ground |
| ct::flag::368::3511 | 4 | native_cv_strict:flag->reject; native_cv_constrained:flag->reject; oracle400_strict:flag->ground; oracle400_constrained:flag->ground |
| ct::flag::4675::2550 | 4 | native_cv_strict:flag->ground; native_cv_constrained:flag->ground; oracle400_strict:flag->ground; oracle400_constrained:flag->ground |
| ct::reject::1060::12635 | 4 | native_cv_strict:reject->flag; native_cv_constrained:reject->flag; oracle400_strict:reject->ground; oracle400_constrained:reject->ground |
| ct::reject::22::97 | 4 | native_cv_strict:reject->flag; native_cv_constrained:reject->flag; oracle400_strict:reject->ground; oracle400_constrained:reject->ground |
| ct::verify::143457::15070 | 4 | native_cv_strict:verify->defer; native_cv_constrained:verify->defer; oracle400_strict:verify->defer; oracle400_constrained:verify->defer |
| ct::verify::38487::47651 | 4 | native_cv_strict:verify->defer; native_cv_constrained:verify->defer; oracle400_strict:verify->defer; oracle400_constrained:verify->defer |
| ct::verify::42767::3541 | 4 | native_cv_strict:verify->defer; native_cv_constrained:verify->defer; oracle400_strict:verify->defer; oracle400_constrained:verify->defer |
| ct::verify::48428::49338 | 4 | native_cv_strict:verify->defer; native_cv_constrained:verify->defer; oracle400_strict:verify->defer; oracle400_constrained:verify->defer |
| ct::verify::5277::29636 | 4 | native_cv_strict:verify->defer; native_cv_constrained:verify->defer; oracle400_strict:verify->defer; oracle400_constrained:verify->defer |
| ct::verify::65190::9649 | 4 | native_cv_strict:verify->defer; native_cv_constrained:verify->defer; oracle400_strict:verify->defer; oracle400_constrained:verify->defer |
| ct::verify::69004::7330 | 4 | native_cv_strict:verify->defer; native_cv_constrained:verify->defer; oracle400_strict:verify->defer; oracle400_constrained:verify->defer |
| ct::reject::155::6192 | 3 | native_cv_constrained:reject->flag; oracle400_strict:reject->ground; oracle400_constrained:reject->ground |
| ct::verify::24969::44875 | 3 | native_cv_constrained:verify->defer; oracle400_strict:verify->defer; oracle400_constrained:verify->defer |

## Main Diagnosis

- `verify` is consistently pulled toward `defer`; this is the most stable action-class failure.
- `flag` is often treated as `ground`, especially after oracle-400 warm start.
- Oracle-400 warm start collapses many `reject` examples to `ground`, suggesting the larger artifact over-emphasizes cited positive-looking failure rows without enough contrastive pressure for mixed-endpoint and invalid-value cases.
- Parse stability is solved here; the next bottleneck is action discrimination under similar tool observations.

## Next Action

Build a balanced SFT variant that increases effective pressure on `verify` and `flag`, then rerun the same CV/oracle evaluation harness before DPO/RLVR.

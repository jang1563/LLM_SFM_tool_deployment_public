# SFT Pressure Failure Analysis: 2026-06-26

Raw run artifacts are under `post_training/runs/` and ignored by git.
This file records row-level diagnostics for the pressure-SFT rerun.

## Condition Summary

| condition | accuracy | failures | parse failures | class accuracy |
| --- | --- | --- | --- | --- |
| pressure_strict | 0.4 | 24 | 0 | defer 0/8, flag 4/8, ground 0/8, reject 4/8, verify 8/8 |
| pressure_constrained | 0.45 | 22 | 0 | defer 0/8, flag 6/8, ground 1/8, reject 3/8, verify 8/8 |
| oracle_balanced_strict | 0.2 | 32 | 12 | defer 0/8, flag 0/8, ground 8/8, reject 0/8, verify 0/8 |
| oracle_balanced_constrained | 0.2 | 32 | 0 | defer 0/8, flag 0/8, ground 8/8, reject 0/8, verify 0/8 |

## Confusion Matrices

### pressure_strict

| gold | predictions |
| --- | --- |
| defer | verify 8 |
| flag | flag 4, reject 4 |
| ground | flag 5, reject 3 |
| reject | flag 3, reject 4, verify 1 |
| verify | verify 8 |

### pressure_constrained

| gold | predictions |
| --- | --- |
| defer | verify 8 |
| flag | flag 6, reject 2 |
| ground | flag 6, ground 1, reject 1 |
| reject | flag 4, reject 3, verify 1 |
| verify | verify 8 |

Gold candidate ranks in constrained scoring:

```json
{
  "defer": {
    "mean_gold_minus_winner_mean_nll": 0.6544,
    "rank_counts": {
      "2": 7,
      "3": 1
    },
    "total": 8
  },
  "flag": {
    "mean_gold_minus_winner_mean_nll": 0.0111,
    "rank_counts": {
      "1": 6,
      "2": 2
    },
    "total": 8
  },
  "ground": {
    "mean_gold_minus_winner_mean_nll": 0.0593,
    "rank_counts": {
      "1": 1,
      "2": 6,
      "3": 1
    },
    "total": 8
  },
  "reject": {
    "mean_gold_minus_winner_mean_nll": 0.0622,
    "rank_counts": {
      "1": 3,
      "2": 1,
      "3": 4
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

### oracle_balanced_strict

| gold | predictions |
| --- | --- |
| defer | None 7, ground 1 |
| flag | ground 8 |
| ground | ground 8 |
| reject | ground 8 |
| verify | None 5, ground 3 |

### oracle_balanced_constrained

| gold | predictions |
| --- | --- |
| defer | reject 8 |
| flag | ground 8 |
| ground | ground 8 |
| reject | ground 8 |
| verify | reject 8 |

Gold candidate ranks in constrained scoring:

```json
{
  "defer": {
    "mean_gold_minus_winner_mean_nll": 0.8436,
    "rank_counts": {
      "3": 8
    },
    "total": 8
  },
  "flag": {
    "mean_gold_minus_winner_mean_nll": 1.4437,
    "rank_counts": {
      "2": 6,
      "3": 2
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
    "mean_gold_minus_winner_mean_nll": 4.2179,
    "rank_counts": {
      "3": 5,
      "5": 2,
      "9": 1
    },
    "total": 8
  },
  "verify": {
    "mean_gold_minus_winner_mean_nll": 0.3822,
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
| ct::defer::124533::4281 | 4 | pressure_strict:defer->verify; pressure_constrained:defer->verify; oracle_balanced_strict:defer->None; oracle_balanced_constrained:defer->reject |
| ct::defer::134453::3090 | 4 | pressure_strict:defer->verify; pressure_constrained:defer->verify; oracle_balanced_strict:defer->None; oracle_balanced_constrained:defer->reject |
| ct::defer::158039::15870 | 4 | pressure_strict:defer->verify; pressure_constrained:defer->verify; oracle_balanced_strict:defer->None; oracle_balanced_constrained:defer->reject |
| ct::defer::166731::7134 | 4 | pressure_strict:defer->verify; pressure_constrained:defer->verify; oracle_balanced_strict:defer->None; oracle_balanced_constrained:defer->reject |
| ct::defer::26670::54004 | 4 | pressure_strict:defer->verify; pressure_constrained:defer->verify; oracle_balanced_strict:defer->None; oracle_balanced_constrained:defer->reject |
| ct::defer::50845::834 | 4 | pressure_strict:defer->verify; pressure_constrained:defer->verify; oracle_balanced_strict:defer->None; oracle_balanced_constrained:defer->reject |
| ct::defer::59110::32859 | 4 | pressure_strict:defer->verify; pressure_constrained:defer->verify; oracle_balanced_strict:defer->None; oracle_balanced_constrained:defer->reject |
| ct::defer::6426::28490 | 4 | pressure_strict:defer->verify; pressure_constrained:defer->verify; oracle_balanced_strict:defer->ground; oracle_balanced_constrained:defer->reject |
| ct::flag::148::1617 | 4 | pressure_strict:flag->reject; pressure_constrained:flag->reject; oracle_balanced_strict:flag->ground; oracle_balanced_constrained:flag->ground |
| ct::flag::368::3511 | 4 | pressure_strict:flag->reject; pressure_constrained:flag->reject; oracle_balanced_strict:flag->ground; oracle_balanced_constrained:flag->ground |
| ct::reject::1060::12635 | 4 | pressure_strict:reject->flag; pressure_constrained:reject->flag; oracle_balanced_strict:reject->ground; oracle_balanced_constrained:reject->ground |
| ct::reject::22104::5857 | 4 | pressure_strict:reject->flag; pressure_constrained:reject->flag; oracle_balanced_strict:reject->ground; oracle_balanced_constrained:reject->ground |
| ct::reject::22::97 | 4 | pressure_strict:reject->flag; pressure_constrained:reject->flag; oracle_balanced_strict:reject->ground; oracle_balanced_constrained:reject->ground |
| ct::reject::41090::27993 | 4 | pressure_strict:reject->verify; pressure_constrained:reject->verify; oracle_balanced_strict:reject->ground; oracle_balanced_constrained:reject->ground |
| ct::flag::383::5105 | 3 | pressure_strict:flag->reject; oracle_balanced_strict:flag->ground; oracle_balanced_constrained:flag->ground |
| ct::flag::4675::2550 | 3 | pressure_strict:flag->reject; oracle_balanced_strict:flag->ground; oracle_balanced_constrained:flag->ground |
| ct::reject::155::6192 | 3 | pressure_constrained:reject->flag; oracle_balanced_strict:reject->ground; oracle_balanced_constrained:reject->ground |

## Diagnosis

- Native pressure fixed the previous `verify -> defer` failure, but it over-rotated: all true `defer` rows now become `verify`.
- Native pressure improved `flag` under constrained scoring, but `ground` became fragile and is often treated as `flag` or `reject`.
- `reject` is still not stable; pressure CV splits it across `reject`, `flag`, and `verify`.
- Balanced-oracle warm start is not useful in this form. Strict generation has 12 parse failures and both strict/constrained scoring collapse toward `ground` or `reject` priors rather than evidence-sensitive actions.
- The next SFT step should not be global class balancing. It should be contrastive/curriculum SFT organized around near-neighbor action confusions.

## Curriculum Prescription

1. Keep the native balanced CV held-out folds unchanged.
2. Build train folds with small contrastive packs instead of broad class oversampling.
3. Pair `ground` vs `flag` examples that share cited-NCT structure but differ by impossible-value evidence.
4. Pair `verify` vs `defer` examples where the target indication lacks a valid efficacy failure, separating other-indication failures from no-failure cases.
5. Pair `reject` against `ground`/`flag` examples so mixed-endpoint evidence overrides single-row positive-looking support.
6. Keep `defer` and `ground` support in the curriculum; the pressure run showed that fixing `verify`/`flag` by oversampling alone can erase them.

## Next Action

Create a curriculum SFT artifact that tags contrast family and keeps per-fold held-out splits fixed. Then rerun the same CV harness before DPO/RLVR.

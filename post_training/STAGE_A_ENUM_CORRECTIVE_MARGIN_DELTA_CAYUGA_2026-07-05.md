# Stage A Enum Corrective Margin-Delta Diagnostic

Date: 2026-07-05

Purpose: compare base-model held-out margins to post-SFT held-out margins for
the enum corrective contrast pairs. This answers whether the partial 2/4
corrective result came from useful movement, train overfit, or no movement. It
is still an SFT/margin diagnostic, not DPO/RLVR.

## Result

| Metric | Value |
| --- | ---: |
| Base held-out margin wins | 0/4 |
| Trained held-out margin wins | 2/4 |
| Newly won held-out pairs | 2/4 |
| Newly lost held-out pairs | 0/4 |
| Mean base margin | -0.116510 |
| Mean trained margin | -0.020685 |
| Mean margin delta | 0.095825 |
| Minimum margin delta | 0.047804 |

## Held-Out Delta Readout

Margin is `chosen_score - ground/supported_score`.

| Case family | Chosen pair | Base margin | Trained margin | Delta | Outcome |
| --- | --- | ---: | ---: | ---: | --- |
| contradicted_or_mixed_endpoint_claim | `reject` / `contradicted` | -0.034748 | 0.014703 | 0.049451 | newly_won |
| insufficient_evidence | `defer` / `insufficient` | -0.257193 | -0.052623 | 0.204570 | remained_lost |
| related_evidence_requires_verification | `verify` / `insufficient` | -0.020147 | 0.061328 | 0.081475 | newly_won |
| invalid_value_attribution_failure | `flag` / `invalid_value` | -0.153953 | -0.106149 | 0.047804 | remained_lost |

## Train-Pair Check

The trained model wins 9/16 train-pair margins. The per-family train readout is
uneven:

| Chosen pair | Train wins | Mean train margin |
| --- | ---: | ---: |
| `reject` / `contradicted` | 4/4 | 0.059480 |
| `verify` / `insufficient` | 4/4 | 0.064089 |
| `defer` / `insufficient` | 1/4 | -0.049959 |
| `flag` / `invalid_value` | 0/4 | -0.100009 |

## Interpretation

The corrective SFT is moving in the right direction: every held-out family has a
positive margin delta and no held-out pair regresses. But the run is still not a
stable enum repair. The invalid-value family fails even on train-pair margins,
and defer/insufficient remains weak. The next step should be targeted enum
diagnosis or corrective rows for those two families, not DPO/RLVR or broad
progression to `tool_query`.

## Trace

- Compact JSON summary: `post_training/stage_a_enum_corrective_margin_delta_cayuga_summary_2026-07-05.json`
- Runner: `post_training/run_stage_a_enum_corrective_sft_smoke.py`
- Raw reports, margin JSONL files, model state, and Slurm logs remain untracked
  under ignored `post_training/runs/` in the cluster working copy.
- Report SHA-256:
  `da0cc6d75770c93c8ec71677bd7efa143875403af6518fcdbbeb31e60bac67b7`
- Margin delta report SHA-256:
  `f62aa9969a43391ff68596e094189aba0fb6431975afc1009aec6bd2660b7a41`
- Base margin report SHA-256:
  `7ca0fdb74e1848211f2f0f8d208bb37333c071884656c192b5ff0ab69254c4d4`
- Train margin report SHA-256:
  `394ba37a84991fc73eb331cd150ed134519dda97f09f74c6ba188ad75a7a72da`
- Held-out trained margin report SHA-256:
  `45f48f7e6168498f535f477acdbaec93bf0896ee20f9a7fcf7d95ffd5d6640ae`

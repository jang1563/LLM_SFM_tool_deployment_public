# LLM-SFM Tool Deployment

[![Public QA](https://github.com/jang1563/LLM_SFM_tool_deployment_public/actions/workflows/public-qa.yml/badge.svg)](https://github.com/jang1563/LLM_SFM_tool_deployment_public/actions/workflows/public-qa.yml)

Deployment and post-training experiments for biology tool-use agents.

This repository asks a practical question: when an LLM is the reasoning layer
above scientific databases and specialist models, which parts should be solved
with tools, deterministic validators, retrieval, SFT, preference optimization,
or RL from verifiable rewards? The test bed is biomedical negative evidence:
can an agent decide when a drug has genuinely failed for an indication, when it
should verify, and when it should defer instead of hallucinating certainty?

## Why This Matters

- Scientific agents fail quietly when they over-trust weak evidence, cite the
  wrong record, or treat "not found" as "not failed."
- Reliable biology deployment needs binding tool-use and evidence constraints,
  not only better-looking explanations.
- The repo turns that idea into runnable evaluators, post-training data, and an
  entity-resolution stack that separates retrieval, reranking, and abstention.

## What Is Here

| Area | What it does | Why it matters |
|---|---|---|
| `llm_sfm_tool_deployment/` | Generic trajectory evaluator for tool-use decisions. | Scores whether the model used the right action, evidence, source, and deferral behavior. |
| `negbiodb_ct/` | NegBioDB-CT task builder and tool-use runners. | Creates balanced drug/condition packets for ground, reject, verify, defer, and flag decisions. |
| `negbiodb_ct/stage_a_mini_manifest.jsonl` | Public-safe Stage A benchmark manifest. | Separates model-visible tasks from hidden labels for trajectory evaluation. |
| `post_training/` | SFT, preference, process-supervision, guardrail, split, and validation artifacts. | Converts clean tool-use trajectories into training/evaluation data and checks schema integrity before training. |
| `a2_freetext/` | Free-text drug/disease resolver plus MONDO band reranker. | Handles the deployment bottleneck: messy names must resolve before retrieval can work. |
| `research/2026-06-25_posttrain_tool_use_landscape/` | Source notes and synthesis for post-training/tool-use methods. | Keeps the literature/context scan separate from measured project artifacts. |

## If You Are Reviewing This Repo

| Time | Start here | Goal |
|---|---|---|
| 5 minutes | [REVIEWER_GUIDE.md](REVIEWER_GUIDE.md) | Understand the claim, runnable path, and limitations. |
| 15 minutes | [BENCHMARK_CARD.md](BENCHMARK_CARD.md) and [REPRODUCIBILITY.md](REPRODUCIBILITY.md) | Check the evaluator gates and public-safe validation path. |
| 30+ minutes | `tests/`, `negbiodb_ct/stage_a_manifest.py`, `post_training/validate_post_training_data.py` | Audit hidden-label isolation, data exports, and failure modes. |

## Key Results

| Result | Measured outcome | Interpretation |
|---|---:|---|
| Stage A benchmark substrate | 25 manifest cases; oracle 25/25 pass; self-answer, wrong-tool, partial-query baselines 0/25 pass | The evaluator catches shortcut trajectories before any live API or model-training spend. |
| Stage A post-training artifacts | 25 SFT rows, 150 preference pairs, 25 process rows; train/held-out split has 0 source overlap | The benchmark now has validated trajectory data for local SFT/preference smoke tests. |
| Stage A prediction-output scorer | Held-out oracle adapter smoke 5/5 pass; missing/extra prediction rows fail closed in tests | Saved API, cluster, or prompt-only outputs can be scored offline with the same evaluator. |
| Stage A saved-prediction producer | Self-answer artifact smoke writes 5 rows and scores 0/5 | Prompt-only/API/cluster outputs now have an artifact-first entrypoint before interpretation. |
| Stage A v3 tool-trace prompt contract | Requires an ordered four-tool JSON trace with `drug_id` and `condition_id` arguments while keeping hidden labels/source IDs out of the prompt | Next Cayuga saved-output smoke can isolate tool/query compliance from evidence-label supervision. |
| Stage A v3 Cayuga saved-output result | Qwen2.5-0.5B writes 5 rows but scores 0/5, mean 0.000; all rows fail on invalid `evidence_status: verified` | The immediate blocker is canonical enum/schema compliance before evidence routing or tool-query learning. |
| Stage A v4 canonical JSON Cayuga result | Qwen2.5-0.5B still scores 0/5; invalid `verified` status disappears, but 4/5 rows miss the top-level action and 1/5 is not valid JSON | Prompt-only repair is not enough; next repair should use constrained decoding or component-level enum/action target formatting. |
| Stage A saved-prediction candidate readout | No-model dry-run scores 3/5, mean 0.943 with only 2 missing-attribution failures | Constrained action/status/tool/query construction is now reproducible; the next Cayuga run can test finite-candidate model selection without free-form JSON failure. |
| Stage A candidate-readout Cayuga result | Qwen2.5-0.5B finite-candidate scoring removes parse/tool/query failures but still scores 0/5, mean 0.657; top pair is `ground` / `supported` in 5/5 | The current model bottleneck is action/status candidate-selection collapse, not JSON envelope validity or query shape. |
| Stage A saved-candidate gate result | Qwen2.5-0.5B score-gap gating trusts 1/5 supported row with 0 unsafe trust under both train-observed and all-valid candidate policies; fail-closed strict final correctness is 2/5 | Candidate scores can support a narrow fail-closed boundary, but they do not beat deterministic runtime enforcement or reopen `tool_query`, DPO/RLVR, HF, or release tagging. |
| Stage A saved-output initial next decision | Compact readiness/gate artifacts selected `targeted_action_status_calibration_probe`; minimum next-gate target was 0 unsafe trust and 4/5 fail-closed strict final | The calibration probe was derived from public-safe checkpoints, not broad retraining or publication pressure. |
| Stage A saved-output calibration probe | 20 target-vs-`ground`/`supported` pairs: 16 train-allowed, 4 held-out evaluation-only, 0 train/held-out source overlap | Converts the next-decision checkpoint into a split-safe action/status calibration substrate without reopening `tool_query`, DPO/RLVR, HF publication, or release tagging. |
| Stage A saved-output calibration readout path | Dry-run scores all 20 probe pairs and evaluates a train-selected fail-closed gate on 4 held-out probe rows without model load | Cayuga/Expanse can now test whether model score gaps prefer target action/status outputs over `ground` / `supported` collapse before any escalation. |
| Stage A saved-output calibration readout result | Cayuga Qwen2.5-0.5B scores `ground` / `supported` above the target in 20/20 probe rows; train and held-out exact top-1 are 0 | The saved-output bottleneck is still action/status collapse; keep `tool_query`, DPO/RLVR, HF publication, release tagging, and broad retraining gated. |
| Stage A saved-output margin SFT path | Dry-run validates 16 train-only target-vs-collapse pairs, 4 held-out evaluation-only pairs, pairwise margin objective, and Cayuga/Expanse templates | Next cluster run can test whether targeted SFT moves target action/status outputs above `ground` / `supported`; held-out rows stay excluded from training. |
| Stage A saved-output margin SFT result | Cayuga Qwen2.5-0.5B moves held-out target-vs-collapse margins from 0/4 to 1/4 wins; mean margin improves -0.081 -> -0.046 | Targeted SFT gives partial movement but not repair; keep runtime enforcement and escalation gates in place. |
| Stage A saved-output focused margin SFT result | Cayuga Qwen2.5-0.5B with non-verify family oversampling moves held-out target-vs-collapse margins from 0/4 to 3/4 wins; mean margin improves -0.081 -> -0.003 | Corrective SFT is stronger but still not a repair: `flag` / `invalid_value` remains below `ground` / `supported`, so escalation stays gated. |
| Stage A saved-output target-format diagnostic path | Margin SFT now supports `full`, `action_status_only`, `action_only`, and `status_only` target projections | Next Cayuga run can isolate whether the unresolved `flag` / `invalid_value` failure is action-label learning, status-label learning, or full-JSON/citation coupling. |
| Stage A saved-output target-format diagnostic result | `flag` held-out margin flips with `action_only` (-0.812 -> +0.842) and `action_status_only` (-0.283 -> +0.437); `status_only` is already positive | The remaining negative full-JSON `flag` / `invalid_value` result is target-format/citation/tool-call coupling, not an isolated label-token failure. |
| Stage A saved-output same-model target-format scoring path | Full-target training can now score `full`, `action_only`, `action_status_only`, and `status_only` in one Cayuga run | Next diagnostic tests whether a full-trained model contains the correct action/status preferences even when the full JSON target still loses. |
| Stage A saved-output same-model target-format result | Full-target `flag`-focused SFT moves held-out full JSON wins 0/4 -> 4/4 and `flag` / `invalid_value` margin -0.175 -> +0.026; action/status projections are also positive | First teacher-forced full-target repair signal; finite-candidate ranking, free generation, and full-trajectory readiness remain gated. |
| Stage A saved-output candidate-rank path | Margin SFT can now score base/trained held-out finite candidates with the train-observed target pairs plus `ground` / `supported` collapse | Next Cayuga run can test whether the teacher-forced full-target repair survives 5-way candidate selection before any `tool_query` or optimizer escalation. |
| Stage A saved-output candidate-rank result | Teacher-forced full margin stays 4/4, but 5-way candidate top-1 is only 1/4; top pair shifts from `ground` / `supported` in 4/4 base rows to `flag` / `invalid_value` in 4/4 trained rows | Margin repair does not transfer to candidate selection; keep runtime enforcement, `tool_query`, DPO/RLVR, HF, and release tagging gated. |
| Stage A saved-output candidate field diagnostic | Trained field ranks show `both_field_failure` in 3/4 non-flag rows and `pair_top1` only for the `flag` / `invalid_value` row | The failure is over-selection of the combined `flag` / `invalid_value` pair, not one isolated action or status field. |
| Stage A saved-output candidate calibration | Train-derived pair-mean centering improves held-out candidate top-1 from 1/4 to 2/4, but the train-selected zero-unsafe score-gap gate trusts 0/4 and reaches only 1/4 strict final | Score calibration helps diagnose the prior but is not a trust-ready deployment gate. |
| Stage A saved-output candidate arbitration | Raw candidate top-1 is 1/4, calibrated top-1 is 2/4, score-gap gating is 1/4, while model-visible evidence and hybrid evidence-then-gate policies are 4/4 | Runtime evidence arbitration is now the saved-output baseline any model-heavy checkpoint must meet or beat. |
| Stage A saved-output next-decision update | The targeted calibration probe is closed out; next model-heavy Cayuga output must meet or beat runtime evidence/hybrid arbitration at 4/4 with 0 unsafe candidate trust | `tool_query`, DPO/RLVR, HF publication, release tagging, and broad retraining remain gated. |
| Stage A saved-output meet-or-beat gate | Current raw, calibrated, and score-gap candidate policies all fail the reusable gate; runtime evidence/hybrid policies define the 4/4 baseline | Future compact Cayuga outputs can be converted with `build_stage_a_saved_output_policy_summary.py` and judged through `--model-policy-summary` without reading raw predictions, candidate-score JSONL, scheduler logs, or model state. |
| Stage A saved-output intake contract | Current compact bundle passes hash, criteria, and public-safe flag checks before accepting future Cayuga policy summaries | The next model-heavy result must enter through the compact policy adapter and meet-or-beat gate; raw artifacts stay outside the public surface. |
| Stage A cluster HF baseline | Cayuga Qwen2.5-1.5B held-out run writes 5 rows and scores 0/5, mean 0.114 | First real cluster model artifact shows strict schema/tool gates catching unsupported chat outputs before post-training claims. |
| Stage A strict prompt contract | Cayuga Qwen2.5-1.5B strict-contract run removes parse errors and improves mean score 0.114 -> 0.372, still 0/5 pass | Prompt contracts fix formatting, but ordered tool use and evidence/action routing remain the scientific training target. |
| Stage A strict-contract training targets | 25 SFT rows, 50 observed-collapse preference pairs, 25 process rows; train/held-out split has 0 source overlap | The next SFT/preference smoke can train against the same JSON contract used by Cayuga/API prediction artifacts. |
| Stage A strict-contract SFT smoke path | Dry-run validates 20 train and 5 held-out strict-contract rows without model load | Cayuga/Expanse can now run the tiny SFT follow-up without using laptop compute. |
| Stage A strict-contract SFT smoke result | Cayuga 0.5B and 1.5B tiny SFT both score 0/5; 1.5B mean stays 0.372 | Lower train loss is not enough; tool-loop and routing gates remain the bottleneck. |
| Stage A strict component diagnostics | enum-constrained oracle 5/5; route-only no-tools 0/5 mean 0.714; ordered tool names only 0/5 mean 0.857; full tool loop with collapsed routing 1/5 | Do not escalate to DPO/RLVR until enum decoding, structured query arguments, and evidence/action routing are separately measurable. |
| Stage A strict component targets | 75 slice targets: enum/action, tool-query, routing-after-loop; 60 train and 15 held-out rows with 0 source overlap | The next Cayuga/Expanse experiment can measure each failure source before any DPO/RLVR escalation. |
| Stage A strict component SFT smoke path | Dry-run validates 20 train and 5 held-out rows per slice without model load | Cayuga/Expanse can now measure enum/action, tool-query, and routing-after-loop slices before DPO/RLVR escalation. |
| Stage A `enum_action` component SFT result | Cayuga Qwen2.5-0.5B tiny SFT scores 0/5, mean 0.250; all held-out outputs use invalid `evidence_status: valid` plus an extra `tool` key | Fix constrained enum output or target format before DPO/RLVR or broad retraining. |
| Stage A `enum_action` candidate repair | Candidate scoring improves 0/5 mean 0.250 -> 1/5 mean 0.800; target-key and enum-validity accuracy reach 1.0 | Schema/enum drift is fixed, but enum-pair selection is still biased toward `ground` / `supported`. |
| Stage A enum full-rank diagnostic | Full 30-candidate Cayuga rerun keeps 1/5, mean 0.800; gold ranks are 1, 5, 13, 4, and 24 | The failure is not just calibration: insufficient and invalid-value cases need stronger enum supervision or a narrower valid-pair target. |
| Stage A enum field-rank diagnostic | Full-rank reanalysis gives action top-1 2/5 and evidence-status top-1 1/5; invalid-value has pair rank 24, action rank 6, status rank 2 | The invalid-value bottleneck is weak `flag` action representation, not only missing `invalid_value` status exposure. |
| Stage A observed-pair counterfactual | Restricting to train-observed valid pairs still gives 1/5; top action is `ground` / `supported` in 5/5 | The 30-way candidate space is not the main bottleneck; add evidence-conditioned enum supervision before `tool_query`. |
| Stage A enum corrective pairs | 20 contrast pairs: 16 train, 4 held-out, each rejecting `ground` / `supported` collapse | Provides a component-specific corrective substrate without starting DPO/RLVR or broad retraining. |
| Stage A enum action-contrast pairs | 20 same-status/wrong-action contrast pairs: 16 train, 4 held-out; invalid-value rejects `ground` / `invalid_value` | The next Cayuga smoke can test `flag` action learning without confounding it with status-label selection. |
| Stage A enum action-contrast result | Cayuga action-contrast SFT improves base held-out margins from 1/4 to 2/4 wins and -0.079 -> -0.001 mean margin | Useful action-field movement, but `flag` / `invalid_value` and `defer` / `insufficient` still lose to `ground`; keep `tool_query`, DPO, and RLVR gated. |
| Stage A enum action-only result | Removing `evidence_status` gives 1/4 held-out wins, mean margin -0.033, and 0/4 train wins for `flag`, `defer`, and `reject` families | The bottleneck is action-label evidence, not just action/status JSON coupling; add targeted action rows before changing component slices. |
| Stage A enum pairwise-margin result | Supervised action-only margin SFT reaches 4/4 held-out wins and mean margin 0.262 | This repairs the action-only chosen-over-`ground` slice; next gate is full action+status enum scoring before `tool_query` or DPO/RLVR. |
| Stage A enum full pairwise-margin result | Full action+status margin SFT reaches 4/4 held-out wins on same-status action contrasts and 4/4 on `ground` / `supported` corrective contrasts | Positive full-target component result; still requires finite-candidate/free-generation and trajectory checks before DPO/RLVR. |
| Stage A enum candidate readout | Pairwise-margin SFT keeps 30-way and 5-way finite-candidate top-1 at 0/4, while improving mean gold rank 15.0 -> 7.0 and 3.5 -> 2.75 | Teacher-forced margin repair does not solve candidate selection; keep `tool_query`, DPO, and RLVR gated. |
| Stage A enum candidate-CE result | 5-way candidate CE keeps margin wins at 4/4 and improves candidate top-1 from 0/4 to 1/4, mean gold rank 3.5 -> 2.5 | Useful partial movement, but not enough to leave `enum_action`; next test should target calibration or a constrained candidate gate. |
| Stage A enum candidate gate diagnostic | Score-gap gating over the candidate-CE run has zero useful zero-false-trust coverage: threshold 0.050027 trusts 0/4 rows | Runtime should fail closed rather than trust enum top-1; `tool_query`, DPO, and RLVR remain gated. |
| Stage A enum field-CE result | Factorized action/status CE keeps margins at 4/4 and candidate top-1 at 1/4; useful zero-false-trust coverage remains 0/4 | Loss-shape tweaks are not enough; next enum repair should change candidate calibration/routing or add evidence-conditioned supervision. |
| Stage A component visibility audit | Hidden-label leaks are 0/75, but 50 evidence-routing rows lack model-visible evidence or tool results | The current enum/routing substrate is underconditioned; expose evidence-conditioned state before `tool_query`, DPO, RLVR, or HF publication. |
| Stage A evidence-conditioned component targets | 75 slice targets; 50 evidence-conditioned enum/routing rows; visibility audit closes underdetermined routing from 50/75 to 0/75 with hidden-label leaks 0/75 | The next Cayuga smoke can measure enum/routing with model-visible public-safe evidence state before `tool_query`, DPO, RLVR, or HF publication. |
| Stage A evidence-conditioned `enum_action` result | Cayuga 5-way observed-pair component SFT scores 1/5, mean 0.800; exact-match 0.2; top prediction remains `ground` / `supported` in 5/5 | Evidence visibility fixes the substrate but not enum top-1 selection; run evidence-conditioned `routing_after_loop` next and keep `tool_query`, DPO/RLVR, and HF gated. |
| Stage A evidence-conditioned `routing_after_loop` result | Cayuga free-form component SFT scores 0/5, mean 0.200; target-key and enum-validity accuracy are 0.0; 1 parse error | Visible tool results are not enough for free-form routing; next repair should constrain routing output or split action/status and citation selection. |
| Stage A constrained routing readout | `routing_observed_pair_score` dry-run validates 5 train-observed routing pairs and prompt-visible citation extraction | Next Cayuga routing run can separate schema/enum/citation readout from free-form JSON generation before `tool_query`, DPO/RLVR, or HF. |
| Stage A constrained routing result | Cayuga constrained routing scores 2/5, mean 0.850; schema/enum gates pass; remaining failures are action/status routing misses | This is useful component progress, but insufficient, verify, and invalid-value routing remain unresolved; keep `tool_query`, DPO/RLVR, and HF gated. |
| Stage A routing action/status contrast pairs | 15 evidence-conditioned routing contrast pairs: 12 train, 3 held-out; rejected targets mirror the constrained-routing failures | Provides a targeted repair substrate for insufficient, verify, and invalid-value routing without starting DPO/RLVR or broad retraining. |
| Stage A routing contrast SFT smoke path | Dry-run validates 12 train and 3 held-out routing contrast pairs, optional pairwise margin objective, and Cayuga/Expanse submit templates | Next Cayuga run can test whether targeted SFT/margin pressure repairs action/status routing before `tool_query`, DPO/RLVR, or HF. |
| Stage A routing contrast SFT result | Cayuga routing margin diagnostic improves held-out wins from 0/3 to 3/3 and mean margin -0.117 -> 0.115 | Positive teacher-forced action/status repair; finite-candidate/free-form routing and trajectory gates still block `tool_query`, DPO/RLVR, and HF publication. |
| Stage A routing candidate-rank result | Cayuga finite-candidate routing improves exact top-1 from 0/3 to 2/3 and mean gold rank 3.0 -> 1.33 | Partial transfer beyond teacher-forced margins; `defer` / `insufficient` still loses to `verify` / `insufficient`, so `tool_query`, DPO/RLVR, and HF stay gated. |
| Stage A routing defer-vs-verify boundary pairs | 10 contrast pairs: 8 train, 2 held-out; insufficient and verification-needed families are symmetric | Next Cayuga smoke can target the remaining `defer` / `insufficient` vs `verify` / `insufficient` confusion without broad retraining. |
| Stage A routing defer-vs-verify result | Cayuga targeted smoke stays 1/2 exact top-1; `defer` / `insufficient` margin improves but remains below `verify` / `insufficient` | Negative/partial result; next work should diagnose calibration or fail-closed boundary routing before `tool_query`, DPO/RLVR, or HF. |
| Stage A routing fail-closed gate diagnostic | On the 2-case defer-vs-verify held-out slice, threshold 0.025 trusts 1 correct `verify` row and fails closed on 1 `defer` row, with 0 unsafe trusted rows | Promising runtime-enforcement diagnostic only; not deployment calibration, and not enough to reopen `tool_query`, DPO/RLVR, or HF. |
| Stage A routing evidence-boundary gate | No-model runtime gate using only prompt-visible tool-result fields reaches 10/10 overall and 2/2 held-out on defer-vs-verify pairs | Runtime enforcement is the baseline to beat; this supports verifier/gate design, not more model training yet. |
| Stage A routing evidence gate | No-model runtime gate over all 25 `routing_after_loop` evidence-conditioned rows reaches 25/25 overall and 5/5 held-out | Runtime enforcement is now the broader Stage A routing baseline to beat before `tool_query`, DPO/RLVR, HF publication, or release tagging. |
| Stage A routing gate baseline comparison | Runtime gate and oracle are 25/25; `ground/supported` collapse is 5/25 with 20 unsafe overrides; citationless routing is 15/25 with 10 citation mismatches | Future model outputs must beat collapse and citationless baselines and compete with the runtime gate before escalation. |
| Stage A routing model-readiness gate | Best all-family Cayuga routing readout is 2/5, below citationless routing at 3/5 and runtime gate at 5/5 | Keep `tool_query`, DPO/RLVR, HF publication, and release tagging gated; runtime enforcement remains required. |
| Stage A full-trajectory arbitration | Runtime gate and hybrid runtime-over-collapse score 25/25 full trajectories; collapse is 5/25 with 20 unsafe overrides; citationless routing is 15/25 with attribution failures | Runtime arbitration now runs through the canonical trajectory evaluator, not only component slices. |
| Stage A saved-prediction readiness | Existing real Cayuga saved-output summaries remain 0/5 held-out; the best fail-closed candidate gate reaches 2/5 strict final with 0 unsafe trust, below citationless 3/5 and runtime 5/5 | Saved model outputs and their candidate gates must improve before `tool_query`, DPO/RLVR, HF publication, or release tagging. |
| Stage A evidence candidate-routing full smoke | Qwen2.5-0.5B SFT selects `verify` / `insufficient` for all rows: train 4/20, held-out 1/5, bridge-focus 1/4 | The model does not beat the 1/5 static prior or the 5/5 runtime evidence gate; freeze this diagnostic slice and build a new sealed evaluation extension before further model claims. |
| Stage A sealed evaluation commitment | 25 private rows, 5 per action family; source-task, split-group, and normalized-claim overlap are all 0 against declared public exclusions | Row-level labels remain private. Complete `tool_query`, freeze the policy, then evaluate this extension once. |
| Stage A routing gate arbitration | Raw candidate top-1 is 1/2; score-gap fail-closed, evidence-boundary override, and hybrid policies are 2/2 on held-out defer-vs-verify | System design should route through runtime enforcement before adding new training objectives. |
| Stage A component saved-prediction baselines | No-model component JSONL producer scores oracle, ground/supported collapse, missing-citation routing, and tool-names-only baselines | Keeps cluster/API component outputs comparable to deterministic offline gates while routing repairs move from margin scoring to saved predictions. |
| Native CT tool-use reference | 40/40 clean native-profile reference trajectories | The task/harness can express high-quality grounded trajectories. |
| Evidence-rationale guardrail | model 0.50 -> guardrail 1.00; 20 rescued, 0 introduced | Deterministic evidence checks can recover failures that prompting/SFT alone misses. |
| Boundary preference data | held-out pairwise margin wins; all-candidate selection still weak at 0.25 | Preference data is useful, but naive all-candidate ranking collapses without tighter candidates. |
| A2 ambiguous-band reranker | SapBERT 0.750; Qwen2.5-7B 0.810; SFT-1.5B 0.875; Claude-haiku 0.900; ceiling 0.970 | A small open reranker closes most of the ambiguous entity-resolution gap without API dependency. |
| From-base GRPO on A2 band | band_cases 0.785 -> 0.755; band_eval 0.402 -> 0.710 | GRPO learned the easier abstain gate, not the hard in-band disambiguation. |

## Quickstart

Public-safe smoke path:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-public.txt
python scripts/check_public_release.py
python scripts/check_public_git_history.py
python scripts/check_research_plan.py
python post_training/validate_post_training_data.py
python examples/run_public_demo.py
python post_training/run_stage_a_sft_smoke_eval.py --json
python post_training/generate_stage_a_predictions.py \
  --mode self_answer \
  --sft post_training/stage_a_sft_heldout_v1.jsonl \
  --out /tmp/stage_a_self_answer_predictions.jsonl \
  --run-id self_answer_saved_prediction_smoke
python post_training/evaluate_stage_a_predictions.py \
  --predictions /tmp/stage_a_self_answer_predictions.jsonl \
  --expected-sft post_training/stage_a_sft_heldout_v1.jsonl \
  --run-id self_answer_saved_prediction_smoke \
  --json
python post_training/run_stage_a_saved_prediction_candidate_readout.py \
  --dry-run \
  --out-dir /tmp/stage_a_saved_candidate_readout \
  --run-id stage_a_saved_candidate_readout_dry
python post_training/run_stage_a_saved_output_calibration_margin_sft.py \
  --dry-run \
  --out-dir /tmp/stage_a_saved_output_calibration_margin_sft \
  --run-id stage_a_saved_output_calibration_margin_sft_dry \
  --pairwise-margin-weight 1 \
  --pairwise-margin 0.05 \
  --score-base-margins \
  --score-train-margins \
  --score-base-candidates \
  --score-trained-candidates
python post_training/evaluate_stage_a_predictions.py \
  --predictions post_training/stage_a_sft_heldout_v1.jsonl \
  --expected-sft post_training/stage_a_sft_heldout_v1.jsonl \
  --run-id heldout_oracle_adapter_smoke \
  --json
python post_training/run_stage_a_strict_component_diagnostics.py --compact
python post_training/analyze_stage_a_component_visibility.py \
  --out-json /tmp/stage_a_component_visibility_audit.json \
  --out-md /tmp/stage_a_component_visibility_audit.md
python post_training/export_stage_a_evidence_conditioned_component_targets.py
python post_training/analyze_stage_a_component_visibility.py \
  --targets post_training/stage_a_evidence_conditioned_component_targets_v1.jsonl \
  --out-json /tmp/stage_a_evidence_conditioned_component_visibility_audit.json \
  --out-md /tmp/stage_a_evidence_conditioned_component_visibility_audit.md
python post_training/run_stage_a_strict_component_sft_smoke.py \
  --dry-run \
  --component enum_action \
  --targets post_training/stage_a_evidence_conditioned_component_targets_v1.jsonl \
  --train-targets post_training/stage_a_evidence_conditioned_component_targets_train_v1.jsonl \
  --heldout-targets post_training/stage_a_evidence_conditioned_component_targets_heldout_v1.jsonl
python post_training/run_stage_a_strict_component_sft_smoke.py \
  --dry-run \
  --component enum_action \
  --decode-mode enum_observed_pair_score
python post_training/run_stage_a_strict_component_sft_smoke.py \
  --dry-run \
  --component routing_after_loop \
  --decode-mode routing_observed_pair_score \
  --targets post_training/stage_a_evidence_conditioned_component_targets_v1.jsonl \
  --train-targets post_training/stage_a_evidence_conditioned_component_targets_train_v1.jsonl \
  --heldout-targets post_training/stage_a_evidence_conditioned_component_targets_heldout_v1.jsonl
python post_training/export_stage_a_routing_action_status_contrast_pairs.py
python post_training/export_stage_a_routing_defer_verify_contrast_pairs.py
python post_training/run_stage_a_routing_contrast_sft_smoke.py \
  --dry-run \
  --pairwise-margin-weight 1 \
  --pairwise-margin 0.05 \
  --score-base-routing-candidates \
  --score-trained-routing-candidates
python post_training/run_stage_a_routing_contrast_sft_smoke.py \
  --dry-run \
  --pairs post_training/stage_a_routing_defer_verify_contrast_pairs_v1.jsonl \
  --train-pairs post_training/stage_a_routing_defer_verify_contrast_pairs_train_v1.jsonl \
  --heldout-pairs post_training/stage_a_routing_defer_verify_contrast_pairs_heldout_v1.jsonl \
  --manifest post_training/stage_a_routing_defer_verify_contrast_pairs_manifest.json \
  --pairwise-margin-weight 1 \
  --pairwise-margin 0.05 \
  --score-base-routing-candidates \
  --score-trained-routing-candidates
python post_training/evaluate_stage_a_routing_evidence_boundary_gate.py \
  --out-json /tmp/stage_a_routing_evidence_boundary_gate.json \
  --out-md /tmp/STAGE_A_ROUTING_EVIDENCE_BOUNDARY_GATE.md
python post_training/evaluate_stage_a_routing_evidence_gate.py \
  --out-json /tmp/stage_a_routing_evidence_gate.json \
  --out-md /tmp/STAGE_A_ROUTING_EVIDENCE_GATE.md
python post_training/evaluate_stage_a_routing_gate_baseline_comparison.py \
  --out-json /tmp/stage_a_routing_gate_baseline_comparison.json \
  --out-md /tmp/STAGE_A_ROUTING_GATE_BASELINE_COMPARISON.md
python post_training/evaluate_stage_a_routing_model_readiness.py \
  --out-json /tmp/stage_a_routing_model_readiness.json \
  --out-md /tmp/STAGE_A_ROUTING_MODEL_READINESS.md
python post_training/evaluate_stage_a_full_trajectory_arbitration.py \
  --out-json /tmp/stage_a_full_trajectory_arbitration.json \
  --out-md /tmp/STAGE_A_FULL_TRAJECTORY_ARBITRATION.md
python post_training/evaluate_stage_a_saved_prediction_readiness.py \
  --out-json /tmp/stage_a_saved_prediction_readiness.json \
  --out-md /tmp/STAGE_A_SAVED_PREDICTION_READINESS.md
python post_training/evaluate_stage_a_saved_output_candidate_arbitration.py \
  --out-json /tmp/stage_a_saved_output_candidate_arbitration.json \
  --out-md /tmp/STAGE_A_SAVED_OUTPUT_CANDIDATE_ARBITRATION.md
python post_training/evaluate_stage_a_saved_output_next_decision.py \
  --out-json /tmp/stage_a_saved_output_next_decision.json \
  --out-md /tmp/STAGE_A_SAVED_OUTPUT_NEXT_DECISION.md
python post_training/evaluate_stage_a_saved_output_meet_or_beat_gate.py \
  --out-json /tmp/stage_a_saved_output_meet_or_beat_gate.json \
  --out-md /tmp/STAGE_A_SAVED_OUTPUT_MEET_OR_BEAT_GATE.md
python post_training/evaluate_stage_a_routing_gate_arbitration.py \
  --out-json /tmp/stage_a_routing_gate_arbitration.json \
  --out-md /tmp/STAGE_A_ROUTING_GATE_ARBITRATION.md
python post_training/export_stage_a_enum_corrective_pairs.py
python post_training/run_stage_a_enum_corrective_sft_smoke.py --dry-run
python post_training/run_stage_a_enum_corrective_sft_smoke.py \
  --dry-run \
  --focus-chosen-pairs flag/invalid_value \
  --focus-repeat 4 \
  --target-format full \
  --pairwise-margin-weight 1 \
  --pairwise-margin 0.05 \
  --candidate-ce-weight 1 \
  --candidate-ce-mode field \
  --score-base-enum-candidates \
  --score-enum-candidates \
  --enum-candidate-policy pair_observed_outputs
python -m pytest -q \
  tests/test_trajectory_evaluator.py \
  tests/test_public_demo.py \
  tests/test_public_release_checker.py \
  tests/test_stage_a_manifest.py \
  tests/test_stage_a_manifest_eval_script.py \
  tests/test_stage_a_export.py \
  tests/test_stage_a_strict_contract_export.py \
  tests/test_stage_a_strict_contract_sft_smoke.py \
  tests/test_stage_a_strict_component_diagnostics.py \
  tests/test_stage_a_strict_component_targets.py \
  tests/test_stage_a_strict_component_sft_smoke.py \
  tests/test_stage_a_routing_action_status_contrast_pairs.py \
  tests/test_stage_a_routing_contrast_sft_smoke.py \
  tests/test_stage_a_enum_corrective_pairs.py \
  tests/test_stage_a_enum_corrective_sft_smoke.py \
  tests/test_stage_a_routing_evidence_gate.py \
  tests/test_stage_a_routing_gate_baseline_comparison.py \
  tests/test_stage_a_routing_model_readiness.py \
  tests/test_stage_a_full_trajectory_arbitration.py \
  tests/test_stage_a_saved_prediction_readiness.py \
  tests/test_stage_a_saved_output_calibration_margin_sft.py \
  tests/test_stage_a_split.py \
  tests/test_stage_a_sft_smoke_eval.py \
  tests/test_stage_a_prediction_eval.py \
  tests/test_stage_a_prediction_generator.py \
  tests/test_post_training_data_validator.py
```

The public demo uses only synthetic cases from [demo/public_trajectory_cases.jsonl](demo/public_trajectory_cases.jsonl).
It does not require the private NegBioDB SQLite database, model weights, or API keys.

Full local experiment path:

```bash
pip install -r requirements.txt
python -m pytest -q
```

The full path installs model-training and API-client dependencies used by local
post-training, prompt-only, and reranker experiments.

## Public Artifact Index

- [Benchmark Card](BENCHMARK_CARD.md): Stage A task schema, action space,
  evaluator gates, baselines, and limitations.
- [Benchmark Verifier Map](BENCHMARK_VERIFIER_MAP.md): verifier ladder,
  escalation gates, and current Stage A runtime-enforcement scorecard.
- [Reviewer Guide](REVIEWER_GUIDE.md): fast review paths and claim-boundary
  checklist.
- [Dataset Card](DATASET_CARD.md): Stage A schema, failure modes, intended use,
  and limitations.
- [Model Card](MODEL_CARD.md): optional future A2 reranker model surface.
- [Reproducibility](REPRODUCIBILITY.md): public-safe and full local validation
  paths.
- [Roadmap](ROADMAP.md): research-first Stage A component gates, C5 transfer,
  and release milestones.
- [Changelog](CHANGELOG.md): public-facing release history.
- [Release Manifest](release/public_release_manifest.json): record counts,
  checksums, and publishability flags for public artifacts.
- [Release Surface](release/README.md): validation gate and publication boundary.
- [Contributing](CONTRIBUTING.md): benchmark contribution scope and local checks.
- [Security Policy](SECURITY.md): private-data and leakage reporting guidance.

For NegBioDB-CT runners that need the private SQLite database, set:

```bash
export NEGBIODB_ROOT=/path/to/Negative_result_DB
export NEGBIODB_CT_DB=/path/to/negbiodb_ct.db
```

For A2 reranker scripts that need local MONDO/SapBERT artifacts, set:

```bash
export A2_FREETEXT_DIR=/path/to/a2_freetext
```

## Research Summary

The concise scientific overview is in [RESEARCH_SUMMARY.md](RESEARCH_SUMMARY.md).
For a compact visual/result summary, see [figures/RESULTS_SNAPSHOT.md](figures/RESULTS_SNAPSHOT.md).

## Release Status

- GitHub target: [jang1563/LLM_SFM_tool_deployment_public](https://github.com/jang1563/LLM_SFM_tool_deployment_public).
  This tree is the validated clean-snapshot release surface; live visibility is
  verified separately from the repository contents.
- Earlier private development history is intentionally excluded from this
  release surface.
- Hugging Face: project-specific model/dataset/space repositories are not yet
  published.
- Stage A benchmark artifacts are now registered in the public-release manifest
  with record counts and checksums.
- Draft release docs are included here: [MODEL_CARD.md](MODEL_CARD.md),
  [DATASET_CARD.md](DATASET_CARD.md), and [RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md).
- Public-release metadata and safety checks are tracked in
  [release/public_release_manifest.json](release/public_release_manifest.json),
  [scripts/check_public_release.py](scripts/check_public_release.py), and
  [scripts/check_public_git_history.py](scripts/check_public_git_history.py).
- License is currently all-rights-reserved until an explicit open-source license
  decision is made.

## Limitations

- Several CT experiments depend on the private NegBioDB SQLite substrate and
  should be released only with a public-compatible data slice or manifest.
- Some post-training artifacts are oracle-derived by design; they are evaluation
  and training scaffolds, not claims that the model discovered hidden evidence.
- The A2 de-leak correction retracts earlier full-parity language. The
  honest result is stronger for deployment: SFT-1.5B reaches 0.875, above
  SapBERT and Qwen2.5-7B zero-shot, but below Claude-haiku at 0.900.
- HPC scripts are run recipes. They need site-specific account, storage, and
  model-cache configuration before public reuse.

## Source Anchors

- [SOURCE_MAP.md](SOURCE_MAP.md) separates locally verified source material from
  broader scan targets.
- [post_training/README.md](post_training/README.md) explains the NegBioDB-CT
  post-training artifacts and validator.
- [research/2026-06-25_posttrain_tool_use_landscape/LONG_TERM_RESEARCH_PLAN_2026-07-04.md](research/2026-06-25_posttrain_tool_use_landscape/LONG_TERM_RESEARCH_PLAN_2026-07-04.md)
  is the active long-term research execution plan.
- [a2_freetext/A2_RERANKER_RL_REPORT_2026-06-28.md](a2_freetext/A2_RERANKER_RL_REPORT_2026-06-28.md)
  is the current corrected A2/reranker report.
- [POST_TRAIN_DEPLOYMENT_FRAMING.md](POST_TRAIN_DEPLOYMENT_FRAMING.md) records
  the broader research and deployment framing.
- [demo/README.md](demo/README.md) documents the public-safe synthetic trajectory demo.

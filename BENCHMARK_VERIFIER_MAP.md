# Benchmark Verifier Map

This project treats model output as one signal inside a verifier-enforced
scientific tool-use system. The benchmark is not passed by fluent prose; it is
passed by a valid `tool -> evidence packet -> terminal action` trajectory.

## Verifier Layers

| Layer | Artifact or entrypoint | What it checks | Failure caught |
| --- | --- | --- | --- |
| Manifest boundary | `negbiodb_ct/stage_a_manifest.py` | Hidden labels stay outside model-visible prompts; split groups and source IDs are auditable. | label leakage, train/eval source overlap, missing required fields |
| Trajectory evaluator | `llm_sfm_tool_deployment/trajectory.py` | Required tools, evidence status, attribution, terminal action, policy compliance. | self-answering, wrong action, missing citation, unsupported trust |
| Stage A trajectory adapter | `negbiodb_ct/stage_a_manifest.py::score_stage_a_trajectory` | Generic trajectory score plus Stage A query-field completeness. | partial query, missing tool arguments, shortcut tool traces |
| Saved prediction scorer | `post_training/evaluate_stage_a_predictions.py` | Provider-agnostic saved outputs scored offline through the same evaluator. | parse errors, missing/extra case IDs, non-reproducible live-only claims |
| Component scorer | `post_training/run_stage_a_strict_component_sft_smoke.py` | Enum/action, tool-query, and routing-after-loop slices. | schema drift, invalid enum values, citationless routing |
| Evidence visibility audit | `post_training/analyze_stage_a_component_visibility.py` | Whether model-visible prompts expose the evidence needed for routing. | underconditioned routing targets, hidden-label dependence |
| Routing evidence gate | `post_training/evaluate_stage_a_routing_evidence_gate.py` | Deterministic routing from prompt-visible tool-result fields. | unsupported trust, insufficient-as-negative, invalid-value misses |
| Baseline comparison | `post_training/evaluate_stage_a_routing_gate_baseline_comparison.py` | Runtime gate versus oracle, collapse, citationless, and empty-object routing. | unsafe `ground` / `supported` collapse, incomplete evidence packets |
| Model-readiness gate | `post_training/evaluate_stage_a_routing_model_readiness.py` | Existing compact model summaries against deterministic baselines. | premature `tool_query`, DPO/RLVR, HF, or release escalation |
| Full trajectory arbitration | `post_training/evaluate_stage_a_full_trajectory_arbitration.py` | Runtime/hybrid policies projected into the canonical full trajectory evaluator. | component-only wins that fail tool, citation, or terminal-action gates |
| Saved-prediction readiness | `post_training/evaluate_stage_a_saved_prediction_readiness.py` | Compact real saved-output summaries and saved-candidate fail-closed gates against full-trajectory arbitration baselines. | treating adapter sanity checks, weak saved outputs, or narrow gate coverage as optimization readiness |
| Saved-candidate gate | `post_training/analyze_stage_a_saved_candidate_gate.py` | Score-gap fail-closed thresholds over ignored saved candidate-score JSONL. | trusting collapsed candidate top-1 without calibration or runtime enforcement |
| Saved-output next decision | `post_training/evaluate_stage_a_saved_output_next_decision.py` | Public-safe checkpoint-derived selection of the next Stage A saved-output experiment. | broad retraining, `tool_query`, DPO/RLVR, HF, or release escalation without passing readiness evidence |
| Saved-output calibration probe | `post_training/export_stage_a_saved_output_calibration_probe.py` | Split-safe target-vs-`ground`/`supported` pairs derived from the saved-output next-decision checkpoint. | mixing held-out rows into calibration, broad retraining drift, or treating collapse repair as trajectory readiness |
| Saved-output calibration readout | `post_training/run_stage_a_saved_output_calibration_probe_readout.py` | Train-selected score-gap gate over target-vs-collapse probe outputs. | trusting high-gap collapse, tuning on held-out probe rows, or treating dry-run as model evidence |
| Saved-output margin SFT | `post_training/run_stage_a_saved_output_calibration_margin_sft.py` | Train-only target-vs-collapse margin repair with held-out margin and delta reports. | using held-out probe rows for training, treating targeted SFT movement as DPO/RLVR, or skipping base-vs-trained verification |

## Current Gate Ladder

| Gate | Current result | Decision |
| --- | ---: | --- |
| Stage A oracle trajectory | 25/25 | Evaluator can express the target trajectories. |
| Self-answer baseline | 0/25 | Tool use remains required. |
| `ground` / `supported` collapse | 5/25; 20 unsafe overrides | Collapse is unsafe and not a model-readiness signal. |
| Citationless routing | 15/25; 10 attribution failures | Action/status alone is not a complete evidence packet. |
| Runtime evidence gate | 25/25 | Runtime enforcement is the Stage A routing baseline to beat. |
| Best all-family Cayuga routing readout | 2/5 held-out | Not ready for `tool_query`, DPO/RLVR, HF, or release tagging. |
| Full-trajectory hybrid runtime-over-collapse | 25/25 | Runtime arbitration can rescue model-like collapse when the full trajectory still satisfies tool/query/citation gates. |
| Best real saved-output trajectory summary | 0/5 held-out | Existing saved model outputs remain below collapse, citationless, and runtime full-trajectory baselines. |
| v3 tool-trace Cayuga saved output | 0/5 held-out; 5 invalid `verified` statuses | Repair canonical enum/schema emission before interpreting evidence routing or tool-query competence. |
| v4 canonical JSON Cayuga saved output | 0/5 held-out; 5 parse errors | The invalid-status failure moved to missing `action`/JSON-envelope failure; prompt-only repair is not enough. |
| Candidate-readout Cayuga saved output | 0/5 held-out; mean 0.657; no parse/tool/query failures | Constrained candidates expose action/status collapse to `ground` / `supported`; runtime gate remains required. |
| Saved-candidate score-gap gate | 1/5 trusted, 0 unsafe trust, 2/5 fail-closed strict final correct | Score gaps can identify only the supported row; this is not enough to beat runtime enforcement or reopen optimization/release gates. |
| Saved-output calibration probe | 20 pairs; 16 train-allowed, 4 held-out evaluation-only, 0 source overlap | A targeted repair substrate now exists, but it does not by itself reopen `tool_query`, DPO/RLVR, HF, or release tagging. |
| Saved-output calibration readout | 0/20 exact top-1; top pair is `ground` / `supported` in 20/20 | Qwen2.5-0.5B still prefers collapse over every target repair pair; readiness remains gated. |
| Saved-output margin SFT path | dry-run validates 16 train-only and 4 held-out probe rows | Next Cayuga diagnostic can test narrow supervised movement; no readiness claim changes until compact held-out margins and deltas pass. |
| Saved-output margin SFT result | held-out wins 0/4 -> 1/4; mean margin -0.081 -> -0.046 | Partial supervised movement, not repair; runtime enforcement and escalation gates remain required. |
| Saved-output focused margin SFT result | held-out wins 0/4 -> 3/4; mean margin -0.081 -> -0.003 | Stronger corrective movement, but `flag` / `invalid_value` remains unresolved; no `tool_query`, DPO/RLVR, HF, or release-tag escalation. |
| Saved-output target-format diagnostic | `--target-format full|action_status_only|action_only|status_only` | Isolates action/status label learning from full JSON/citation/tool-call formatting before any optimizer escalation. |
| Saved-output target-format result | action-only and action+status flip held-out `flag` / `invalid_value`; full JSON remains negative | Remaining repair should target full-output scoring/coupling, not broader optimizer escalation. |
| Saved-output same-model target-format scoring | `--score-target-formats full:action_only:action_status_only:status_only` | Scores multiple target projections on the same trained model so full-target coupling is not confused with separate projection training. |
| Saved-output same-model target-format result | full-target held-out wins 0/4 -> 4/4; full `flag` / `invalid_value` margin -0.175 -> +0.026 | Meaningful teacher-forced repair signal, but candidate top-1, free generation, and full-trajectory readiness are still unproven. |
| Saved-output candidate-rank result | 5-way candidate top-1 is 1/4 after full-target repair; trained top pair is `flag` / `invalid_value` in 4/4 rows | Teacher-forced repair does not yet survive candidate selection; fail closed and keep escalation gated. |
| Saved-output candidate field-rank analyzer | `post_training/analyze_stage_a_saved_output_candidate_fields.py` | Separates action-field, evidence-status-field, and joint pair-selection failures from ignored candidate-score JSONL without publishing raw tables. |
| Saved-output candidate field result | trained non-flag rows are 3/3 `both_field_failure`; `flag` / `invalid_value` is the only pair top-1 | Candidate failure is combined pair over-selection, so next repair should test calibration or pair/field routing under runtime gates. |
| Saved-output non-flag balanced checkpoint | raw/calibrated held-out candidate top-1 remains 1/4; runtime evidence and hybrid arbitration stay 4/4 | Simple pair oversampling changes the collapse pattern but does not teach evidence-conditioned action/status selection. |
| Saved-output candidate-CE checkpoint spec | `post_training/evaluate_stage_a_saved_output_candidate_ce_checkpoint_spec.py` | Adds an explicit pair+field candidate-routing objective and keeps the same meet-or-beat runtime baseline gate. |
| Saved-output candidate-CE checkpoint result | raw/calibrated held-out candidate top-1 remains 1/4; raw top pair is `verify` / `insufficient` in 4/4 | Standalone candidate-routing SFT changes the prior but still fails the runtime evidence arbitration baseline. |
| Saved-output post-candidate-CE next decision | `post_training/evaluate_stage_a_saved_output_post_candidate_ce_next_decision.py` | Selects an evidence-conditioned saved-output bridge before more standalone SFT, `tool_query`, DPO/RLVR, HF, or release tagging. |
| Saved-output evidence bridge | `post_training/evaluate_stage_a_saved_output_evidence_bridge.py` | Joins compact candidate-failure case IDs to prompt-visible runtime evidence reasons without reading raw candidate-score tables. |
| Saved-output evidence candidate-routing rows | `post_training/export_stage_a_saved_output_evidence_candidate_routing_rows.py` | Exports 25 finite-candidate rows with 20/5 split preservation and held-out bridge-focus flags before any new model-heavy objective. |
| Saved-output evidence candidate-routing readout | `post_training/evaluate_stage_a_saved_output_evidence_candidate_routing_readout.py` | Compares runtime evidence routing against static single-pair priors over the candidate-routing substrate before a Cayuga smoke spec. |
| Saved-output evidence candidate-routing smoke spec | `post_training/evaluate_stage_a_saved_output_evidence_candidate_routing_smoke_spec.py` | Defines the next small Cayuga smoke contract and requires 5/5 held-out plus 4/4 bridge-focus exact before escalation. |
| Saved-output evidence candidate-routing smoke runner | `post_training/run_stage_a_saved_output_evidence_candidate_routing_smoke.py` | Validates the finite-candidate slice locally and gates full Cayuga model loading behind `--allow-model-load`. |
| Saved-output evidence candidate-routing smoke result | `post_training/evaluate_stage_a_saved_output_evidence_candidate_routing_smoke_result.py` | Converts compact runner eval reports into pass/fail summaries, rejects raw candidate-score or raw model-text fields, and redacts external compact input paths. |
| Saved-output evidence candidate-routing dry-run checkpoint | `post_training/evaluate_stage_a_saved_output_evidence_candidate_routing_dry_run_checkpoint.py` | Records public-safe local/Cayuga dry-run readiness and keeps full submission behind an explicit decision. |
| Saved-output evidence candidate-routing Cayuga result | train 4/20, held-out 1/5, bridge-focus 1/4; all predictions are `verify` / `insufficient` | Fails the 5/5 runtime-baseline gate and freezes the reused slice as diagnostic data before a new sealed evaluation extension. |
| Stage A sealed extension commitment | `post_training/build_stage_a_sealed_extension.py` and `post_training/stage_a_sealed_extension_commitment_2026-07-10.json` | Requires private external paths, excludes public source-task/split-group/claim overlap, validates five-family balance, and publishes hashes plus aggregate checks without row-level labels. The current 25-row commitment passes with all overlap counts at 0. |

## Escalation Rule

Do not move to `tool_query`, DPO/RLVR, Hugging Face publication, or release
tagging from model scores alone. A model path must first beat collapse and
citationless baselines, then approach runtime-gate full-trajectory performance
on broader held-out slices.

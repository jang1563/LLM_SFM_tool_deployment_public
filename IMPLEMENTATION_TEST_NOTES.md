# Implementation Test Notes

Date: 2026-06-25

## What Was Implemented

This is the first executable slice of the research plan:

```text
task -> tool loop -> evidence packet -> terminal action -> deterministic score
```

The code is intentionally not a full model trainer yet. It is a trajectory
evaluator plus a minimal SFT smoke path that can score prompt-only, SFT, DPO, or
RLVR-generated traces once those traces share the same schema.

## Files

| path | purpose |
| --- | --- |
| `llm_sfm_tool_deployment/trajectory.py` | Action enum, evidence packet, trajectory schema, deterministic evaluator. |
| `negbiodb_ct/adapter.py` | Converts NegBioDB-CT pilot task records into `TaskSpec` and scoreable trajectories. |
| `negbiodb_ct/model_output.py` | Parses prompt/SFT/DPO/RL-style JSON outputs into scoreable trajectories. |
| `negbiodb_ct/MODEL_OUTPUT_SCHEMA.md` | Documents the compact JSON object model runs should emit. |
| `tests/test_trajectory_evaluator.py` | Unit tests for NegBioDB/NullAtlas-style loops and C5 trust-gate policy. |
| `tests/test_negbiodb_ct_adapter.py` | Unit tests for task-record to evaluator-schema mapping. |
| `tests/test_negbiodb_ct_model_output.py` | Unit tests for model-output parsing and scoring. |
| `tests/test_negbiodb_ct_runner.py` | Unit tests for runner-output normalization into model-output JSON. |
| `examples/run_toy_trajectory_eval.py` | No-API smoke demo for one good and one bad trajectory. |
| `examples/run_negbiodb_ct_adapter_demo.py` | Smoke demo over real `tasks_pilot.jsonl` records. |
| `examples/run_negbiodb_ct_baselines.py` | Tiny Gate-1-style oracle/self-answer/constant-policy baseline runner. |
| `examples/run_negbiodb_ct_model_output_demo.py` | Smoke demo for prompt-style JSON -> trajectory -> evaluator. |
| `examples/analyze_negbiodb_ct_runner_results.py` | Re-scores saved runner outputs under full-loop and native-tool profiles. |
| `examples/analyze_negbiodb_ct_prompt_only.py` | Summarizes prompt-only no-tool baseline failure modes. |
| `negbiodb_ct/run_agent.py` | Existing LLM runner, now also writes parser-compatible `model_output`, generic score, and generic violations. |
| `negbiodb_ct/run_prompt_only.py` | Runs the same CT packet set without tools as a prompt-only baseline. |
| `negbiodb_ct/run_open_model_prompt_only.py` | Runs a local Hugging Face causal LM on the same packet set as a trainable-model baseline. |
| `negbiodb_ct/export_post_training_data.py` | Exports clean native runner outputs into SFT JSONL and preference-pair JSONL. |
| `post_training/` | First tracked post-training data artifacts generated from the clean n=40 native runner reference. |
| `post_training/split_sft_data.py` | Builds deterministic stratified SFT train/held-out splits. |
| `post_training/build_sft_cv_splits.py` | Builds deterministic stratified cross-validation folds from the tracked native SFT examples. |
| `post_training/export_oracle_sft_data.py` | Exports larger full-pilot deterministic-oracle SFT examples with native CT tool observations. |
| `post_training/build_sft_pressure_data.py` | Builds native pressure folds and class-balanced oracle SFT artifacts from the row-level failure diagnosis. |
| `post_training/analyze_sft_pressure_failures.py` | Analyzes pressure-run row-level failures and candidate ranks. |
| `post_training/build_sft_curriculum_data.py` | Builds contrast-family curriculum SFT artifacts from the pressure-failure diagnosis. |
| `post_training/build_sft_curriculum_v2_data.py` | Builds targeted curriculum-v2 artifacts from persistent curriculum-v1 failures. |
| `post_training/build_sft_boundary_rationale_data.py` | Builds paired boundary-rationale SFT folds after targeted oversampling failed. |
| `post_training/evidence_rationale.py` | Shared evidence-derived boundary-rationale rules for native CT tool observations. |
| `post_training/apply_evidence_rationale.py` | Applies the deployable evidence-rationale layer to native SFT JSONL artifacts. |
| `post_training/evaluate_evidence_guardrail.py` | Evaluates evidence-derived override as a guardrail on held-out model outputs. |
| `post_training/build_boundary_preference_data.py` | Builds terminal-action preference pairs for evidence boundary confusions. |
| `post_training/build_boundary_preference_hard_modes.py` | Filters negative-margin boundary preference pairs into a hard-mode subset. |
| `post_training/split_boundary_preference_hard_modes.py` | Builds deterministic train/held-out splits for hard boundary preference pairs. |
| `post_training/run_boundary_preference_margin.py` | Scores chosen vs rejected boundary preference actions by model likelihood margin. |
| `post_training/run_boundary_preference_dpo_smoke.py` | Runs a reference-free DPO-style pairwise smoke loop over boundary preference pairs. |
| `post_training/run_boundary_preference_candidate_eval.py` | Scores all legal final-decision candidates for boundary preference prompts. |
| `post_training/run_boundary_preference_candidate_ce_smoke.py` | Runs an all-candidate cross-entropy smoke loop over hard boundary preference prompts, with boundary scheduling and optional eval-gated checkpoint selection. |
| `post_training/analyze_sft_boundary_rationale_failures.py` | Analyzes boundary-rationale row-level failures and defer-vs-verify candidate margins. |
| `post_training/run_sft_boundary_rationale_ablation.py` | Runs held-out oracle/evidence rationale ablations against existing boundary-rationale fold states. |
| `post_training/summarize_sft_pressure_runs.py` | Summarizes full pressure-rerun outputs into tracked JSON/Markdown result anchors. |
| `post_training/summarize_sft_curriculum_run.py` | Summarizes full curriculum-rerun outputs into tracked JSON/Markdown result anchors. |
| `post_training/summarize_sft_curriculum_v2_run.py` | Summarizes full curriculum-v2 rerun outputs into tracked JSON/Markdown result anchors. |
| `post_training/summarize_sft_boundary_rationale_run.py` | Summarizes full boundary-rationale rerun outputs into tracked JSON/Markdown result anchors. |
| `post_training/summarize_sft_boundary_rationale_ablation.py` | Summarizes held-out rationale ablation outputs into tracked JSON/Markdown result anchors. |
| `post_training/analyze_sft_curriculum_failures.py` | Analyzes curriculum-run row-level failures and persistent strict/constrained errors. |
| `post_training/run_sft_cv_sweep.py` | Orchestrates repeated SFT train/loss/strict/constrained eval across CV folds. |
| `post_training/run_sft_oracle_warmstart.py` | Trains on the oracle SFT artifact and evaluates against native held-out CV folds. |
| `post_training/analyze_sft_sweep_failures.py` | Aggregates row-level SFT sweep failures, confusion matrices, recurrent packet failures, and constrained candidate ranks. |
| `post_training/run_sft_smoke.py` | Minimal local SFT smoke loop over tracked native trajectory examples. |
| `post_training/evaluate_sft_loss.py` | Teacher-forced loss comparison before and after loading an SFT smoke state. |
| `post_training/run_sft_decision_eval.py` | Reloads an SFT smoke state and evaluates final-decision generation on the matched tracked examples. |
| `post_training/run_sft_constrained_eval.py` | Scores legal final-decision candidates by likelihood to separate schema parsing from action selection. |

## Implemented Checks

- required ordered tool loop,
- source attribution,
- evidence-status match,
- terminal action match,
- web-zero cannot self-answer,
- contradicted or invalid claims must be rejected/flagged,
- uncalibrated specialist outputs cannot be trusted,
- antibody-antigen trust requires regime-matched calibration metadata,
- baseline dominance blocks specialist trust.

## Verification

```bash
python3 -m pytest -q
python3 examples/run_toy_trajectory_eval.py
python3 examples/run_negbiodb_ct_adapter_demo.py
python3 examples/run_negbiodb_ct_baselines.py
python3 examples/run_negbiodb_ct_model_output_demo.py
python3 examples/analyze_negbiodb_ct_runner_results.py negbiodb_ct/agent_sonnet_n40.json
python3 negbiodb_ct/run_agent.py --dry-run --n 5 --out negbiodb_ct/agent_dry_run.json
python3 negbiodb_ct/export_post_training_data.py --runner negbiodb_ct/agent_sonnet_n40_mixedfix.json
python3 post_training/validate_post_training_data.py
python3 negbiodb_ct/run_prompt_only.py --task-ids-from post_training/negbiodb_ct_native_sft_v1.jsonl --out negbiodb_ct/agent_prompt_only_sonnet_n40.json
python3 examples/analyze_negbiodb_ct_prompt_only.py negbiodb_ct/agent_prompt_only_sonnet_n40.json
python3 negbiodb_ct/run_open_model_prompt_only.py --model Qwen/Qwen2.5-0.5B-Instruct --task-ids-from post_training/negbiodb_ct_native_sft_v1.jsonl --out negbiodb_ct/agent_open_model_qwen05_n40.json
python3 examples/analyze_negbiodb_ct_prompt_only.py negbiodb_ct/agent_open_model_qwen05_n40.json
python3 post_training/split_sft_data.py
python3 post_training/run_sft_smoke.py --limit 2 --max-steps 1 --batch-size 1 --max-length 512 --train-last-layers 1 --device auto --out-dir post_training/runs/qwen_sft_smoke
python3 post_training/evaluate_sft_loss.py --state post_training/runs/qwen_sft_smoke/trainable_state.pt --limit 40 --batch-size 2 --max-length 512 --device auto --out post_training/runs/qwen_sft_loss_compare.json
python3 post_training/run_sft_decision_eval.py --state post_training/runs/qwen_sft_smoke/trainable_state.pt --limit 40 --device auto --out post_training/runs/qwen_sft_decision_eval_n40.json
python3 post_training/run_sft_constrained_eval.py --limit 40 --max-length 512 --device auto --score-mode mean --out post_training/runs/qwen_sft_constrained_base_n40.json
python3 post_training/run_sft_constrained_eval.py --state post_training/runs/qwen_sft_smoke/trainable_state.pt --limit 40 --max-length 512 --device auto --score-mode mean --out post_training/runs/qwen_sft_constrained_loaded_n40.json
python3 post_training/build_sft_cv_splits.py
python3 post_training/export_oracle_sft_data.py
python3 post_training/run_sft_cv_sweep.py --dry-run --only-fold 1 --out-dir post_training/runs/qwen_sft_cv4_schema_action_80_dry --skip-strict-generation --skip-constrained
python3 post_training/run_sft_cv_sweep.py --only-fold 1 --out-dir post_training/runs/qwen_sft_cv4_runner_smoke --max-steps 1 --skip-strict-generation --skip-constrained
python3 post_training/run_sft_oracle_warmstart.py --dry-run --only-eval fold0_heldout --out-dir post_training/runs/qwen_oracle400_warmstart_dry --skip-strict-generation --skip-constrained
python3 post_training/run_sft_oracle_warmstart.py --train-limit 2 --max-steps 1 --only-eval fold0_heldout --out-dir post_training/runs/qwen_oracle_warmstart_runner_smoke --skip-strict-generation --skip-constrained
python3 post_training/analyze_sft_sweep_failures.py
python3 post_training/build_sft_pressure_data.py
python3 post_training/run_sft_cv_sweep.py --dry-run --manifest post_training/negbiodb_ct_native_sft_cv4_pressure_manifest.json --only-fold 0 --out-dir post_training/runs/qwen_sft_cv4_pressure_dry --skip-strict-generation --skip-constrained
python3 post_training/run_sft_oracle_warmstart.py --dry-run --train-sft post_training/negbiodb_ct_oracle_sft_balanced_v1.jsonl --train-limit 700 --only-eval fold0_heldout --out-dir post_training/runs/qwen_oracle_balanced_warmstart_dry --skip-strict-generation --skip-constrained
python3 post_training/run_sft_smoke.py --dry-run --sft post_training/pressure/negbiodb_ct_native_sft_cv4_pressure_v1_fold0_train.jsonl --limit 54 --max-length 512 --out-dir post_training/runs/qwen_sft_pressure_encode_dry
python3 post_training/run_sft_smoke.py --dry-run --sft post_training/negbiodb_ct_oracle_sft_balanced_v1.jsonl --limit 20 --max-length 512 --out-dir post_training/runs/qwen_oracle_balanced_encode_dry
python3 post_training/run_sft_cv_sweep.py --manifest post_training/negbiodb_ct_native_sft_cv4_pressure_manifest.json --out-dir post_training/runs/qwen_sft_cv4_pressure_schema_action_80
python3 post_training/run_sft_oracle_warmstart.py --train-sft post_training/negbiodb_ct_oracle_sft_balanced_v1.jsonl --train-limit 700 --out-dir post_training/runs/qwen_oracle_balanced_warmstart_cvheldout
python3 post_training/summarize_sft_pressure_runs.py
python3 post_training/analyze_sft_pressure_failures.py
python3 post_training/build_sft_curriculum_data.py
python3 post_training/run_sft_cv_sweep.py --dry-run --manifest post_training/negbiodb_ct_native_sft_cv4_curriculum_manifest.json --only-fold 0 --out-dir post_training/runs/qwen_sft_cv4_curriculum_dry --skip-strict-generation --skip-constrained
python3 post_training/run_sft_smoke.py --dry-run --sft post_training/curriculum/negbiodb_ct_native_sft_cv4_curriculum_v1_fold0_train.jsonl --limit 72 --max-length 512 --out-dir post_training/runs/qwen_sft_curriculum_encode_dry
python3 post_training/run_sft_cv_sweep.py --manifest post_training/negbiodb_ct_native_sft_cv4_curriculum_manifest.json --out-dir post_training/runs/qwen_sft_cv4_curriculum_schema_action_80
python3 post_training/summarize_sft_curriculum_run.py
python3 post_training/analyze_sft_curriculum_failures.py
python3 post_training/build_sft_curriculum_v2_data.py
python3 post_training/run_sft_cv_sweep.py --dry-run --manifest post_training/negbiodb_ct_native_sft_cv4_curriculum_v2_manifest.json --only-fold 0 --out-dir post_training/runs/qwen_sft_cv4_curriculum_v2_dry --skip-strict-generation --skip-constrained
python3 post_training/run_sft_smoke.py --dry-run --sft post_training/curriculum_v2/negbiodb_ct_native_sft_cv4_curriculum_v2_targeted_fold0_train.jsonl --limit 102 --max-length 512 --out-dir post_training/runs/qwen_sft_curriculum_v2_encode_dry
python3 post_training/run_sft_cv_sweep.py --manifest post_training/negbiodb_ct_native_sft_cv4_curriculum_v2_manifest.json --out-dir post_training/runs/qwen_sft_cv4_curriculum_v2_targeted_schema_action_80_evalfast --skip-train-loss --skip-base-constrained
python3 post_training/summarize_sft_curriculum_v2_run.py
python3 post_training/build_sft_boundary_rationale_data.py
python3 post_training/run_sft_cv_sweep.py --dry-run --manifest post_training/negbiodb_ct_native_sft_cv4_boundary_rationale_manifest.json --only-fold 0 --out-dir post_training/runs/qwen_sft_cv4_boundary_rationale_dry --skip-strict-generation --skip-constrained
python3 post_training/run_sft_smoke.py --dry-run --sft post_training/boundary_rationale/negbiodb_ct_native_sft_cv4_boundary_rationale_v1_fold0_train.jsonl --limit 60 --max-length 512 --out-dir post_training/runs/qwen_sft_boundary_rationale_encode_dry
python3 post_training/run_sft_smoke.py --sft post_training/boundary_rationale/negbiodb_ct_native_sft_cv4_boundary_rationale_v1_fold0_train.jsonl --limit 60 --max-steps 1 --batch-size 2 --max-length 512 --train-last-layers 1 --device auto --out-dir post_training/runs/qwen_sft_boundary_rationale_fold0_smoke
python3 post_training/run_sft_cv_sweep.py --manifest post_training/negbiodb_ct_native_sft_cv4_boundary_rationale_manifest.json --only-fold 0 --out-dir post_training/runs/qwen_sft_cv4_boundary_rationale_fold0_evalfast --skip-train-loss --skip-base-constrained
python3 post_training/run_sft_cv_sweep.py --manifest post_training/negbiodb_ct_native_sft_cv4_boundary_rationale_manifest.json --out-dir post_training/runs/qwen_sft_cv4_boundary_rationale_schema_action_80_evalfast --skip-train-loss --skip-base-constrained
python3 post_training/summarize_sft_boundary_rationale_run.py
python3 post_training/analyze_sft_boundary_rationale_failures.py
python3 post_training/run_sft_boundary_rationale_ablation.py
python3 post_training/summarize_sft_boundary_rationale_ablation.py
python3 post_training/run_sft_boundary_rationale_ablation.py --rationale-mode evidence --ablation-out-dir post_training/boundary_rationale_heldout_evidence_ablation --ablation-prefix negbiodb_ct_native_sft_cv4_boundary_rationale_heldout_evidence_v1 --ablation-manifest-out post_training/negbiodb_ct_native_sft_cv4_boundary_rationale_heldout_evidence_ablation_manifest.json --dataset negbiodb_ct_native_sft_boundary_rationale_heldout_evidence_v1 --out-dir post_training/runs/qwen_sft_cv4_boundary_rationale_heldout_evidence_ablation
python3 post_training/summarize_sft_boundary_rationale_ablation.py --root post_training/runs/qwen_sft_cv4_boundary_rationale_heldout_evidence_ablation --manifest post_training/negbiodb_ct_native_sft_cv4_boundary_rationale_heldout_evidence_ablation_manifest.json --out-json post_training/sft_boundary_rationale_heldout_evidence_ablation_summary_2026-06-27.json --out-md post_training/SFT_BOUNDARY_RATIONALE_HELDOUT_EVIDENCE_ABLATION_2026-06-27.md
python3 post_training/apply_evidence_rationale.py
python3 post_training/apply_evidence_rationale.py --sft post_training/negbiodb_ct_oracle_sft_v1.jsonl --out post_training/negbiodb_ct_oracle_sft_evidence_rationale_v1.jsonl --manifest-out post_training/negbiodb_ct_oracle_sft_evidence_rationale_manifest.json --dataset negbiodb_ct_oracle_sft_evidence_rationale_v1 --strategy pilot400_evidence_boundary_rationale_stress_v1
python3 post_training/evaluate_evidence_guardrail.py
python3 post_training/run_sft_oracle_warmstart.py --train-sft post_training/negbiodb_ct_oracle_sft_evidence_rationale_v1.jsonl --train-limit 400 --eval-cv-manifest post_training/negbiodb_ct_native_sft_cv4_manifest.json --out-dir post_training/runs/qwen_oracle400_evidence_distill_base_cvheldout --skip-base-constrained
python3 post_training/run_sft_decision_eval.py --state post_training/runs/qwen_oracle400_evidence_distill_base_cvheldout/train/trainable_state.pt --sft post_training/boundary_rationale_heldout_evidence_ablation/negbiodb_ct_native_sft_cv4_boundary_rationale_heldout_evidence_v1_fold0_heldout.jsonl --tasks negbiodb_ct/tasks_pilot.jsonl --limit 10 --device auto --max-new-tokens 64 --out post_training/runs/qwen_oracle400_evidence_distill_prompted_fold0/decision_eval.json
python3 post_training/run_sft_constrained_eval.py --state post_training/runs/qwen_oracle400_evidence_distill_base_cvheldout/train/trainable_state.pt --sft post_training/boundary_rationale_heldout_evidence_ablation/negbiodb_ct_native_sft_cv4_boundary_rationale_heldout_evidence_v1_fold0_heldout.jsonl --tasks negbiodb_ct/tasks_pilot.jsonl --limit 10 --max-length 512 --device auto --score-mode mean --out post_training/runs/qwen_oracle400_evidence_distill_prompted_fold0/constrained_loaded.json
python3 post_training/build_sft_generative_rationale_data.py
python3 post_training/build_sft_generative_rationale_data.py --sft post_training/cv/negbiodb_ct_native_sft_cv4_v1_fold0_heldout.jsonl --out post_training/generative_rationale/negbiodb_ct_native_sft_cv4_generative_rationale_v1_fold0_heldout.jsonl --manifest-out post_training/negbiodb_ct_native_sft_cv4_generative_rationale_fold0_manifest.json --dataset negbiodb_ct_native_sft_generative_rationale_fold0_v1
python3 post_training/run_sft_smoke.py --dry-run --sft post_training/negbiodb_ct_oracle_sft_generative_rationale_v1.jsonl --limit 20 --max-length 768 --out-dir post_training/runs/qwen_oracle400_generative_rationale_encode_dry
python3 post_training/run_sft_smoke.py --dry-run --sft post_training/generative_rationale/negbiodb_ct_native_sft_cv4_generative_rationale_v1_fold0_heldout.jsonl --limit 10 --max-length 768 --out-dir post_training/runs/qwen_generative_rationale_fold0_encode_dry
python3 post_training/run_sft_smoke.py --sft post_training/negbiodb_ct_oracle_sft_generative_rationale_v1.jsonl --limit 4 --max-steps 1 --batch-size 1 --max-length 768 --train-last-layers 1 --device auto --out-dir post_training/runs/qwen_generative_rationale_smoke
python3 post_training/run_sft_smoke.py --sft post_training/negbiodb_ct_oracle_sft_generative_rationale_v1.jsonl --limit 400 --max-steps 160 --batch-size 2 --max-length 768 --train-last-layers 2 --lr 5e-5 --device auto --out-dir post_training/runs/qwen_oracle400_generative_rationale_fold0_diagnostic/train
python3 post_training/run_sft_decision_eval.py --state post_training/runs/qwen_oracle400_generative_rationale_fold0_diagnostic/train/trainable_state.pt --sft post_training/generative_rationale/negbiodb_ct_native_sft_cv4_generative_rationale_v1_fold0_heldout.jsonl --tasks negbiodb_ct/tasks_pilot.jsonl --limit 10 --device auto --max-new-tokens 192 --out post_training/runs/qwen_oracle400_generative_rationale_fold0_diagnostic/generative_fold0_decision_eval.json
python3 post_training/run_sft_decision_eval.py --state post_training/runs/qwen_oracle400_generative_rationale_fold0_diagnostic/train/trainable_state.pt --sft post_training/cv/negbiodb_ct_native_sft_cv4_v1_fold0_heldout.jsonl --tasks negbiodb_ct/tasks_pilot.jsonl --limit 10 --device auto --max-new-tokens 192 --out post_training/runs/qwen_oracle400_generative_rationale_fold0_diagnostic/base_fold0_decision_eval.json
python3 post_training/run_sft_constrained_eval.py --state post_training/runs/qwen_oracle400_generative_rationale_fold0_diagnostic/train/trainable_state.pt --sft post_training/cv/negbiodb_ct_native_sft_cv4_v1_fold0_heldout.jsonl --tasks negbiodb_ct/tasks_pilot.jsonl --limit 10 --max-length 768 --device auto --score-mode mean --out post_training/runs/qwen_oracle400_generative_rationale_fold0_diagnostic/base_fold0_constrained_loaded.json
python3 post_training/build_boundary_preference_data.py
python3 post_training/run_boundary_preference_margin.py --limit 8 --max-length 768 --device auto --out post_training/runs/qwen_boundary_preference_margin/base_limit8.json
python post_training/run_boundary_preference_margin.py --max-length 768 --device auto --out post_training/runs/qwen_boundary_preference_margin/base_full.json
python3 post_training/build_boundary_preference_hard_modes.py
python post_training/run_boundary_preference_dpo_smoke.py --dry-run --limit 8 --max-length 768 --out-dir post_training/runs/qwen_boundary_preference_dpo_hard_dry
python post_training/run_boundary_preference_dpo_smoke.py --limit 8 --max-steps 16 --batch-size 1 --max-length 768 --train-last-layers 2 --lr 5e-5 --beta 0.1 --logprob-mode mean --device auto --out-dir post_training/runs/qwen_boundary_preference_dpo_hard_smoke_limit8_steps16
python3 post_training/split_boundary_preference_hard_modes.py
python post_training/run_boundary_preference_dpo_smoke.py --dry-run --preferences post_training/negbiodb_ct_oracle_boundary_preferences_hard_train_v1.jsonl --eval-preferences post_training/negbiodb_ct_oracle_boundary_preferences_hard_heldout_v1.jsonl --limit 16 --eval-limit 8 --max-length 768 --out-dir post_training/runs/qwen_boundary_preference_dpo_hard_split_dry
python post_training/run_boundary_preference_dpo_smoke.py --preferences post_training/negbiodb_ct_oracle_boundary_preferences_hard_train_v1.jsonl --eval-preferences post_training/negbiodb_ct_oracle_boundary_preferences_hard_heldout_v1.jsonl --limit 0 --eval-limit 0 --train-eval-limit 32 --max-steps 48 --batch-size 2 --max-length 768 --train-last-layers 2 --lr 5e-5 --beta 0.1 --logprob-mode mean --device auto --out-dir post_training/runs/qwen_boundary_preference_dpo_hard_split_steps48
python post_training/run_boundary_preference_candidate_eval.py --preferences post_training/negbiodb_ct_oracle_boundary_preferences_hard_heldout_v1.jsonl --limit 0 --max-length 768 --device auto --score-mode mean --out post_training/runs/qwen_boundary_preference_candidate_eval_hard_heldout/base_mean.json
python post_training/run_boundary_preference_candidate_eval.py --preferences post_training/negbiodb_ct_oracle_boundary_preferences_hard_heldout_v1.jsonl --state post_training/runs/qwen_boundary_preference_dpo_hard_split_steps48/trainable_state.pt --limit 0 --max-length 768 --device auto --score-mode mean --out post_training/runs/qwen_boundary_preference_candidate_eval_hard_heldout/dpo_loaded_mean.json
python post_training/run_boundary_preference_candidate_ce_smoke.py --dry-run --limit 0 --limit-per-action 8 --eval-limit 0 --eval-limit-per-action 4 --max-length 768 --out-dir post_training/runs/qwen_boundary_preference_candidate_ce_balanced_dry
python post_training/run_boundary_preference_candidate_ce_smoke.py --preferences post_training/negbiodb_ct_oracle_boundary_preferences_hard_train_v1.jsonl --eval-preferences post_training/negbiodb_ct_oracle_boundary_preferences_hard_heldout_v1.jsonl --limit 0 --limit-per-action 8 --eval-limit 0 --eval-limit-per-action 4 --max-steps 24 --batch-size 1 --candidate-batch-size 1 --max-length 768 --train-last-layers 2 --lr 5e-5 --temperature 1.0 --logprob-mode mean --device auto --out-dir post_training/runs/qwen_boundary_preference_candidate_ce_balanced_a8_eval4_steps24
python post_training/run_boundary_preference_candidate_ce_smoke.py --dry-run --training-schedule boundary_round_robin --phase-order defer,flag,reject --limit 0 --limit-per-action 8 --eval-limit 0 --eval-limit-per-action 4 --max-length 768 --out-dir post_training/runs/qwen_boundary_preference_candidate_ce_round_robin_dry
python post_training/run_boundary_preference_candidate_ce_smoke.py --preferences post_training/negbiodb_ct_oracle_boundary_preferences_hard_train_v1.jsonl --eval-preferences post_training/negbiodb_ct_oracle_boundary_preferences_hard_heldout_v1.jsonl --training-schedule boundary_round_robin --phase-order defer,flag,reject --limit 0 --limit-per-action 8 --eval-limit 0 --eval-limit-per-action 4 --max-steps 24 --batch-size 1 --candidate-batch-size 1 --max-length 768 --train-last-layers 2 --lr 5e-5 --temperature 1.0 --logprob-mode mean --device auto --out-dir post_training/runs/qwen_boundary_preference_candidate_ce_round_robin_a8_eval4_steps24
python post_training/run_boundary_preference_candidate_ce_smoke.py --dry-run --training-schedule boundary_round_robin --phase-order defer,flag,reject,flag --limit 0 --limit-per-action 8 --eval-limit 0 --eval-limit-per-action 4 --max-length 768 --out-dir post_training/runs/qwen_boundary_preference_candidate_ce_flag_rehearsal_dry
python post_training/run_boundary_preference_candidate_ce_smoke.py --preferences post_training/negbiodb_ct_oracle_boundary_preferences_hard_train_v1.jsonl --eval-preferences post_training/negbiodb_ct_oracle_boundary_preferences_hard_heldout_v1.jsonl --training-schedule boundary_round_robin --phase-order defer,flag,reject,flag --limit 0 --limit-per-action 8 --eval-limit 0 --eval-limit-per-action 4 --max-steps 24 --batch-size 1 --candidate-batch-size 1 --max-length 768 --train-last-layers 2 --lr 5e-5 --temperature 1.0 --logprob-mode mean --device auto --out-dir post_training/runs/qwen_boundary_preference_candidate_ce_flag_rehearsal_a8_eval4_steps24
python post_training/run_boundary_preference_candidate_ce_smoke.py --preferences post_training/negbiodb_ct_oracle_boundary_preferences_hard_train_v1.jsonl --eval-preferences post_training/negbiodb_ct_oracle_boundary_preferences_hard_heldout_v1.jsonl --training-schedule boundary_round_robin --phase-order defer,flag,reject,flag --limit 0 --limit-per-action 8 --eval-limit 0 --eval-limit-per-action 4 --max-steps 23 --batch-size 1 --candidate-batch-size 1 --max-length 768 --train-last-layers 2 --lr 5e-5 --temperature 1.0 --logprob-mode mean --device auto --out-dir post_training/runs/qwen_boundary_preference_candidate_ce_flag_rehearsal_a8_eval4_steps23
python post_training/run_boundary_preference_candidate_ce_smoke.py --dry-run --training-schedule boundary_round_robin --phase-order defer,flag,reject --limit 0 --limit-per-action 2 --eval-limit 0 --eval-limit-per-action 2 --eval-checkpoint-every 1 --checkpoint-selection min_action_accuracy --max-length 768 --out-dir post_training/runs/qwen_boundary_preference_candidate_ce_eval_gated_dry
python post_training/run_boundary_preference_candidate_ce_smoke.py --preferences post_training/negbiodb_ct_oracle_boundary_preferences_hard_train_v1.jsonl --eval-preferences post_training/negbiodb_ct_oracle_boundary_preferences_hard_heldout_v1.jsonl --training-schedule boundary_round_robin --phase-order defer,flag,reject --limit 0 --limit-per-action 2 --eval-limit 0 --eval-limit-per-action 2 --max-steps 6 --batch-size 1 --candidate-batch-size 1 --max-length 768 --train-last-layers 1 --lr 5e-5 --temperature 1.0 --logprob-mode mean --eval-checkpoint-every 1 --checkpoint-selection min_action_accuracy --device auto --out-dir post_training/runs/qwen_boundary_preference_candidate_ce_eval_gated_plumbing_a2_eval2_steps6
python post_training/run_boundary_preference_candidate_ce_smoke.py --preferences post_training/negbiodb_ct_oracle_boundary_preferences_hard_train_v1.jsonl --eval-preferences post_training/negbiodb_ct_oracle_boundary_preferences_hard_heldout_v1.jsonl --training-schedule boundary_round_robin --phase-order defer,flag,reject --limit 0 --limit-per-action 8 --eval-limit 0 --eval-limit-per-action 4 --max-steps 24 --batch-size 1 --candidate-batch-size 1 --max-length 768 --train-last-layers 2 --lr 5e-5 --temperature 1.0 --logprob-mode mean --eval-checkpoint-every 3 --checkpoint-selection min_action_accuracy --device auto --out-dir post_training/runs/qwen_boundary_preference_candidate_ce_eval_gated_a8_eval4_steps24_every3
python post_training/run_boundary_preference_candidate_ce_smoke.py --dry-run --training-schedule boundary_phase_batch --phase-order defer,flag,reject --limit 0 --limit-per-action 2 --eval-limit 0 --eval-limit-per-action 2 --max-length 768 --out-dir post_training/runs/qwen_boundary_preference_candidate_ce_phase_batch_dry
python post_training/run_boundary_preference_candidate_ce_smoke.py --preferences post_training/negbiodb_ct_oracle_boundary_preferences_hard_train_v1.jsonl --eval-preferences post_training/negbiodb_ct_oracle_boundary_preferences_hard_heldout_v1.jsonl --training-schedule boundary_phase_batch --phase-order defer,flag,reject --limit 0 --limit-per-action 2 --eval-limit 0 --eval-limit-per-action 2 --max-steps 3 --batch-size 1 --candidate-batch-size 1 --max-length 768 --train-last-layers 1 --lr 5e-5 --temperature 1.0 --logprob-mode mean --eval-checkpoint-every 1 --checkpoint-selection min_action_accuracy --device auto --out-dir post_training/runs/qwen_boundary_preference_candidate_ce_phase_batch_plumbing_a2_eval2_steps3
python post_training/run_boundary_preference_candidate_ce_smoke.py --preferences post_training/negbiodb_ct_oracle_boundary_preferences_hard_train_v1.jsonl --eval-preferences post_training/negbiodb_ct_oracle_boundary_preferences_hard_heldout_v1.jsonl --training-schedule boundary_phase_batch --phase-order defer,flag,reject --limit 0 --limit-per-action 8 --eval-limit 0 --eval-limit-per-action 4 --max-steps 8 --batch-size 1 --candidate-batch-size 1 --max-length 768 --train-last-layers 2 --lr 5e-5 --temperature 1.0 --logprob-mode mean --eval-checkpoint-every 1 --checkpoint-selection min_action_accuracy --device auto --out-dir post_training/runs/qwen_boundary_preference_candidate_ce_phase_batch_a8_eval4_steps8_every1
python post_training/run_boundary_preference_candidate_ce_smoke.py --dry-run --training-schedule boundary_phase_batch --phase-order defer,flag,reject --selection-seed 11 --limit 0 --limit-per-action 8 --eval-limit 0 --eval-limit-per-action 4 --max-length 768 --out-dir post_training/runs/qwen_boundary_preference_candidate_ce_phase_batch_seed11_dry
python post_training/run_boundary_preference_candidate_ce_smoke.py --preferences post_training/negbiodb_ct_oracle_boundary_preferences_hard_train_v1.jsonl --eval-preferences post_training/negbiodb_ct_oracle_boundary_preferences_hard_heldout_v1.jsonl --training-schedule boundary_phase_batch --phase-order defer,flag,reject --selection-seed 11 --limit 0 --limit-per-action 8 --eval-limit 0 --eval-limit-per-action 4 --max-steps 8 --batch-size 1 --candidate-batch-size 1 --max-length 768 --train-last-layers 2 --lr 5e-5 --temperature 1.0 --logprob-mode mean --eval-checkpoint-every 1 --checkpoint-selection min_action_accuracy --device auto --out-dir post_training/runs/qwen_boundary_preference_candidate_ce_phase_batch_seed11_a8_eval4_steps8_every1
python post_training/run_boundary_preference_candidate_ce_smoke.py --dry-run --training-schedule boundary_phase_batch --phase-order defer,flag,reject --selection-seed 11 --action-loss-weights flag=2.0 --limit 0 --limit-per-action 8 --eval-limit 0 --eval-limit-per-action 4 --max-length 768 --out-dir post_training/runs/qwen_boundary_preference_candidate_ce_phase_batch_flagw2_seed11_dry
python post_training/run_boundary_preference_candidate_ce_smoke.py --preferences post_training/negbiodb_ct_oracle_boundary_preferences_hard_train_v1.jsonl --eval-preferences post_training/negbiodb_ct_oracle_boundary_preferences_hard_heldout_v1.jsonl --training-schedule boundary_phase_batch --phase-order defer,flag,reject --selection-seed 11 --action-loss-weights flag=2.0 --limit 0 --limit-per-action 8 --eval-limit 0 --eval-limit-per-action 4 --max-steps 8 --batch-size 1 --candidate-batch-size 1 --max-length 768 --train-last-layers 2 --lr 5e-5 --temperature 1.0 --logprob-mode mean --eval-checkpoint-every 1 --checkpoint-selection min_action_accuracy --device auto --out-dir post_training/runs/qwen_boundary_preference_candidate_ce_phase_batch_flagw2_seed11_a8_eval4_steps8_every1
python post_training/run_boundary_preference_candidate_ce_smoke.py --preferences post_training/negbiodb_ct_oracle_boundary_preferences_hard_train_v1.jsonl --eval-preferences post_training/negbiodb_ct_oracle_boundary_preferences_hard_heldout_v1.jsonl --training-schedule boundary_phase_batch --phase-order defer,flag,reject --selection-seed 11 --action-loss-weights flag=1.25 --limit 0 --limit-per-action 8 --eval-limit 0 --eval-limit-per-action 4 --max-steps 8 --batch-size 1 --candidate-batch-size 1 --max-length 768 --train-last-layers 2 --lr 5e-5 --temperature 1.0 --logprob-mode mean --eval-checkpoint-every 1 --checkpoint-selection min_action_accuracy --device auto --out-dir post_training/runs/qwen_boundary_preference_candidate_ce_phase_batch_flagw125_seed11_a8_eval4_steps8_every1
python post_training/run_boundary_preference_candidate_ce_smoke.py --dry-run --training-schedule boundary_phase_batch --phase-order defer,flag,reject --selection-seed 11 --action-loss-weights flag=1.25,reject=1.25 --skip-train-eval --skip-pre-eval --limit 0 --limit-per-action 8 --eval-limit 0 --eval-limit-per-action 4 --max-length 768 --out-dir post_training/runs/qwen_boundary_preference_candidate_ce_phase_batch_flagw125_rejectw125_seed11_light_dry
python post_training/run_boundary_preference_candidate_ce_smoke.py --preferences post_training/negbiodb_ct_oracle_boundary_preferences_hard_train_v1.jsonl --eval-preferences post_training/negbiodb_ct_oracle_boundary_preferences_hard_heldout_v1.jsonl --training-schedule boundary_phase_batch --phase-order defer,flag,reject --selection-seed 11 --action-loss-weights flag=1.25,reject=1.25 --skip-train-eval --skip-pre-eval --limit 0 --limit-per-action 8 --eval-limit 0 --eval-limit-per-action 4 --max-steps 6 --batch-size 1 --candidate-batch-size 1 --max-length 768 --train-last-layers 2 --lr 5e-5 --temperature 1.0 --logprob-mode mean --eval-checkpoint-every 2 --checkpoint-selection min_action_accuracy --device auto --no-save-trainable-state --out-dir post_training/runs/qwen_boundary_preference_candidate_ce_phase_batch_flagw125_rejectw125_seed11_light_steps6_every2
python post_training/run_boundary_preference_candidate_ce_smoke.py --dry-run --training-schedule boundary_phase_batch --phase-order defer,flag,reject --selection-seed 11 --loss-target action --skip-train-eval --skip-pre-eval --limit 0 --limit-per-action 8 --eval-limit 0 --eval-limit-per-action 4 --max-length 768 --out-dir post_training/runs/qwen_boundary_preference_action_ce_phase_batch_seed11_light_dry
python post_training/run_boundary_preference_candidate_ce_smoke.py --preferences post_training/negbiodb_ct_oracle_boundary_preferences_hard_train_v1.jsonl --eval-preferences post_training/negbiodb_ct_oracle_boundary_preferences_hard_heldout_v1.jsonl --training-schedule boundary_phase_batch --phase-order defer,flag,reject --selection-seed 11 --loss-target action --skip-train-eval --skip-pre-eval --limit 0 --limit-per-action 8 --eval-limit 0 --eval-limit-per-action 4 --max-steps 6 --batch-size 1 --candidate-batch-size 1 --max-length 768 --train-last-layers 2 --lr 5e-5 --temperature 1.0 --logprob-mode mean --eval-checkpoint-every 2 --checkpoint-selection min_action_accuracy --device auto --no-save-trainable-state --out-dir post_training/runs/qwen_boundary_preference_action_ce_phase_batch_seed11_light_steps6_every2
python post_training/run_boundary_preference_candidate_ce_smoke.py --dry-run --training-schedule boundary_phase_batch --phase-order defer,flag,reject --selection-seed 11 --checkpoint-selection action_floor --checkpoint-action-floors flag=0.25,reject=0.25 --skip-train-eval --skip-pre-eval --limit 0 --limit-per-action 8 --eval-limit 0 --eval-limit-per-action 4 --max-length 768 --out-dir post_training/runs/qwen_boundary_preference_candidate_ce_action_floor_seed11_dry
python post_training/run_boundary_preference_candidate_ce_smoke.py --preferences post_training/negbiodb_ct_oracle_boundary_preferences_hard_train_v1.jsonl --eval-preferences post_training/negbiodb_ct_oracle_boundary_preferences_hard_heldout_v1.jsonl --training-schedule boundary_phase_batch --phase-order defer,flag,reject --selection-seed 11 --checkpoint-selection action_floor --checkpoint-action-floors flag=0.25,reject=0.25 --skip-train-eval --skip-pre-eval --limit 0 --limit-per-action 8 --eval-limit 0 --eval-limit-per-action 4 --max-steps 8 --batch-size 1 --candidate-batch-size 1 --max-length 768 --train-last-layers 2 --lr 5e-5 --temperature 1.0 --logprob-mode mean --eval-checkpoint-every 1 --device auto --no-save-trainable-state --out-dir post_training/runs/qwen_boundary_preference_candidate_ce_action_floor_seed11_a8_eval4_steps8_every1
python post_training/run_boundary_preference_candidate_ce_smoke.py --dry-run --training-schedule boundary_phase_batch --phase-order defer,flag,reject --selection-seed 11 --checkpoint-selection action_floor --checkpoint-action-floors flag=0.25,reject=0.25 --action-margin-penalties 'flag>reject=0.25' --action-margin-weight 1.0 --skip-train-eval --skip-pre-eval --limit 0 --limit-per-action 8 --eval-limit 0 --eval-limit-per-action 4 --max-length 768 --out-dir post_training/runs/qwen_boundary_preference_candidate_ce_flag_reject_margin_seed11_dry
python post_training/run_boundary_preference_candidate_ce_smoke.py --preferences post_training/negbiodb_ct_oracle_boundary_preferences_hard_train_v1.jsonl --eval-preferences post_training/negbiodb_ct_oracle_boundary_preferences_hard_heldout_v1.jsonl --training-schedule boundary_phase_batch --phase-order defer,flag,reject --selection-seed 11 --checkpoint-selection action_floor --checkpoint-action-floors flag=0.25,reject=0.25 --action-margin-penalties 'flag>reject=0.25' --action-margin-weight 1.0 --skip-train-eval --skip-pre-eval --limit 0 --limit-per-action 8 --eval-limit 0 --eval-limit-per-action 4 --max-steps 8 --batch-size 1 --candidate-batch-size 1 --max-length 768 --train-last-layers 2 --lr 5e-5 --temperature 1.0 --logprob-mode mean --eval-checkpoint-every 2 --device auto --no-save-trainable-state --out-dir post_training/runs/qwen_boundary_preference_candidate_ce_flag_reject_margin_seed11_a8_eval4_steps8_every2
python post_training/run_boundary_preference_candidate_ce_smoke.py --dry-run --training-schedule boundary_phase_batch --phase-order defer,flag,reject --selection-seed 11 --checkpoint-selection action_floor --checkpoint-action-floors flag=0.25,reject=0.25 --action-margin-penalties 'flag>reject=0.25,reject>flag=0.25' --action-margin-weight 1.0 --skip-train-eval --skip-pre-eval --limit 0 --limit-per-action 8 --eval-limit 0 --eval-limit-per-action 4 --max-length 768 --out-dir post_training/runs/qwen_boundary_preference_candidate_ce_two_sided_margin_seed11_dry
python post_training/run_boundary_preference_candidate_ce_smoke.py --preferences post_training/negbiodb_ct_oracle_boundary_preferences_hard_train_v1.jsonl --eval-preferences post_training/negbiodb_ct_oracle_boundary_preferences_hard_heldout_v1.jsonl --training-schedule boundary_phase_batch --phase-order defer,flag,reject --selection-seed 11 --checkpoint-selection action_floor --checkpoint-action-floors flag=0.25,reject=0.25 --action-margin-penalties 'flag>reject=0.25,reject>flag=0.25' --action-margin-weight 1.0 --skip-train-eval --skip-pre-eval --limit 0 --limit-per-action 8 --eval-limit 0 --eval-limit-per-action 4 --max-steps 8 --batch-size 1 --candidate-batch-size 1 --max-length 768 --train-last-layers 2 --lr 5e-5 --temperature 1.0 --logprob-mode mean --eval-checkpoint-every 2 --device auto --no-save-trainable-state --out-dir post_training/runs/qwen_boundary_preference_candidate_ce_two_sided_margin_seed11_a8_eval4_steps8_every2
```

Current result:

```text
157 passed
good: score=1.000 passed=True
bad: score=0.167 passed=False
adapter oracle_mean_score=1.000
self-contained model-output demo score=1.000
self_answer_no_tool mean_score=0.258 passed=0/400
constant_defer_full_loop mean_score=0.683 passed=120/400
existing DB deterministic-policy action-accuracy=1.000
existing G2 surface-confound verdict=PASS
runner dry-run generic_mean_score=0.267
post_training export:
  sft_examples = 40
  preference_pairs = 64
  failure_modes = missing_attribution 16, mixed_endpoint_over_grounding 8, self_answering_without_tools 40
  oracle_sft_examples = 400
  oracle_sft_by_class = defer 120, flag 40, ground 140, reject 40, verify 60
  validation issues = []
prompt_only baseline on same 40 packets:
  action_accuracy = 0.150
  mean_reward = 0.150
  native_profile_mean_score = 0.450
  tool_call_rate = 0.000
  by_class = defer 6/8, flag 0/8, ground 0/8, reject 0/8, verify 0/8
  top bucket = conservative_defer_wrong_without_tools 25
open_model prompt-only baseline on same 40 packets:
  model = Qwen/Qwen2.5-0.5B-Instruct
  action_accuracy = 0.200
  mean_reward = 0.200
  native_profile_mean_score = 0.467
  tool_call_rate = 0.000
  by_class = verify 8/8, defer 0/8, flag 0/8, ground 0/8, reject 0/8
  top buckets = missed_positive_or_invalid_evidence 16, oververify_without_evidence 8, missed_mixed_endpoint_contradiction 8
SFT smoke loop:
  model = Qwen/Qwen2.5-0.5B-Instruct
  device = mps
  examples = 2
  max_steps = 1
  train_last_layers = 1
  trainable_params = 14913280
  loss = 4.4349
  trainable_state = post_training/runs/qwen_sft_smoke/trainable_state.pt
SFT teacher-forced loss comparison on same 40 tracked examples:
  base_loss = 2.4869
  loaded_loss = 2.4281
  loss_delta = -0.0589
  target_tokens = 536
  loaded_tensors = 13
SFT decision eval on same 40 tracked examples:
  loaded_tensors = 13
  unexpected_tensors = []
  action_accuracy = 0.000
  mean_reward = 0.000
  generic_mean_score = 0.433
  tool_call_rate = 1.000
  parse_failures = 40
SFT constrained decision eval on same 40 tracked examples:
  base_action_accuracy = 0.275
  loaded_action_accuracy = 0.275
  changed_rows = 0
  parse_failures = 0
  by_class = ground 8/8, verify 3/8, defer 0/8, flag 0/8, reject 0/8
Longer same-40 schema/action SFT diagnostic:
  train_last_layers = 2
  trainable_params = 29825664
  first_loss = 2.2059
  last_loss = 0.0293
  teacher_forced_loaded_loss = 0.0546
  strict_generation_action_accuracy = 0.725
  strict_generation_parse_failures = 0
  constrained_action_accuracy = 0.725
  generic_mean_score = 0.908
  by_class = defer 4/8, flag 3/8, ground 6/8, reject 8/8, verify 8/8
SFT split:
  seed = 20260626
  train_examples = 30
  heldout_examples = 10
  train_by_class = defer 6, flag 6, ground 6, reject 6, verify 6
  heldout_by_class = defer 2, flag 2, ground 2, reject 2, verify 2
Train/held-out schema/action SFT diagnostic:
  train_teacher_forced_loaded_loss = 0.0422
  heldout_teacher_forced_loaded_loss = 0.0857
  heldout_strict_generation_action_accuracy = 0.700
  heldout_strict_generation_parse_failures = 0
  heldout_constrained_base_action_accuracy = 0.300
  heldout_constrained_loaded_action_accuracy = 0.500
  heldout_strict_by_class = defer 1/2, flag 1/2, ground 2/2, reject 2/2, verify 1/2
SFT cross-validation split:
  folds = 4
  seed = 20260626
  source_examples = 40
  each_fold_train_examples = 30
  each_fold_heldout_examples = 10
  each_fold_heldout_by_class = defer 2, flag 2, ground 2, reject 2, verify 2
  heldout_coverage_unique_examples = 40
  heldout_coverage_min_count = 1
  heldout_coverage_max_count = 1
Larger deterministic-oracle SFT artifact:
  dataset = negbiodb_ct_oracle_sft_v1
  source_runner = deterministic_oracle_policy
  sft_examples = 400
  by_class = defer 120, flag 40, ground 140, reject 40, verify 60
  skipped = []
  boundary = Deterministic oracle-policy SFT data; not live runner behavior.
CV sweep runner smoke:
  fold = 1
  max_steps = 1
  eval_mode = loss_only
  train_first_loss = 2.2059
  train_teacher_forced_loaded_loss = 1.2253
  heldout_teacher_forced_loaded_loss = 1.2144
Oracle warm-start runner smoke:
  train_limit = 2
  max_steps = 1
  eval = fold0_heldout
  eval_mode = loss_only
  train_first_loss = 1.6071
  fold0_heldout_teacher_forced_loaded_loss = 1.3513
Full CV/oracle SFT execution:
  result_anchor = post_training/SFT_SWEEP_RESULTS_2026-06-26.md
  failure_analysis_anchor = post_training/SFT_FAILURE_ANALYSIS_2026-06-26.md
  native_cv_strict_action_accuracy_mean = 0.475
  native_cv_strict_action_accuracy_range = 0.300..0.700
  native_cv_constrained_base_accuracy_mean = 0.275
  native_cv_constrained_loaded_accuracy_mean = 0.400
  native_cv_parse_failures_total = 0
  oracle400_strict_action_accuracy_mean = 0.450
  oracle400_strict_action_accuracy_range = 0.400..0.500
  oracle400_constrained_base_accuracy_mean = 0.275
  oracle400_constrained_loaded_accuracy_mean = 0.400
  oracle400_parse_failures_total = 0
Row-level SFT failure analysis:
  native_cv_strict_class_accuracy = defer 3/8, flag 3/8, ground 6/8, reject 6/8, verify 1/8
  native_cv_constrained_class_accuracy = defer 3/8, flag 3/8, ground 6/8, reject 4/8, verify 0/8
  oracle400_strict_class_accuracy = defer 8/8, flag 0/8, ground 8/8, reject 2/8, verify 0/8
  oracle400_constrained_class_accuracy = defer 8/8, flag 0/8, ground 8/8, reject 0/8, verify 0/8
  recurrent_failure_count = 16
Pressure SFT artifacts:
  pressure_anchor = post_training/SFT_PRESSURE_ARTIFACTS_2026-06-26.md
  native_pressure_train_examples_per_fold = 54
  native_pressure_train_by_class = defer 6, flag 18, ground 6, reject 6, verify 18
  oracle_balanced_examples = 700
  oracle_balanced_by_class = defer 140, flag 140, ground 140, reject 140, verify 140
Full pressure rerun:
  pressure_result_anchor = post_training/SFT_PRESSURE_RUN_RESULTS_2026-06-26.md
  pressure_result_json = post_training/sft_pressure_run_summary_2026-06-26.json
  native_pressure_strict_action_accuracy_mean = 0.400
  native_pressure_constrained_loaded_accuracy_mean = 0.450
  native_pressure_strict_parse_failures_total = 0
  native_pressure_strict_class_accuracy = defer 0/8, flag 4/8, ground 0/8, reject 4/8, verify 8/8
  oracle_balanced_strict_action_accuracy_mean = 0.200
  oracle_balanced_constrained_loaded_accuracy_mean = 0.200
  oracle_balanced_strict_parse_failures_total = 12
  oracle_balanced_strict_class_accuracy = defer 0/8, flag 0/8, ground 8/8, reject 0/8, verify 0/8
Pressure failure analysis and curriculum artifact:
  pressure_failure_anchor = post_training/SFT_PRESSURE_FAILURE_ANALYSIS_2026-06-26.md
  pressure_failure_json = post_training/sft_pressure_failure_analysis_2026-06-26.json
  pressure_recurrent_failure_count = 17
  curriculum_anchor = post_training/SFT_CURRICULUM_ARTIFACTS_2026-06-26.md
  curriculum_manifest = post_training/negbiodb_ct_native_sft_cv4_curriculum_manifest.json
  curriculum_train_examples_per_fold = 72
  curriculum_train_by_class = defer 12, flag 18, ground 18, reject 12, verify 12
  curriculum_train_by_family = base 30, ground_flag 12, reject_override 18, verify_defer 12
Full curriculum rerun:
  curriculum_result_anchor = post_training/SFT_CURRICULUM_RUN_RESULTS_2026-06-26.md
  curriculum_result_json = post_training/sft_curriculum_run_summary_2026-06-26.json
  native_curriculum_strict_action_accuracy_mean = 0.475
  native_curriculum_strict_action_accuracy_range = 0.300..0.700
  native_curriculum_constrained_base_accuracy_mean = 0.275
  native_curriculum_constrained_loaded_accuracy_mean = 0.425
  native_curriculum_constrained_loaded_accuracy_range = 0.300..0.600
  native_curriculum_strict_parse_failures_total = 0
  native_curriculum_strict_class_accuracy = defer 3/8, flag 6/8, ground 3/8, reject 4/8, verify 3/8
  native_curriculum_constrained_loaded_class_accuracy = defer 3/8, flag 7/8, ground 2/8, reject 2/8, verify 3/8
Curriculum failure analysis:
  curriculum_failure_anchor = post_training/SFT_CURRICULUM_FAILURE_ANALYSIS_2026-06-26.md
  curriculum_failure_json = post_training/sft_curriculum_failure_analysis_2026-06-26.json
  curriculum_persistent_failure_count = 20
  curriculum_strict_failure_pair_counts = defer->verify 5, flag->reject 2, ground->flag 4, ground->reject 1, reject->flag 4, verify->defer 5
  curriculum_constrained_failure_pair_counts = defer->verify 5, flag->reject 1, ground->flag 5, ground->reject 1, reject->flag 6, verify->defer 5
Targeted curriculum-v2:
  curriculum_v2_manifest = post_training/negbiodb_ct_native_sft_cv4_curriculum_v2_manifest.json
  curriculum_v2_result_anchor = post_training/SFT_CURRICULUM_V2_RUN_RESULTS_2026-06-26.md
  curriculum_v2_result_json = post_training/sft_curriculum_v2_run_summary_2026-06-26.json
  curriculum_v2_train_examples_per_fold = 102, 109, 103, 103
  curriculum_v2_targeted_examples_per_fold = 30, 37, 31, 31
  curriculum_v2_strict_action_accuracy_mean = 0.375
  curriculum_v2_strict_action_accuracy_range = 0.200..0.600
  curriculum_v2_constrained_loaded_accuracy_mean = 0.300
  curriculum_v2_constrained_loaded_accuracy_range = 0.200..0.500
  curriculum_v2_strict_parse_failures_total = 0
  curriculum_v2_constrained_loaded_class_accuracy = defer 3/8, flag 4/8, ground 3/8, reject 0/8, verify 2/8
Boundary-rationale SFT artifact:
  boundary_rationale_manifest = post_training/negbiodb_ct_native_sft_cv4_boundary_rationale_manifest.json
  boundary_rationale_anchor = post_training/SFT_BOUNDARY_RATIONALE_ARTIFACTS_2026-06-26.md
  boundary_rationale_result_json = post_training/sft_boundary_rationale_smoke_summary_2026-06-26.json
  boundary_rationale_train_examples_per_fold = 60
  boundary_rationale_train_by_class = defer 12, flag 12, ground 12, reject 12, verify 12
  boundary_rationale_train_by_role = base 30, rationale 30
  boundary_rationale_fold0_strict_action_accuracy = 0.500
  boundary_rationale_fold0_constrained_loaded_action_accuracy = 0.500
  boundary_rationale_fold0_strict_parse_failures = 0
Boundary-rationale full CV:
  boundary_rationale_run_anchor = post_training/SFT_BOUNDARY_RATIONALE_RUN_RESULTS_2026-06-26.md
  boundary_rationale_run_json = post_training/sft_boundary_rationale_run_summary_2026-06-26.json
  boundary_rationale_strict_action_accuracy_mean = 0.500
  boundary_rationale_constrained_loaded_accuracy_mean = 0.500
  boundary_rationale_strict_parse_failures_total = 0
  boundary_rationale_strict_class_accuracy = defer 0/8, flag 3/8, ground 3/8, reject 6/8, verify 8/8
  boundary_rationale_constrained_loaded_class_accuracy = defer 0/8, flag 3/8, ground 4/8, reject 5/8, verify 8/8
Boundary-rationale failure analysis:
  boundary_rationale_failure_anchor = post_training/SFT_BOUNDARY_RATIONALE_FAILURE_ANALYSIS_2026-06-26.md
  boundary_rationale_failure_json = post_training/sft_boundary_rationale_failure_analysis_2026-06-26.json
  defer_failure_count = 8
  all_defer_failures_predicted_verify = True
  all_defer_observations_empty = True
  heldout_defer_prompt_has_boundary_rationale = False
  mean_defer_minus_verify_mean_nll = 0.3479
Held-out oracle-rationale ablation:
  heldout_ablation_anchor = post_training/SFT_BOUNDARY_RATIONALE_HELDOUT_ABLATION_2026-06-27.md
  heldout_ablation_json = post_training/sft_boundary_rationale_heldout_ablation_summary_2026-06-27.json
  heldout_ablation_strict_action_accuracy_mean = 1.000
  heldout_ablation_constrained_loaded_accuracy_mean = 1.000
  heldout_ablation_strict_parse_failures_total = 0
  heldout_ablation_strict_class_accuracy = defer 8/8, flag 8/8, ground 8/8, reject 8/8, verify 8/8
Held-out evidence-rationale ablation:
  heldout_evidence_ablation_anchor = post_training/SFT_BOUNDARY_RATIONALE_HELDOUT_EVIDENCE_ABLATION_2026-06-27.md
  heldout_evidence_ablation_json = post_training/sft_boundary_rationale_heldout_evidence_ablation_summary_2026-06-27.json
  heldout_evidence_ablation_manifest = post_training/negbiodb_ct_native_sft_cv4_boundary_rationale_heldout_evidence_ablation_manifest.json
  heldout_evidence_ablation_strict_action_accuracy_mean = 1.000
  heldout_evidence_ablation_constrained_loaded_accuracy_mean = 1.000
  heldout_evidence_ablation_strict_parse_failures_total = 0
  heldout_evidence_ablation_strict_class_accuracy = defer 8/8, flag 8/8, ground 8/8, reject 8/8, verify 8/8
  heldout_evidence_ablation_action_mismatches = 0, 0, 0, 0
Reusable evidence-rationale layer:
  evidence_rationale_anchor = post_training/EVIDENCE_RATIONALE_LAYER_2026-06-27.md
  evidence_rationale_artifact = post_training/negbiodb_ct_native_sft_evidence_rationale_v1.jsonl
  evidence_rationale_manifest = post_training/negbiodb_ct_native_sft_evidence_rationale_manifest.json
  evidence_rationale_examples = 40
  evidence_rationale_by_action_class = defer 8, flag 8, ground 8, reject 8, verify 8
  evidence_rationale_by_evidence_action = defer 8, flag 8, ground 8, reject 8, verify 8
  evidence_rationale_matches = 40
  evidence_rationale_mismatches = 0
Pilot-400 evidence-rationale stress test:
  evidence_pilot400_anchor = post_training/EVIDENCE_RATIONALE_PILOT400_STRESS_2026-06-27.md
  evidence_pilot400_artifact = post_training/negbiodb_ct_oracle_sft_evidence_rationale_v1.jsonl
  evidence_pilot400_manifest = post_training/negbiodb_ct_oracle_sft_evidence_rationale_manifest.json
  evidence_pilot400_examples = 400
  evidence_pilot400_by_action_class = defer 120, flag 40, ground 140, reject 40, verify 60
  evidence_pilot400_by_evidence_action = defer 120, flag 40, ground 140, reject 40, verify 60
  evidence_pilot400_matches = 400
  evidence_pilot400_mismatches = 0
Evidence-rationale guardrail evaluation:
  evidence_guardrail_anchor = post_training/EVIDENCE_RATIONALE_GUARDRAIL_EVAL_2026-06-27.md
  evidence_guardrail_json = post_training/evidence_rationale_guardrail_eval_2026-06-27.json
  strict_model_action_accuracy = 0.500
  strict_guardrail_action_accuracy = 1.000
  strict_rescued_errors = 20
  strict_introduced_errors = 0
  constrained_model_action_accuracy = 0.500
  constrained_guardrail_action_accuracy = 1.000
  constrained_rescued_errors = 20
  constrained_introduced_errors = 0
Evidence-rationale distillation fold0 diagnostic:
  evidence_distill_anchor = post_training/SFT_EVIDENCE_DISTILL_FOLD0_DIAGNOSTIC_2026-06-27.md
  evidence_distill_json = post_training/sft_evidence_distill_fold0_summary_2026-06-27.json
  train_examples = 400
  train_loss_delta = -1.5084
  strict_action_accuracy = 0.200
  strict_parse_failures = 4
  constrained_loaded_action_accuracy = 0.200
  constrained_loaded_class_accuracy = defer 0/2, flag 0/2, ground 2/2, reject 0/2, verify 0/2
Evidence-rationale prompted-rule upper-bound:
  evidence_prompted_upper_bound_anchor = post_training/SFT_EVIDENCE_PROMPTED_UPPER_BOUND_FOLD0_2026-06-27.md
  evidence_prompted_upper_bound_json = post_training/sft_evidence_prompted_upper_bound_fold0_summary_2026-06-27.json
  same_checkpoint = post_training/runs/qwen_oracle400_evidence_distill_base_cvheldout/train/trainable_state.pt
  base_prompt_strict_action_accuracy = 0.200
  prompted_strict_action_accuracy = 1.000
  base_prompt_constrained_loaded_action_accuracy = 0.200
  prompted_constrained_loaded_action_accuracy = 1.000
  prompted_class_accuracy = defer 2/2, flag 2/2, ground 2/2, reject 2/2, verify 2/2
Generative evidence-rationale artifact:
  generative_rationale_anchor = post_training/SFT_GENERATIVE_RATIONALE_ARTIFACTS_2026-06-27.md
  generative_rationale_json = post_training/sft_generative_rationale_smoke_summary_2026-06-27.json
  train_examples = 400
  train_matches = 400
  fold0_examples = 10
  fold0_matches = 10
  oracle_encode_dry_length_range = 422..751
  fold0_encode_dry_length_range = 424..570
  smoke_loss = 2.3865
Generative evidence-rationale fold0 diagnostic:
  generative_rationale_diagnostic_anchor = post_training/SFT_GENERATIVE_RATIONALE_FOLD0_DIAGNOSTIC_2026-06-27.md
  generative_rationale_diagnostic_json = post_training/sft_generative_rationale_fold0_diagnostic_2026-06-27.json
  train_loss_delta = -2.3574
  generative_fold0_strict_action_accuracy = 0.300
  generative_fold0_strict_parse_failures = 0
  native_base_fold0_strict_action_accuracy = 0.000
  native_base_fold0_strict_parse_failures = 10
  native_base_fold0_constrained_loaded_action_accuracy = 0.400
  native_base_fold0_constrained_loaded_class_accuracy = defer 0/2, flag 0/2, ground 2/2, reject 0/2, verify 2/2
Evidence-boundary preference artifact:
  boundary_preference_anchor = post_training/BOUNDARY_PREFERENCE_ARTIFACTS_2026-06-27.md
  boundary_preference_artifact = post_training/negbiodb_ct_oracle_boundary_preferences_v1.jsonl
  boundary_preference_manifest = post_training/negbiodb_ct_oracle_boundary_preferences_manifest.json
  source_examples = 400
  preference_pairs = 620
  chosen_passed = 620
  rejected_passed = 0
  boundary_preference_failure_modes = boundary_defer_over_verify 120, boundary_verify_over_defer 60, boundary_ground_over_flag 140, boundary_ground_over_reject 140, boundary_flag_over_ground 40, boundary_flag_over_reject 40, boundary_reject_over_ground 40, boundary_reject_over_flag 40
Boundary preference base-margin diagnostic:
  boundary_preference_margin_anchor = post_training/BOUNDARY_PREFERENCE_MARGIN_BASE_2026-06-27.md
  boundary_preference_margin_json = post_training/boundary_preference_margin_base_summary_2026-06-27.json
  boundary_preference_margin_raw = post_training/runs/qwen_boundary_preference_margin/base_full.json
  model = Qwen/Qwen2.5-0.5B-Instruct
  n = 620
  mean_win_rate = 0.615
  sum_win_rate = 0.453
  hard_negative_modes = boundary_defer_over_verify 0.008/-0.1125, boundary_flag_over_ground 0.000/-0.1656, boundary_reject_over_ground 0.000/-2.6783, boundary_reject_over_flag 0.000/-2.5255
  easy_or_aligned_modes = boundary_verify_over_defer 1.000/0.1042, boundary_ground_over_flag 1.000/0.1784, boundary_ground_over_reject 1.000/2.6621, boundary_flag_over_reject 1.000/2.4932
Boundary preference hard-mode subset and DPO-style smoke:
  hard_mode_anchor = post_training/BOUNDARY_PREFERENCE_HARD_MODE_ARTIFACTS_2026-06-27.md
  hard_mode_json = post_training/boundary_preference_hard_mode_summary_2026-06-27.json
  hard_mode_pairs = 240
  hard_mode_failure_modes = boundary_defer_over_verify 120, boundary_flag_over_ground 40, boundary_reject_over_ground 40, boundary_reject_over_flag 40
  dpo_smoke_condition = reference_free_dpo_style_limit8_overfit
  dpo_smoke_selected_pairs = 8
  dpo_smoke_pre_win_rate = 0.000
  dpo_smoke_pre_mean_margin = -1.4004
  dpo_smoke_post_win_rate = 1.000
  dpo_smoke_post_mean_margin = 6.9414
  dpo_smoke_loss_delta = -0.2598
Boundary preference hard split and heldout-aware DPO-style smoke:
  hard_split_anchor = post_training/BOUNDARY_PREFERENCE_HARD_SPLIT_DPO_2026-06-27.md
  hard_split_json = post_training/boundary_preference_hard_split_dpo_summary_2026-06-27.json
  hard_split_train_pairs = 208
  hard_split_heldout_pairs = 32
  hard_split_overlap_source_ids = 0
  dpo_split_selected_pairs = 208
  dpo_split_eval_pairs = 32
  dpo_split_pre_eval_win_rate = 0.000
  dpo_split_pre_eval_mean_margin = -1.3701
  dpo_split_post_eval_win_rate = 1.000
  dpo_split_post_eval_mean_margin = 39.5352
  dpo_split_loss_delta = -0.6113
Boundary preference all-candidate final-decision eval:
  candidate_eval_anchor = post_training/BOUNDARY_PREFERENCE_CANDIDATE_EVAL_2026-06-28.md
  candidate_eval_json = post_training/boundary_preference_candidate_eval_summary_2026-06-28.json
  base_action_accuracy = 0.000
  base_exact_candidate_accuracy = 0.000
  dpo_loaded_action_accuracy = 0.250
  dpo_loaded_exact_candidate_accuracy = 0.250
  dpo_loaded_defer_over_verify = 8/8
  dpo_loaded_flag_over_ground = 0/8, pred defer 8
  dpo_loaded_reject_over_flag = 0/8, pred defer 8, expected rank 2 for 8/8
  dpo_loaded_reject_over_ground = 0/8, pred defer 8, expected rank 2 for 8/8
Boundary preference all-candidate CE smoke:
  candidate_ce_anchor = post_training/BOUNDARY_PREFERENCE_CANDIDATE_CE_SMOKE_2026-06-28.md
  candidate_ce_json = post_training/boundary_preference_candidate_ce_smoke_summary_2026-06-28.json
  balanced_train_sets = 24
  balanced_eval_sets = 12
  balanced_train_by_expected_action = defer 8, flag 8, reject 8
  balanced_eval_by_expected_action = defer 4, flag 4, reject 4
  source_order_post_eval_action_accuracy = 0.333
  source_order_post_eval_defer = 0/4, pred reject 4
  source_order_post_eval_flag = 4/4, pred flag 4
  source_order_post_eval_reject = 0/4, pred flag 4
  round_robin_post_eval_action_accuracy = 0.583
  round_robin_post_eval_defer = 3/4, pred defer 3, reject 1
  round_robin_post_eval_flag = 0/4, pred reject 4
  round_robin_post_eval_reject = 4/4, pred reject 4
  flag_rehearsal_24_post_eval_action_accuracy = 0.333
  flag_rehearsal_24_post_eval_defer = 0/4, pred reject 4
  flag_rehearsal_24_post_eval_flag = 4/4, pred flag 4
  flag_rehearsal_24_post_eval_reject = 0/4, pred flag 4
  flag_rehearsal_stop23_post_eval_action_accuracy = 0.417
  flag_rehearsal_stop23_post_eval_defer = 1/4, pred defer 1, reject 3
  flag_rehearsal_stop23_post_eval_flag = 4/4, pred flag 4
  flag_rehearsal_stop23_post_eval_reject = 0/4, pred flag 4
  eval_gated_plumbing_selected_checkpoint = step 6, phase reject
  eval_gated_plumbing_post_evaluation_state = selected_checkpoint
  eval_gated_plumbing_post_eval_action_accuracy = 0.667 on tiny 2/action slice
  eval_gated_plumbing_post_eval_defer = 2/2
  eval_gated_plumbing_post_eval_flag = 2/2
  eval_gated_plumbing_post_eval_reject = 0/2, pred flag 2
  eval_gated_a8_eval4_selected_checkpoint = step 15, phase reject
  eval_gated_a8_eval4_post_eval_action_accuracy = 0.667
  eval_gated_a8_eval4_post_eval_defer = 4/4
  eval_gated_a8_eval4_post_eval_flag = 0/4, pred defer 1, reject 3
  eval_gated_a8_eval4_post_eval_reject = 4/4
  phase_batch_plumbing_post_eval_action_accuracy = 0.333 on tiny 2/action slice
  phase_batch_plumbing_each_loss_step_phase_counts = defer 1, flag 1, reject 1
  phase_batch_plumbing_post_eval_defer = 2/2
  phase_batch_plumbing_post_eval_flag = 0/2, pred defer 2
  phase_batch_plumbing_post_eval_reject = 0/2, pred defer 2
  phase_batch_a8_eval4_selected_checkpoint = step 7, phase phase_batch:defer+flag+reject
  phase_batch_a8_eval4_post_eval_action_accuracy = 0.917
  phase_batch_a8_eval4_post_eval_defer = 4/4
  phase_batch_a8_eval4_post_eval_flag = 4/4
  phase_batch_a8_eval4_post_eval_reject = 3/4, pred flag 1, reject 3
  phase_batch_a8_eval4_step8_regression = action_accuracy 0.667, reject 0/4
  phase_batch_seed11_selected_checkpoint = step 8, phase phase_batch:defer+flag+reject
  phase_batch_seed11_post_eval_action_accuracy = 0.750
  phase_batch_seed11_post_eval_defer = 4/4
  phase_batch_seed11_post_eval_flag = 1/4, pred defer 1, flag 1, reject 2
  phase_batch_seed11_post_eval_reject = 4/4
  phase_batch_flagw125_seed11_selected_checkpoint = step 7, phase phase_batch:defer+flag+reject
  phase_batch_flagw125_seed11_action_loss_weights = flag 1.25
  phase_batch_flagw125_seed11_post_eval_action_accuracy = 0.667
  phase_batch_flagw125_seed11_post_eval_defer = 4/4
  phase_batch_flagw125_seed11_post_eval_flag = 4/4
  phase_batch_flagw125_seed11_post_eval_reject = 0/4, pred flag 4
  phase_batch_flagw2_seed11_selected_checkpoint = step 2, phase phase_batch:defer+flag+reject
  phase_batch_flagw2_seed11_action_loss_weights = flag 2.0
  phase_batch_flagw2_seed11_post_eval_action_accuracy = 0.667
  phase_batch_flagw2_seed11_post_eval_defer = 4/4
  phase_batch_flagw2_seed11_post_eval_flag = 4/4
  phase_batch_flagw2_seed11_post_eval_reject = 0/4, pred flag 4
  phase_batch_flagw125_rejectw125_seed11_light_selected_checkpoint = step 6, phase phase_batch:defer+flag+reject
  phase_batch_flagw125_rejectw125_seed11_light_action_loss_weights = flag 1.25, reject 1.25
  phase_batch_flagw125_rejectw125_seed11_light_skip_train_eval = true
  phase_batch_flagw125_rejectw125_seed11_light_post_eval_action_accuracy = 0.500
  phase_batch_flagw125_rejectw125_seed11_light_post_eval_defer = 2/4, pred defer 2, reject 2
  phase_batch_flagw125_rejectw125_seed11_light_post_eval_flag = 0/4, pred reject 4
  phase_batch_flagw125_rejectw125_seed11_light_post_eval_reject = 4/4
  phase_batch_action_ce_seed11_light_selected_checkpoint = step 6, phase phase_batch:defer+flag+reject
  phase_batch_action_ce_seed11_light_loss_target = action
  phase_batch_action_ce_seed11_light_skip_train_eval = true
  phase_batch_action_ce_seed11_light_post_eval_action_accuracy = 0.667
  phase_batch_action_ce_seed11_light_post_eval_defer = 4/4
  phase_batch_action_ce_seed11_light_post_eval_flag = 4/4
  phase_batch_action_ce_seed11_light_post_eval_reject = 0/4, pred flag 4
  phase_batch_action_floor_seed11_selected_checkpoint = step 8, phase phase_batch:defer+flag+reject
  phase_batch_action_floor_seed11_checkpoint_selection = action_floor
  phase_batch_action_floor_seed11_checkpoint_action_floors = flag 0.25, reject 0.25
  phase_batch_action_floor_seed11_floor_satisfied = true
  phase_batch_action_floor_seed11_post_eval_action_accuracy = 0.750
  phase_batch_action_floor_seed11_post_eval_defer = 4/4
  phase_batch_action_floor_seed11_post_eval_flag = 1/4, pred defer 1, flag 1, reject 2
  phase_batch_action_floor_seed11_post_eval_reject = 4/4
  phase_batch_flag_reject_margin_seed11_selected_checkpoint = step 8, phase phase_batch:defer+flag+reject
  phase_batch_flag_reject_margin_seed11_action_margin_penalties = flag>reject 0.25
  phase_batch_flag_reject_margin_seed11_checkpoint_selection = action_floor
  phase_batch_flag_reject_margin_seed11_floor_satisfied = false
  phase_batch_flag_reject_margin_seed11_post_eval_action_accuracy = 0.667
  phase_batch_flag_reject_margin_seed11_post_eval_defer = 4/4
  phase_batch_flag_reject_margin_seed11_post_eval_flag = 4/4
  phase_batch_flag_reject_margin_seed11_post_eval_reject = 0/4, pred flag 4
  phase_batch_two_sided_margin_seed11_selected_checkpoint = step 4, phase phase_batch:defer+flag+reject
  phase_batch_two_sided_margin_seed11_action_margin_penalties = flag>reject 0.25, reject>flag 0.25
  phase_batch_two_sided_margin_seed11_checkpoint_selection = action_floor
  phase_batch_two_sided_margin_seed11_floor_satisfied = false
  phase_batch_two_sided_margin_seed11_post_eval_action_accuracy = 0.583
  phase_batch_two_sided_margin_seed11_post_eval_defer = 4/4
  phase_batch_two_sided_margin_seed11_post_eval_flag = 0/4, pred defer 3, reject 1
  phase_batch_two_sided_margin_seed11_post_eval_reject = 3/4, pred defer 1, reject 3
```

Note: existing ignored `agent_sonnet*.json` files may be stale if they were
created before the current `tasks_pilot.jsonl` was regenerated under
efficacy-failure semantics. Re-run the runner before using them as benchmark
evidence.

Current stale-artifact sanity check:

```text
agent_sonnet_n40.json matched_rows=32 stale_or_missing_packet_ids=8
full_profile_mean_score=0.802
native_profile_mean_score=0.969
```

Current n=40b artifact sanity check:

```text
matched_rows=40 stale_or_missing_packet_ids=0
full_profile_mean_score=0.775
native_profile_mean_score=0.942
native_failure_buckets:
  prompt_schema_flag_needs_nct: 8
  ground_efficacy_record_not_recognized: 2
```

Interpretation: the native runner should be evaluated with `tool_profile="native_ct"`;
the default `nullatlas_full` profile remains the stricter post-training target
for trajectories that explicitly spell out the full NullAtlas loop.

Runner attribution refinement after the n=40b analysis:

- `flag` now requires and preserves NCT attribution, just like `ground`.
- impossible p-value injection now targets the gold efficacy NCT rather than
  the first returned record.
- the native tool prompt now states that `failure_category='efficacy'` is the
  decisive failure label even when endpoint/p-value fields are null.

The two remaining ground errors in the old n=40b artifact are now interpreted
as an evidence-representation/prompt/SFT target: the model needs to treat an
efficacy-labeled failure row as decisive support even when the auxiliary fields
are sparse.

Current live rerun after flag-attribution and mixed-endpoint prompt fixes:

```text
agent_sonnet_n5_flagfix.json:
  action_accuracy = 1.000
  native_profile_mean_score = 1.000
  flag missing citations = 0

agent_sonnet_n40_mixedfix.json:
  matched_rows = 40
  stale_or_missing_packet_ids = 0
  action_accuracy = 1.000
  mean_reward = 1.000
  native_profile_mean_score = 1.000
  native_failure_buckets = none
  by_class = defer 8/8, flag 8/8, ground 8/8, reject 8/8, verify 8/8
```

The only failure in the intermediate n=40 flagfix run was
`mixed_endpoint_reject_overridden_by_ground` on `ct::reject::913::39750`. The
runner now states that mixed endpoint evidence overrides grounding, and a
targeted live rerun of that packet passed.

## Current Boundary

The adapter, parser, and runner bridge are now built for the existing pilot
JSONL. They do not yet execute the real NullAtlas MCP tools; they convert
already-materialized task records and prompt-style JSON outputs into the shared
evaluator schema.

The existing `negbiodb_ct/baselines.py` is complementary: it directly queries
the CT database for no-API deterministic-policy and surface-confound checks. The
new evaluator is the post-training-facing scorer for trajectory outputs.

## Next Implementation Step

Use the evidence-rationale layer and boundary preference margins as the next
schema-stable learner/evaluator branch:

1. keep the evidence-rationale layer as the current deployable guardrail path,
2. replace single-pair preference pressure with all-legal negatives or
   multi-negative hard prompts before making decision-level improvement claims,
3. add `flag` robustness pressure or a robustness-aware selector before larger
   hard-split all-candidate runs; seeded phase-batch selection avoids
   zero-family collapse but leaves `flag` weak, while naive flag-only loss
   weighting at 1.25 and 2.0 fixes `flag` by collapsing `reject`; joint
   low `flag`/`reject` weighting preserves `reject` but collapses `flag`;
   action-level CE recovers `flag` but again collapses `reject`, so next use an
   explicit `flag`/`reject` floor or margin-aware selector; the first
   `action_floor` selector recovers a nonzero `flag`/`reject` checkpoint but
   still leaves `flag` weak at 1/4; one-sided `flag>reject` margin pressure
   recovers `flag` 4/4 but again collapses `reject` 0/4; two-sided
   `flag>reject` plus `reject>flag` margin pressure partially preserves
   `reject` but leaves `flag` 0/4 and still fails the action floor,
4. compare against direct tool-loop open-model conditions,
5. leave broad DPO/RLVR until the rule-derived boundary layer or preference
   objective shows stable held-out improvement.

That is the bridge from no-API tests to Gate 1:
regex/feature-row baseline, deterministic policy baseline, prompt-only weak open
model, and frontier reference.

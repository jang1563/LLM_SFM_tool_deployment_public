# Post-Training Data Artifacts

This directory stores small, reproducible post-training artifacts generated
from the NegBioDB-CT native runner. The files are intentionally compact and
tracked so later SFT/DPO/RLVR work has a stable reproducibility anchor.

## Stage A Benchmark Artifact

Dataset family: `negbiodb_ct_stage_a_*`

Generated from:

```bash
python3 post_training/export_stage_a_data.py
```

Inputs:

- `negbiodb_ct/stage_a_mini_manifest.jsonl`
- deterministic Stage A oracle/failure generators in
  `negbiodb_ct.stage_a_manifest`

Outputs:

| path | rows | role |
| --- | ---: | --- |
| `stage_a_sft_v1.jsonl` | 25 | Oracle Stage A tool trajectories for supervised trajectory learning. |
| `stage_a_preferences_v1.jsonl` | 150 | Chosen/rejected trajectory pairs across self-answer, wrong/missing tool, partial-query, attribution, invalid-value, unsupported-trust, and insufficient-evidence failures. |
| `stage_a_process_supervision_v1.jsonl` | 25 | Process targets for required tools, required query fields, evidence status, terminal action, attribution, and source IDs. |
| `stage_a_export_manifest.json` | 1 | Export provenance, class/status balance, failure-mode counts, pass/fail direction, and split-group overlap status. |
| `stage_a_sft_train_v1.jsonl` | 20 | Deterministic train split of Stage A oracle SFT trajectories. |
| `stage_a_sft_heldout_v1.jsonl` | 5 | Deterministic held-out split of Stage A oracle SFT trajectories. |
| `stage_a_preferences_train_v1.jsonl` | 120 | Train split of Stage A preference pairs, grouped by source manifest case. |
| `stage_a_preferences_heldout_v1.jsonl` | 30 | Held-out split of Stage A preference pairs, grouped by source manifest case. |
| `stage_a_process_train_v1.jsonl` | 20 | Train split of Stage A process-supervision rows. |
| `stage_a_process_heldout_v1.jsonl` | 5 | Held-out split of Stage A process-supervision rows. |
| `stage_a_split_manifest.json` | 1 | Split seed, held-out family balance, case IDs, split groups, source task IDs, and overlap checks. |
| `build_stage_a_sealed_extension.py` | - | Builds a source-disjoint private evaluation manifest outside the repository and publishes only aggregate balance/overlap counts plus cryptographic commitments. |
| `stage_a_sealed_extension_commitment_2026-07-10.json` | 1 | Public-safe commitment for 25 private sealed rows with balanced family counts, zero declared public overlap, and no row-level labels. |
| `STAGE_A_SEALED_EXTENSION_COMMITMENT_2026-07-10.md` | - | Human-readable sealed-extension readiness checkpoint. |
| `evaluate_stage_a_tool_query_sft_smoke_result.py` | - | Converts private/ignored tool-query predictions into aggregate schema, violation, and prompt-schema behavior counts without emitting raw text. |
| `stage_a_tool_query_sft_smoke_result_qwen05b_cayuga_2026-07-23.json` | 1 | Compact 0/5 tool-query placeholder-schema result; explicitly excludes identifier-resolution and live-tool claims. |
| `build_stage_a_candidate_routing_policy_freeze.py` | - | Freezes model revision, saved-state hash, prompt/candidate policy, training inputs, evaluator, and sealed commitment before one-time evaluation. |
| `stage_a_candidate_routing_policy_freeze_2026-07-23.json` | 1 | Public-safe frozen-policy commitment; private state and report paths are redacted and the original missing seed is disclosed. |
| `run_stage_a_sealed_candidate_routing_eval.py` | - | One-time private evaluator with hash checks, external-path enforcement, lock protection, oracle-evidence scope declaration, and aggregate-only output. |
| `stage_a_sealed_candidate_routing_result_qwen05b_cayuga_2026-07-23.json` | 1 | Compact sealed result: 5/25 exact, 25/25 `verify/insufficient`, static prior 5/25, runtime oracle 25/25. |
| `stage_a_prospective_real_query_tool_query_v1.jsonl` | 25 | Public development tool-query rows with unique case-specific typed identifiers. |
| `stage_a_prospective_real_query_routing_perturbations_v1.jsonl` | 180 | Synthetic routing states spanning eight tool/evidence perturbations. |
| `stage_a_prospective_real_query_experiment_manifest.json` | 1 | Source separation, artifact hashes, balance, and query-contract checks for the prospective substrate. |
| `run_stage_a_prospective_frozen_policy.py` | - | Loads the pre-prospective frozen routing state without retraining and keeps row-level predictions private. |
| `stage_a_prospective_runtime_hybrid_result_qwen05b_cayuga_2026-07-23.json` | 1 | Aggregate comparison: frozen routing 35/180, static prior 80/180, deterministic gate 180/180, runtime hybrid 115/180. |
| `run_stage_a_prospective_tool_query_transfer.py` | - | Compares base and pre-prospective placeholder-SFT policies on 25 real-query prompts without retraining. |
| `stage_a_prospective_tool_query_transfer_result_qwen05b_cayuga_2026-07-23.json` | 1 | Aggregate transfer result: base and frozen SFT both 0/25 exact. |
| `stage_a_prospective_tool_query_prompt_repair_result_qwen05b_cayuga_2026-07-23.json` | 1 | Adaptive prompt-only result: target keys 25/25, strict tool-call shape 0/25. |
| `evaluate_stage_a_tool_query_runtime_compiler.py` | - | Evaluates exact compilation and fail-closed malformed-input behavior. |
| `stage_a_tool_query_runtime_compiler_result_2026-07-23.json` | 1 | Runtime result: 25/25 clean exact and 150/150 malformed inputs rejected for intended reasons. |
| `run_stage_a_sft_smoke_eval.py` | - | No-API Stage A SFT smoke/eval harness for train/held-out trajectory mechanics. |
| `generate_stage_a_predictions.py` | - | Artifact-first producer for saved Stage A prediction JSONL. |
| `evaluate_stage_a_predictions.py` | - | Offline scorer for saved Stage A API, cluster, prompt-only, or oracle prediction JSONL. |
| `run_stage_a_saved_prediction_candidate_readout.py` | - | Finite-candidate saved-prediction readout that scores canonical compact candidates instead of free-form JSON. Dry-run validates the constrained path without model weights. |
| `analyze_stage_a_saved_candidate_gate.py` | - | Public-safe fail-closed gate analyzer for ignored saved candidate-score JSONL. It reports score-gap thresholds without copying prompts, raw model text, scheduler logs, model state, or full score tables. |
| `run_stage_a_predictions_expanse.sbatch` | - | Expanse GPU job for cluster-side saved-prediction generation and scoring. |
| `run_stage_a_predictions_cayuga.sbatch` | - | Cayuga GPU job for cluster-side saved-prediction generation and scoring. |
| `run_stage_a_saved_prediction_candidate_readout_expanse.sbatch` | - | Expanse GPU job for constrained saved-prediction candidate scoring. |
| `run_stage_a_saved_prediction_candidate_readout_cayuga.sbatch` | - | Cayuga GPU job for constrained saved-prediction candidate scoring. |
| `stage_a_sft_smoke_eval_summary_2026-07-04.json` | 1 | Deterministic policy baseline report for oracle, nearest-train, majority, and self-answer policies. |
| `STAGE_A_SFT_SMOKE_EVAL_2026-07-04.md` | - | Human-readable summary of the Stage A SFT smoke/eval harness result. |
| `stage_a_cayuga_hf_chat_baseline_summary_2026-07-04.json` | 1 | Compact public-safe summary of the first real Cayuga HF chat baseline. |
| `STAGE_A_CAYUGA_HF_CHAT_BASELINE_2026-07-04.md` | - | Human-readable result note for the first real Cayuga HF chat baseline. |
| `stage_a_cayuga_strict_contract_summary_2026-07-04.json` | 1 | Compact public-safe summary of the strict prompt-contract Cayuga baseline. |
| `STAGE_A_CAYUGA_STRICT_CONTRACT_2026-07-04.md` | - | Human-readable result note for the strict prompt-contract Cayuga baseline. |
| `export_stage_a_strict_contract_data.py` | - | Builds SFT, preference, process, and split artifacts for the same `stage_a_v2_strict` JSON contract used by saved-prediction generation. |
| `stage_a_strict_contract_sft_v1.jsonl` | 25 | Strict-contract compact JSON targets for supervised learning of tool-call objects, evidence status, citations, and terminal action. |
| `stage_a_strict_contract_preferences_v1.jsonl` | 50 | Chosen/rejected compact JSON pairs targeting observed strict-baseline collapses. |
| `stage_a_strict_contract_process_v1.jsonl` | 25 | Process targets for strict output keys, ordered tools, query fields, evidence/action routing, and citations. |
| `stage_a_strict_contract_sft_train_v1.jsonl` | 20 | Train split of strict-contract SFT targets. |
| `stage_a_strict_contract_sft_heldout_v1.jsonl` | 5 | Held-out split of strict-contract SFT targets. |
| `stage_a_strict_contract_preferences_train_v1.jsonl` | 40 | Train split of strict-contract observed-collapse preference pairs. |
| `stage_a_strict_contract_preferences_heldout_v1.jsonl` | 10 | Held-out split of strict-contract observed-collapse preference pairs. |
| `stage_a_strict_contract_process_train_v1.jsonl` | 20 | Train split of strict-contract process targets. |
| `stage_a_strict_contract_process_heldout_v1.jsonl` | 5 | Held-out split of strict-contract process targets. |
| `stage_a_strict_contract_manifest.json` | 1 | Strict-contract export provenance, pass/fail direction, failure-mode counts, and split/source-overlap checks. |
| `run_stage_a_strict_contract_sft_smoke.py` | - | Cluster-oriented strict-contract SFT smoke runner; dry-run validates artifacts without model load, full mode trains/generates/scores. |
| `run_stage_a_strict_sft_cayuga.sbatch` | - | Cayuga GPU job for strict-contract SFT smoke training plus held-out scoring. |
| `run_stage_a_strict_sft_expanse.sbatch` | - | Expanse GPU job for strict-contract SFT smoke training plus held-out scoring. |
| `stage_a_strict_sft_cayuga_smoke_summary_2026-07-04.json` | 1 | Compact public-safe summary of the first Cayuga strict-contract SFT smoke. |
| `STAGE_A_STRICT_SFT_CAYUGA_SMOKE_2026-07-04.md` | - | Human-readable result note for the first Cayuga strict-contract SFT smoke. |
| `run_stage_a_strict_component_diagnostics.py` | - | No-API counterfactual diagnostic that separates enum, tool-loop/query-field, and evidence/action routing gates. |
| `stage_a_strict_component_diagnostics_summary_2026-07-04.json` | 1 | Compact public-safe summary of strict-contract component diagnostics. |
| `STAGE_A_STRICT_COMPONENT_DIAGNOSTICS_2026-07-04.md` | - | Human-readable diagnostic note after the negative strict SFT smoke. |
| `export_stage_a_strict_component_targets.py` | - | Builds separable enum/action, tool-query, and routing-after-loop targets from strict-contract SFT rows. |
| `stage_a_strict_component_targets_v1.jsonl` | 75 | Full strict component target set, three component rows per Stage A case. |
| `stage_a_strict_component_targets_train_v1.jsonl` | 60 | Train split of strict component targets. |
| `stage_a_strict_component_targets_heldout_v1.jsonl` | 15 | Held-out split of strict component targets. |
| `stage_a_strict_component_targets_manifest.json` | 1 | Component target provenance, split counts, component counts, and overlap checks. |
| `run_stage_a_strict_component_sft_smoke.py` | - | Cluster-oriented component-slice SFT smoke runner; dry-run validates one slice without model load, full mode trains/generates/scores, and constrained enum/routing readouts isolate free-form JSON failures. |
| `generate_stage_a_component_predictions.py` | - | No-model component saved-prediction producer and scorer for oracle, collapse, citation, and tool-shape baselines. |
| `analyze_stage_a_enum_candidate_scores.py` | - | Public-safe pair-rank, field-rank, and margin analyzer for ignored enum candidate-score prediction artifacts. |
| `analyze_stage_a_enum_candidate_gate.py` | - | Public-safe score-gap/fail-closed gate analyzer for ignored enum candidate-score artifacts. |
| `analyze_stage_a_routing_candidate_scores.py` | - | Public-safe rank/margin analyzer for constrained routing candidate-score artifacts, including citation-sensitive exact targets. |
| `analyze_stage_a_component_visibility.py` | - | Public-safe audit for whether strict component prompts expose model-visible evidence/tool results needed for action/evidence-status routing. |
| `run_stage_a_strict_component_sft_cayuga.sbatch` | - | Cayuga GPU job for strict component-slice SFT smoke. |
| `run_stage_a_strict_component_sft_expanse.sbatch` | - | Expanse GPU job for strict component-slice SFT smoke. |
| `stage_a_component_enum_action_sft_cayuga_summary_2026-07-05.json` | 1 | Compact public-safe summary of the first Cayuga enum/action component SFT smoke. |
| `STAGE_A_COMPONENT_ENUM_ACTION_SFT_CAYUGA_2026-07-05.md` | - | Human-readable result note for the enum/action component SFT smoke. |
| `stage_a_component_enum_action_candidate_sft_cayuga_summary_2026-07-05.json` | 1 | Compact public-safe summary of the finite-candidate enum/action repair run. |
| `STAGE_A_COMPONENT_ENUM_ACTION_CANDIDATE_SFT_CAYUGA_2026-07-05.md` | - | Human-readable result note for the finite-candidate enum/action repair run. |
| `stage_a_component_enum_action_candidate_fullrank_cayuga_summary_2026-07-05.json` | 1 | Compact public-safe full-rank diagnostic for enum/action candidate scoring. |
| `STAGE_A_COMPONENT_ENUM_ACTION_CANDIDATE_FULLRANK_CAYUGA_2026-07-05.md` | - | Human-readable rank/margin diagnostic for the enum/action candidate scorer. |
| `stage_a_component_enum_action_field_rank_cayuga_summary_2026-07-05.json` | 1 | Compact public-safe field-rank reanalysis of the Cayuga full-rank enum/action scores. |
| `STAGE_A_COMPONENT_ENUM_ACTION_FIELD_RANK_CAYUGA_2026-07-05.md` | - | Human-readable field-rank diagnostic for action versus evidence-status failures. |
| `stage_a_component_enum_action_observed_pair_counterfactual_2026-07-05.json` | 1 | Compact public-safe counterfactual over train-observed valid enum pairs. |
| `STAGE_A_COMPONENT_ENUM_ACTION_OBSERVED_PAIR_COUNTERFACTUAL_2026-07-05.md` | - | Human-readable observed-pair pruning diagnostic for enum/action routing. |
| `export_stage_a_enum_corrective_pairs.py` | - | Builds enum-only contrast pairs against the observed `ground` / `supported` collapse. |
| `stage_a_enum_corrective_pairs_v1.jsonl` | 20 | Full enum corrective pair set, excluding cases where the oracle target is already `ground` / `supported`. |
| `stage_a_enum_corrective_pairs_train_v1.jsonl` | 16 | Train split of enum corrective pairs. |
| `stage_a_enum_corrective_pairs_heldout_v1.jsonl` | 4 | Held-out split of enum corrective pairs. |
| `stage_a_enum_corrective_pairs_manifest.json` | 1 | Corrective pair provenance, chosen-pair counts, skipped cases, and split/source-overlap checks. |
| `export_stage_a_enum_action_contrast_pairs.py` | - | Builds same-status/wrong-action enum contrasts after the field-rank diagnostic found weak `flag` action representation. |
| `stage_a_enum_action_contrast_pairs_v1.jsonl` | 20 | Full enum action-contrast pair set, excluding cases where the oracle action is already `ground`. |
| `stage_a_enum_action_contrast_pairs_train_v1.jsonl` | 16 | Train split of same-status action-contrast pairs. |
| `stage_a_enum_action_contrast_pairs_heldout_v1.jsonl` | 4 | Held-out split of same-status action-contrast pairs. |
| `stage_a_enum_action_contrast_pairs_manifest.json` | 1 | Action-contrast provenance, rejected-pair counts, priority flag/invalid-value anchor, and overlap checks. |
| `run_stage_a_enum_corrective_sft_smoke.py` | - | Cluster-oriented enum corrective SFT/margin smoke runner; dry-run validates corrective pairs without model load; `--target-format action_only` diagnoses action-label learning without status JSON coupling; `--pairwise-margin-weight` adds a supervised chosen-over-rejected hinge objective; `--candidate-ce-weight` adds finite-candidate enum cross-entropy; `--candidate-ce-mode` switches pair, field, or pair-plus-field CE; `--score-enum-candidates` adds finite-candidate enum selection readouts. |
| `run_stage_a_enum_corrective_sft_cayuga.sbatch` | - | Cayuga GPU job for enum corrective SFT/margin smoke. |
| `run_stage_a_enum_corrective_sft_expanse.sbatch` | - | Expanse GPU fallback job for enum corrective SFT/margin smoke. |
| `stage_a_enum_corrective_sft_cayuga_summary_2026-07-05.json` | 1 | Compact public-safe summary of the first Cayuga enum corrective SFT/margin smoke. |
| `STAGE_A_ENUM_CORRECTIVE_SFT_CAYUGA_2026-07-05.md` | - | Human-readable result note for the enum corrective SFT/margin smoke. |
| `stage_a_enum_corrective_margin_delta_cayuga_summary_2026-07-05.json` | 1 | Compact public-safe base-vs-trained margin delta summary for the enum corrective smoke. |
| `STAGE_A_ENUM_CORRECTIVE_MARGIN_DELTA_CAYUGA_2026-07-05.md` | - | Human-readable result note for the enum corrective margin-delta diagnostic. |
| `stage_a_enum_corrective_targeted_cayuga_summary_2026-07-05.json` | 1 | Compact public-safe summary of the targeted enum corrective sampling diagnostic. |
| `STAGE_A_ENUM_CORRECTIVE_TARGETED_CAYUGA_2026-07-05.md` | - | Human-readable result note for targeted enum corrective sampling. |
| `stage_a_enum_action_contrast_cayuga_summary_2026-07-05.json` | 1 | Compact public-safe summary of the Cayuga same-status action-contrast diagnostic. |
| `STAGE_A_ENUM_ACTION_CONTRAST_CAYUGA_2026-07-05.md` | - | Human-readable result note for the enum action-contrast diagnostic. |
| `stage_a_enum_action_only_cayuga_summary_2026-07-05.json` | 1 | Compact public-safe summary of the Cayuga action-only target-format diagnostic. |
| `STAGE_A_ENUM_ACTION_ONLY_CAYUGA_2026-07-05.md` | - | Human-readable result note for the enum action-only diagnostic. |
| `stage_a_enum_pairwise_margin_cayuga_summary_2026-07-05.json` | 1 | Compact public-safe summary of the Cayuga supervised pairwise-margin diagnostic. |
| `STAGE_A_ENUM_PAIRWISE_MARGIN_CAYUGA_2026-07-05.md` | - | Human-readable result note for the enum pairwise-margin diagnostic. |
| `stage_a_enum_full_pairwise_margin_cayuga_summary_2026-07-05.json` | 1 | Compact public-safe summary of full action-plus-status pairwise-margin diagnostics. |
| `STAGE_A_ENUM_FULL_PAIRWISE_MARGIN_CAYUGA_2026-07-05.md` | - | Human-readable result note for the full enum pairwise-margin diagnostics. |
| `stage_a_enum_candidate_readout_cayuga_summary_2026-07-05.json` | 1 | Compact public-safe summary of finite-candidate enum selection readouts after pairwise-margin SFT. |
| `STAGE_A_ENUM_CANDIDATE_READOUT_CAYUGA_2026-07-05.md` | - | Human-readable result note for the enum candidate-selection readout. |
| `stage_a_enum_candidate_ce_pair_observed_cayuga_summary_2026-07-05.json` | 1 | Compact public-safe summary of the 5-way enum candidate-CE Cayuga diagnostic. |
| `STAGE_A_ENUM_CANDIDATE_CE_PAIR_OBSERVED_CAYUGA_2026-07-05.md` | - | Human-readable result note for the enum candidate-CE diagnostic. |
| `stage_a_enum_candidate_ce_gate_cayuga_summary_2026-07-05.json` | 1 | Compact public-safe summary of the enum candidate-CE score-gap gate diagnostic. |
| `STAGE_A_ENUM_CANDIDATE_CE_GATE_CAYUGA_2026-07-05.md` | - | Human-readable result note for the score-gap fail-closed gate diagnostic. |
| `stage_a_enum_field_ce_pair_observed_cayuga_summary_2026-07-05.json` | 1 | Compact public-safe summary of the factorized action/status field-CE Cayuga diagnostic. |
| `STAGE_A_ENUM_FIELD_CE_PAIR_OBSERVED_CAYUGA_2026-07-05.md` | - | Human-readable result note for the enum field-CE diagnostic. |
| `stage_a_component_visibility_audit_2026-07-05.json` | 1 | Compact public-safe audit showing hidden-label leaks are 0/75, while 50 evidence-routing component rows lack model-visible evidence/tool results. |
| `STAGE_A_COMPONENT_VISIBILITY_AUDIT_2026-07-05.md` | - | Human-readable visibility audit explaining why the next component substrate should expose evidence-conditioned state before more loss-shape or method escalation. |
| `export_stage_a_evidence_conditioned_component_targets.py` | - | Builds public-safe evidence-conditioned component targets from strict component rows without exposing hidden labels in prompts. |
| `stage_a_evidence_conditioned_component_targets_v1.jsonl` | 75 | Full evidence-conditioned component target set: enum/action and routing rows include model-visible synthetic evidence state; tool-query rows remain query-only. |
| `stage_a_evidence_conditioned_component_targets_train_v1.jsonl` | 60 | Train split of evidence-conditioned component targets. |
| `stage_a_evidence_conditioned_component_targets_heldout_v1.jsonl` | 15 | Held-out split of evidence-conditioned component targets. |
| `stage_a_evidence_conditioned_component_targets_manifest.json` | 1 | Evidence-conditioned component provenance, split counts, component counts, evidence-conditioned row counts, and overlap checks. |
| `stage_a_evidence_conditioned_component_visibility_audit_2026-07-05.json` | 1 | Compact public-safe audit showing 0/75 hidden-label leaks and 0/75 underdetermined evidence-routing rows after adding model-visible evidence/tool-result state. |
| `STAGE_A_EVIDENCE_CONDITIONED_COMPONENT_VISIBILITY_AUDIT_2026-07-05.md` | - | Human-readable audit for the evidence-conditioned component target substrate. |
| `stage_a_evidence_enum_action_observed_pair_cayuga_summary_2026-07-06.json` | 1 | Compact public-safe summary of the first Cayuga evidence-conditioned enum/action component result. |
| `STAGE_A_EVIDENCE_ENUM_ACTION_OBSERVED_PAIR_CAYUGA_2026-07-06.md` | - | Human-readable result note showing evidence visibility does not repair enum top-1 selection by itself. |
| `stage_a_evidence_routing_after_loop_cayuga_summary_2026-07-06.json` | 1 | Compact public-safe summary of the first Cayuga evidence-conditioned routing-after-loop component result. |
| `STAGE_A_EVIDENCE_ROUTING_AFTER_LOOP_CAYUGA_2026-07-06.md` | - | Human-readable result note showing free-form routing still fails schema and enum gates despite visible tool-result content. |
| `stage_a_evidence_routing_observed_pair_cayuga_summary_2026-07-08.json` | 1 | Compact public-safe summary of the constrained evidence-conditioned routing readout. |
| `STAGE_A_EVIDENCE_ROUTING_OBSERVED_PAIR_CAYUGA_2026-07-08.md` | - | Human-readable result note showing constrained routing fixes schema/enum gates but leaves 3/5 action/status routing misses. |
| `export_stage_a_routing_action_status_contrast_pairs.py` | - | Builds routing action/status contrast pairs from the constrained-routing failure families without exposing hidden labels in prompts. |
| `stage_a_routing_action_status_contrast_pairs_v1.jsonl` | 15 | Full routing action/status contrast pair set for insufficient, verify, and invalid-value routing confusions. |
| `stage_a_routing_action_status_contrast_pairs_train_v1.jsonl` | 12 | Train split of routing action/status contrast pairs. |
| `stage_a_routing_action_status_contrast_pairs_heldout_v1.jsonl` | 3 | Held-out split of routing action/status contrast pairs. |
| `stage_a_routing_action_status_contrast_pairs_manifest.json` | 1 | Routing contrast provenance, rejection map, skipped solved families, and split/source-overlap checks. |
| `run_stage_a_routing_contrast_sft_smoke.py` | - | Cluster-oriented routing contrast SFT/margin smoke runner; dry-run validates routing contrast pairs and finite candidate space without model load, while full mode scores base/train/held-out margins plus optional finite-candidate routing ranks. |
| `run_stage_a_routing_contrast_sft_cayuga.sbatch` | - | Cayuga GPU job for routing action/status contrast SFT/margin smoke. |
| `run_stage_a_routing_contrast_sft_expanse.sbatch` | - | Expanse GPU fallback job for routing action/status contrast SFT/margin smoke. |
| `stage_a_routing_contrast_sft_cayuga_summary_2026-07-08.json` | 1 | Compact public-safe summary of the Cayuga routing action/status contrast SFT/margin result. |
| `STAGE_A_ROUTING_CONTRAST_SFT_CAYUGA_2026-07-08.md` | - | Human-readable result note showing teacher-forced routing margins repair on the 3-case held-out contrast slice. |
| `stage_a_routing_contrast_candidate_cayuga_summary_2026-07-08.json` | 1 | Compact public-safe summary of the Cayuga routing contrast finite-candidate rank result. |
| `STAGE_A_ROUTING_CONTRAST_CANDIDATE_CAYUGA_2026-07-08.md` | - | Human-readable result note showing finite-candidate transfer is partial: 0/3 -> 2/3 exact top-1, with defer/insufficient still unresolved. |
| `export_stage_a_routing_defer_verify_contrast_pairs.py` | - | Builds the targeted defer-vs-verify routing boundary pairs from evidence-conditioned routing rows. |
| `stage_a_routing_defer_verify_contrast_pairs_v1.jsonl` | 10 | Full defer-vs-verify routing contrast pair set for insufficient and verification-needed families. |
| `stage_a_routing_defer_verify_contrast_pairs_train_v1.jsonl` | 8 | Train split of defer-vs-verify routing contrast pairs. |
| `stage_a_routing_defer_verify_contrast_pairs_heldout_v1.jsonl` | 2 | Held-out split of defer-vs-verify routing contrast pairs. |
| `stage_a_routing_defer_verify_contrast_pairs_manifest.json` | 1 | Defer-vs-verify routing contrast provenance, split/source-overlap checks, and rejection map. |
| `stage_a_routing_defer_verify_cayuga_summary_2026-07-08.json` | 1 | Compact public-safe summary of the targeted defer-vs-verify Cayuga smoke result. |
| `STAGE_A_ROUTING_DEFER_VERIFY_CAYUGA_2026-07-08.md` | - | Human-readable result note showing the targeted smoke remains 1/2 exact top-1 and does not repair the defer boundary. |
| `analyze_stage_a_routing_candidate_gate.py` | - | Builds compact fail-closed gate diagnostics from ignored routing candidate-score JSONL artifacts. |
| `stage_a_routing_defer_verify_gate_trained_2026-07-08.json` | 1 | Compact public-safe fail-closed gate diagnostic over the trained defer-vs-verify routing candidate scores. |
| `STAGE_A_ROUTING_DEFER_VERIFY_GATE_TRAINED_2026-07-08.md` | - | Human-readable gate diagnostic showing threshold 0.025 has 0 unsafe trusted rows on the tiny 2-case held-out boundary slice. |
| `evaluate_stage_a_routing_evidence_boundary_gate.py` | - | Evaluates a no-model defer-vs-verify runtime gate using only prompt-visible tool-result fields. |
| `stage_a_routing_evidence_boundary_gate_2026-07-08.json` | 1 | Compact public-safe evidence-boundary gate result: 10/10 overall and 2/2 held-out. |
| `STAGE_A_ROUTING_EVIDENCE_BOUNDARY_GATE_2026-07-08.md` | - | Human-readable result note for the deterministic evidence-boundary gate baseline. |
| `evaluate_stage_a_routing_evidence_gate.py` | - | Evaluates a no-model runtime gate across all Stage A `routing_after_loop` evidence-conditioned rows using only prompt-visible tool-result fields. |
| `stage_a_routing_evidence_gate_2026-07-08.json` | 1 | Compact public-safe routing evidence gate result: 25/25 overall, 20/20 train, and 5/5 held-out. |
| `STAGE_A_ROUTING_EVIDENCE_GATE_2026-07-08.md` | - | Human-readable result note for the all-family deterministic routing evidence gate baseline. |
| `evaluate_stage_a_routing_gate_baseline_comparison.py` | - | Compares the all-family runtime evidence gate against oracle, collapse, citationless, and empty-object routing baselines. |
| `stage_a_routing_gate_baseline_comparison_2026-07-09.json` | 1 | Compact public-safe comparison: runtime/oracle 25/25, `ground`/`supported` collapse 5/25 with 20 unsafe overrides, citationless routing 15/25 with 10 citation mismatches. |
| `STAGE_A_ROUTING_GATE_BASELINE_COMPARISON_2026-07-09.md` | - | Human-readable comparison report for routing gate versus deterministic component baselines. |
| `evaluate_stage_a_routing_model_readiness.py` | - | Compares compact Cayuga routing summaries against the public runtime/baseline scorecard and emits escalation gates. |
| `stage_a_routing_model_readiness_2026-07-09.json` | 1 | Compact public-safe readiness report: best all-family model readout is 2/5, below citationless 3/5 and runtime gate 5/5. |
| `STAGE_A_ROUTING_MODEL_READINESS_2026-07-09.md` | - | Human-readable readiness report keeping `tool_query`, DPO/RLVR, HF publication, and release tagging gated. |
| `evaluate_stage_a_full_trajectory_arbitration.py` | - | Projects routing runtime-gate decisions into canonical full Stage A trajectories and compares oracle, collapse, citationless, runtime, and hybrid policies. |
| `stage_a_full_trajectory_arbitration_2026-07-09.json` | 1 | Compact public-safe full-trajectory arbitration report: runtime/hybrid 25/25, collapse 5/25 with 20 unsafe overrides, citationless 15/25 with attribution failures. |
| `STAGE_A_FULL_TRAJECTORY_ARBITRATION_2026-07-09.md` | - | Human-readable full-trajectory arbitration report using the canonical Stage A evaluator. |
| `evaluate_stage_a_saved_prediction_readiness.py` | - | Compares compact real saved-output summaries, deterministic saved-output smokes, and saved-candidate fail-closed gates against the full-trajectory arbitration scorecard. |
| `evaluate_stage_a_saved_output_next_decision.py` | - | Derives the next Stage A saved-output experiment from compact readiness and saved-candidate gate checkpoints without reading raw run folders. |
| `export_stage_a_saved_output_calibration_probe.py` | - | Builds split-safe target-vs-`ground`/`supported` calibration probe pairs from the saved-output next-decision checkpoint. |
| `run_stage_a_saved_output_calibration_probe_readout.py` | - | Scores calibration-probe target-vs-collapse outputs and evaluates a train-selected fail-closed gate; use `--dry-run` for CI and `--allow-model-load` on Cayuga/Expanse. |
| `run_stage_a_saved_output_calibration_probe_readout_cayuga.sbatch` | - | Cayuga submit template for the saved-output calibration probe readout. |
| `run_stage_a_saved_output_calibration_probe_readout_expanse.sbatch` | - | Expanse fallback submit template for the saved-output calibration probe readout. |
| `run_stage_a_saved_output_calibration_margin_sft.py` | - | Train-only saved-output calibration SFT/margin smoke runner; dry-run validates 16 train and 4 held-out probe pairs without model load, while full mode scores base/train/held-out margins plus optional train-side and held-out finite-candidate ranks; optional candidate CE adds explicit pair/field action-status routing pressure without DPO/RLVR. |
| `run_stage_a_saved_output_calibration_margin_sft_cayuga.sbatch` | - | Cayuga GPU job for saved-output calibration margin SFT. |
| `run_stage_a_saved_output_calibration_margin_sft_expanse.sbatch` | - | Expanse fallback GPU job for saved-output calibration margin SFT. |
| `analyze_stage_a_saved_output_candidate_fields.py` | - | Public-safe field-rank analyzer for ignored saved-output candidate-score JSONL; summarizes action/status rank patterns without copying prompts, raw model text, or full score tables. |
| `analyze_stage_a_saved_output_candidate_calibration.py` | - | Public-safe train-derived calibration analyzer for ignored saved-output train/held-out candidate-score JSONL; applies pair-mean centering from train rows without held-out tuning. |
| `evaluate_stage_a_saved_output_candidate_arbitration.py` | - | Public-safe arbitration report over compact saved-output calibration results and tracked model-visible evidence targets; compares raw candidate, calibrated candidate, train-selected score-gap gate, evidence gate, and hybrid policies without reading raw run artifacts. |
| `build_stage_a_saved_output_policy_summary.py` | - | Public-safe adapter that converts compact saved-output summaries, saved-candidate gate summaries, or candidate-arbitration policy rows into the `--model-policy-summary` JSON contract for the meet-or-beat gate. |
| `evaluate_stage_a_saved_output_meet_or_beat_gate.py` | - | Public-safe acceptance gate for future saved-output Cayuga results; consumes compact next-decision and arbitration reports plus optional `--model-policy-summary` JSON, then checks whether candidate/model policies meet or beat runtime evidence arbitration without unsafe trust. |
| `evaluate_stage_a_saved_output_intake_contract.py` | - | Public-safe intake verifier for the saved-output checkpoint bundle; checks compact artifact hashes, next-decision criteria, meet-or-beat gate inputs, and future policy raw-artifact flags before accepting a new Cayuga policy summary. |
| `stage_a_v3_tool_trace_qwen05b_cayuga_summary_2026-07-09.json` | 1 | Compact public-safe Cayuga v3 prompt-contract summary: Qwen2.5-0.5B writes JSON-like outputs but fails 5/5 with invalid `evidence_status: verified`. |
| `stage_a_v4_canonical_json_qwen05b_cayuga_summary_2026-07-09.json` | 1 | Compact public-safe Cayuga v4 prompt-contract summary: invalid `verified` status disappears, but canonical action/envelope generation still fails 5/5. |
| `stage_a_saved_candidate_readout_qwen05b_train_observed_summary_2026-07-09.json` | 1 | Compact public-safe Cayuga finite-candidate readout summary: train-observed candidates remove parse/tool/query failures but select `ground` / `supported` in 5/5. |
| `stage_a_saved_candidate_readout_qwen05b_all_valid_summary_2026-07-09.json` | 1 | Compact public-safe Cayuga finite-candidate readout summary: all valid external action/status pairs still select `ground` / `supported` in 5/5. |
| `analyze_stage_a_saved_candidate_gate.py` | - | Post-hoc fail-closed score-gap analyzer for the ignored raw candidate readout JSONL. Compact output only; use it before interpreting any candidate top-1 score as trusted behavior. |
| `stage_a_saved_candidate_gate_train_observed_qwen05b_2026-07-09.json` | 1 | Compact public-safe gate diagnostic: train-observed candidate scores trust 1/5 supported row with 0 unsafe trust; strict final correctness is 2/5 after fail-closed routing. |
| `STAGE_A_SAVED_CANDIDATE_GATE_TRAIN_OBSERVED_QWEN05B_2026-07-09.md` | - | Human-readable train-observed saved-candidate gate diagnostic. |
| `stage_a_saved_candidate_gate_all_valid_qwen05b_2026-07-09.json` | 1 | Compact public-safe gate diagnostic: all-valid candidate scores trust 1/5 supported row with 0 unsafe trust; strict final correctness is 2/5 after fail-closed routing. |
| `STAGE_A_SAVED_CANDIDATE_GATE_ALL_VALID_QWEN05B_2026-07-09.md` | - | Human-readable all-valid saved-candidate gate diagnostic. |
| `stage_a_saved_prediction_readiness_2026-07-09.json` | 1 | Compact public-safe saved-prediction readiness report: best real saved output remains 0/5, while best fail-closed candidate gate is 2/5 strict final and still below citationless/runtime baselines. |
| `STAGE_A_SAVED_PREDICTION_READINESS_2026-07-09.md` | - | Human-readable saved-prediction readiness report with saved-candidate gate comparison, keeping optimization and publication escalation gated. |
| `stage_a_saved_output_next_decision_2026-07-09.json` | 1 | Compact public-safe next-decision checkpoint selecting a targeted action/status calibration probe and keeping `tool_query`, DPO/RLVR, HF, and release tagging gated. |
| `STAGE_A_SAVED_OUTPUT_NEXT_DECISION_2026-07-09.md` | - | Human-readable next-decision checkpoint for the Stage A saved-output path. |
| `stage_a_saved_output_calibration_probe_v1.jsonl` | 20 | Target-vs-`ground`/`supported` action/status calibration pairs selected from the saved-output next-decision checkpoint. |
| `stage_a_saved_output_calibration_probe_train_v1.jsonl` | 16 | Train-allowed split of saved-output calibration probe pairs, balanced across four target action/status families. |
| `stage_a_saved_output_calibration_probe_heldout_v1.jsonl` | 4 | Evaluation-only split of saved-output calibration probe pairs, one held-out row per target action/status family. |
| `stage_a_saved_output_calibration_probe_manifest.json` | 1 | Public-safe calibration probe manifest with split/source overlap checks, artifact policy, and next-gate criteria. |
| `stage_a_saved_output_calibration_probe_readout_qwen05b_cayuga_summary_2026-07-10.json` | 1 | Compact public-safe Cayuga readout summary: Qwen2.5-0.5B scores `ground` / `supported` above target outputs on all 20 calibration probe rows. |
| `STAGE_A_SAVED_OUTPUT_CALIBRATION_PROBE_READOUT_QWEN05B_CAYUGA_2026-07-10.md` | - | Human-readable saved-output calibration probe readout result. |
| `stage_a_saved_output_calibration_margin_sft_qwen05b_cayuga_summary_2026-07-10.json` | 1 | Compact public-safe Cayuga margin SFT summary: held-out target-vs-collapse wins move from 0/4 to 1/4, with mean margin still below zero. |
| `STAGE_A_SAVED_OUTPUT_CALIBRATION_MARGIN_SFT_QWEN05B_CAYUGA_2026-07-10.md` | - | Human-readable saved-output calibration margin SFT result. |
| `stage_a_saved_output_calibration_margin_sft_focus_nonverify3_qwen05b_cayuga_summary_2026-07-10.json` | 1 | Compact public-safe focused Cayuga margin SFT summary: non-verify family oversampling moves held-out target-vs-collapse wins from 0/4 to 3/4, with `flag` / `invalid_value` still unresolved. |
| `STAGE_A_SAVED_OUTPUT_CALIBRATION_MARGIN_SFT_FOCUS_NONVERIFY3_QWEN05B_CAYUGA_2026-07-10.md` | - | Human-readable focused saved-output calibration margin SFT result. |
| `stage_a_saved_output_target_format_flag_qwen05b_cayuga_summary_2026-07-10.json` | 1 | Compact public-safe target-format diagnostic summary: isolated `flag`, `invalid_value`, and `flag` + `invalid_value` targets pass while full JSON remains unresolved. |
| `STAGE_A_SAVED_OUTPUT_TARGET_FORMAT_FLAG_QWEN05B_CAYUGA_2026-07-10.md` | - | Human-readable saved-output target-format diagnostic result. |
| `stage_a_saved_output_same_model_target_format_qwen05b_cayuga_summary_2026-07-10.json` | 1 | Compact public-safe same-model target-format summary: full-target `flag`-focused SFT flips the held-out full JSON `flag` / `invalid_value` margin positive while keeping action/status projections positive. |
| `STAGE_A_SAVED_OUTPUT_SAME_MODEL_TARGET_FORMAT_QWEN05B_CAYUGA_2026-07-10.md` | - | Human-readable same-model target-format result; this is a teacher-forced repair signal, not candidate-selection or trajectory readiness. |
| `stage_a_saved_output_candidate_rank_qwen05b_cayuga_summary_2026-07-10.json` | 1 | Compact public-safe candidate-rank summary: teacher-forced full margins stay repaired, but 5-way candidate top-1 is 1/4 and the trained model over-selects `flag` / `invalid_value`. |
| `STAGE_A_SAVED_OUTPUT_CANDIDATE_RANK_QWEN05B_CAYUGA_2026-07-10.md` | - | Human-readable candidate-rank result showing margin repair does not transfer to candidate-selection readiness. |
| `stage_a_saved_output_candidate_field_qwen05b_cayuga_summary_2026-07-10.json` | 1 | Compact public-safe field-rank summary: trained candidate selection has `both_field_failure` in the three non-flag held-out rows and `pair_top1` only for `flag` / `invalid_value`. |
| `STAGE_A_SAVED_OUTPUT_CANDIDATE_FIELD_QWEN05B_CAYUGA_2026-07-10.md` | - | Human-readable field diagnostic showing the candidate failure is combined `flag` / `invalid_value` over-selection. |
| `stage_a_saved_output_train_candidate_qwen05b_cayuga_summary_2026-07-10.json` | 1 | Compact public-safe train-candidate diagnostic: trained train rows also top-rank `flag` / `invalid_value` in 16/16 cases, confirming a global candidate prior rather than held-out-only noise. |
| `STAGE_A_SAVED_OUTPUT_TRAIN_CANDIDATE_QWEN05B_CAYUGA_2026-07-10.md` | - | Human-readable train-candidate diagnostic keeping held-out tuning, DPO/RLVR, and release escalation gated. |
| `stage_a_saved_output_candidate_calibration_qwen05b_cayuga_summary_2026-07-10.json` | 1 | Compact public-safe train-derived candidate calibration diagnostic: pair-mean centering improves held-out top-1 from 1/4 to 2/4, but the train-selected zero-unsafe gate trusts 0/4 held-out rows. |
| `STAGE_A_SAVED_OUTPUT_CANDIDATE_CALIBRATION_QWEN05B_CAYUGA_2026-07-10.md` | - | Human-readable calibration/gate diagnostic showing partial correction is not yet trust/deployment readiness. |
| `stage_a_saved_output_candidate_arbitration_2026-07-10.json` | 1 | Compact public-safe arbitration report: raw candidate top-1 is 1/4, calibrated top-1 is 2/4, train-selected score-gap gate is 1/4, while model-visible evidence and hybrid policies are 4/4. |
| `STAGE_A_SAVED_OUTPUT_CANDIDATE_ARBITRATION_2026-07-10.md` | - | Human-readable arbitration report keeping runtime evidence arbitration as the system baseline before more model-heavy optimization. |
| `stage_a_saved_output_next_decision_2026-07-10.json` | 1 | Compact public-safe next-decision update: the targeted calibration probe is complete, and the next model-heavy checkpoint must meet or beat runtime evidence/hybrid arbitration at 4/4 with zero unsafe candidate trust. |
| `STAGE_A_SAVED_OUTPUT_NEXT_DECISION_2026-07-10.md` | - | Human-readable next-decision update keeping `tool_query`, DPO/RLVR, Hugging Face publication, release tagging, and broad retraining gated. |
| `stage_a_saved_output_meet_or_beat_gate_2026-07-10.json` | 1 | Compact public-safe meet-or-beat gate: current raw, calibrated, and score-gap candidate policies fail against the 4/4 runtime evidence/hybrid baseline. |
| `STAGE_A_SAVED_OUTPUT_MEET_OR_BEAT_GATE_2026-07-10.md` | - | Human-readable meet-or-beat gate for judging future saved-output Cayuga summaries without reading raw run artifacts. |
| `stage_a_saved_output_intake_contract_2026-07-10.json` | 1 | Compact public-safe intake contract: current saved-output bundle passes hash, criteria, and public-safe flag checks before future Cayuga policy summaries are accepted. |
| `STAGE_A_SAVED_OUTPUT_INTAKE_CONTRACT_2026-07-10.md` | - | Human-readable intake contract for the saved-output checkpoint bundle. |
| `evaluate_stage_a_routing_gate_arbitration.py` | - | Compares raw candidate top-1, score-gap fail-closed routing, evidence-boundary override, and hybrid arbitration policies. |
| `stage_a_routing_gate_arbitration_2026-07-08.json` | 1 | Compact public-safe routing arbitration report over the 2-case defer-vs-verify held-out boundary. |
| `STAGE_A_ROUTING_GATE_ARBITRATION_2026-07-08.md` | - | Human-readable arbitration result showing runtime policies beat raw candidate top-1 on the held-out boundary slice. |

Validation:

```bash
python3 post_training/validate_post_training_data.py
```

Current Stage A validation status: 25/25 oracle SFT examples pass, 150/150
chosen preference trajectories pass, 0/150 rejected variants pass, and
split-group overlap is empty. The deterministic split uses one held-out case per
case family: 20 train cases and 5 held-out cases, with no train/held-out
`source_manifest_case_id`, `split_group`, or `source_task_id` overlap. The
strict-contract artifacts add 25 SFT targets, 50 observed-collapse preference
pairs, and 25 process rows; all chosen targets pass, all rejected targets fail,
and the same 20/5 split remains source-disjoint. Strict component targets add
75 slice rows: 25 enum/action, 25 tool-query, and 25 routing-after-loop targets,
with 60 train and 15 held-out rows and no train/held-out case, split-group, or
source-task overlap. The component visibility audit finds 0/75 hidden-label
leak rows, but 25/25 `enum_action` rows and 25/25 `routing_after_loop` rows are
underdetermined for evidence routing because the model-visible prompt lacks
evidence content or tool-result payloads. Evidence-conditioned component
targets add the same 75-row slice shape with public-safe evidence state for
50 enum/routing rows; their visibility audit reports 0/75 hidden-label leaks
and 0/75 underdetermined evidence-routing rows. The constrained routing readout
dry-run uses 20 train and 5 held-out routing rows, five train-observed
action/status pairs, and citation candidates extracted only from model-visible
tool-result content. The routing action/status contrast substrate adds 15 pairs
for the three unresolved families, with 12 train and 3 held-out pairs and no
train/held-out `source_manifest_case_id`, `split_group`, or `source_task_id`
overlap.
The all-family routing evidence gate reaches 25/25 overall, 20/20 train, and
5/5 held-out on all `routing_after_loop` evidence-conditioned rows while using
only model-visible tool-result fields. Treat it as the runtime baseline to beat,
not as model competence.
The routing gate baseline comparison shows the same runtime gate and oracle at
25/25, `ground`/`supported` collapse at 5/25 with 20 unsafe overrides, and
citationless routing at 15/25 with 10 citation mismatches.
The routing model-readiness gate keeps escalation blocked: the best all-family
Cayuga routing readout is 2/5, below citationless routing at 3/5 and the
runtime gate at 5/5.
The full-trajectory arbitration scaffold lifts the routing gate into
`score_stage_a_trajectory`: runtime-gate and hybrid runtime-over-collapse
policies reach 25/25 overall and 5/5 held-out, while collapse and citationless
policies fail the expected full-trajectory gates.
The saved-prediction readiness gate then compares existing compact Cayuga saved
output summaries against those full-trajectory baselines: the best real saved
output remains 0/5 held-out, below collapse, citationless, and runtime gates.

Evidence-conditioned component substrate:

```bash
python3 post_training/export_stage_a_evidence_conditioned_component_targets.py
python3 post_training/analyze_stage_a_component_visibility.py \
  --targets post_training/stage_a_evidence_conditioned_component_targets_v1.jsonl \
  --out-json /tmp/stage_a_evidence_conditioned_component_visibility_audit.json \
  --out-md /tmp/stage_a_evidence_conditioned_component_visibility_audit.md
python3 post_training/run_stage_a_strict_component_sft_smoke.py \
  --dry-run \
  --component enum_action \
  --targets post_training/stage_a_evidence_conditioned_component_targets_v1.jsonl \
  --train-targets post_training/stage_a_evidence_conditioned_component_targets_train_v1.jsonl \
  --heldout-targets post_training/stage_a_evidence_conditioned_component_targets_heldout_v1.jsonl
python3 post_training/run_stage_a_strict_component_sft_smoke.py \
  --dry-run \
  --component routing_after_loop \
  --decode-mode routing_observed_pair_score \
  --targets post_training/stage_a_evidence_conditioned_component_targets_v1.jsonl \
  --train-targets post_training/stage_a_evidence_conditioned_component_targets_train_v1.jsonl \
  --heldout-targets post_training/stage_a_evidence_conditioned_component_targets_heldout_v1.jsonl
python3 post_training/export_stage_a_routing_action_status_contrast_pairs.py
python3 post_training/run_stage_a_routing_contrast_sft_smoke.py \
  --dry-run \
  --pairwise-margin-weight 1 \
  --pairwise-margin 0.05
python3 post_training/evaluate_stage_a_routing_evidence_gate.py \
  --out-json /tmp/stage_a_routing_evidence_gate.json \
  --out-md /tmp/STAGE_A_ROUTING_EVIDENCE_GATE.md
python3 post_training/evaluate_stage_a_routing_gate_baseline_comparison.py \
  --out-json /tmp/stage_a_routing_gate_baseline_comparison.json \
  --out-md /tmp/STAGE_A_ROUTING_GATE_BASELINE_COMPARISON.md
python3 post_training/evaluate_stage_a_routing_model_readiness.py \
  --out-json /tmp/stage_a_routing_model_readiness.json \
  --out-md /tmp/STAGE_A_ROUTING_MODEL_READINESS.md
python3 post_training/evaluate_stage_a_full_trajectory_arbitration.py \
  --out-json /tmp/stage_a_full_trajectory_arbitration.json \
  --out-md /tmp/STAGE_A_FULL_TRAJECTORY_ARBITRATION.md
python3 post_training/evaluate_stage_a_saved_prediction_readiness.py \
  --out-json /tmp/stage_a_saved_prediction_readiness.json \
  --out-md /tmp/STAGE_A_SAVED_PREDICTION_READINESS.md
python3 post_training/evaluate_stage_a_saved_output_next_decision.py \
  --out-json /tmp/stage_a_saved_output_next_decision.json \
  --out-md /tmp/STAGE_A_SAVED_OUTPUT_NEXT_DECISION.md
```

No-API SFT smoke/eval harness:

```bash
python3 post_training/run_stage_a_sft_smoke_eval.py \
  --out post_training/stage_a_sft_smoke_eval_summary_2026-07-04.json
```

Current held-out policy baselines:

```text
oracle_replay: 5/5 pass, mean score 1.000
nearest_train_replay: 0/5 pass, mean score 0.657
train_majority_replay: 0/5 pass, mean score 0.771
self_answer: 0/5 pass, mean score 0.229
```

Interpretation: deterministic replay can preserve tool/query mechanics, but
held-out evidence status, terminal action, and attribution still require
case-specific evidence routing. This is the no-API baseline shape for the next
real Stage A SFT checkpoint.

Offline prediction-output scorer:

```bash
python3 post_training/generate_stage_a_predictions.py \
  --mode self_answer \
  --sft post_training/stage_a_sft_heldout_v1.jsonl \
  --out /tmp/stage_a_self_answer_predictions.jsonl \
  --run-id self_answer_saved_prediction_smoke
python3 post_training/evaluate_stage_a_predictions.py \
  --predictions /tmp/stage_a_self_answer_predictions.jsonl \
  --expected-sft post_training/stage_a_sft_heldout_v1.jsonl \
  --run-id self_answer_saved_prediction_smoke \
  --json
python3 post_training/evaluate_stage_a_predictions.py \
  --predictions post_training/stage_a_sft_heldout_v1.jsonl \
  --expected-sft post_training/stage_a_sft_heldout_v1.jsonl \
  --run-id heldout_oracle_adapter_smoke \
  --json
```

This public smoke command reuses the held-out oracle SFT trajectories as saved
predictions and should report 5/5 passing cases. Real API or cluster model
runs should first save one JSON object per Stage A case, then score that JSONL
file with the same command shape. Missing expected cases and unexpected extra
case IDs fail closed.

Producer modes:

| mode | API? | expected scorer behavior |
| --- | ---: | --- |
| `oracle` | No | 5/5 held-out pass when using the tracked oracle SFT targets. |
| `self_answer` | No | 0/5 held-out pass; preserves the shortcut-failure baseline. |
| `compact_tool_names_oracle` | No | Parses but fails query-completeness because arguments are missing. |
| `openai_chat` | Yes, opt-in | Requires `--allow-live-api` and `OPENAI_API_KEY`; writes raw model text for offline scoring. |
| `hf_chat` | Cluster/HPC model load, opt-in | Requires `--allow-model-load`; intended for Cayuga/Expanse jobs. |

Prompt contracts:

| contract | role |
| --- | --- |
| `basic` | Original minimal JSON artifact instruction. |
| `stage_a_v2_strict` | Adds model-visible enum, tool-call object, required-argument, and no-hidden-label instructions. It does not repair outputs after generation. |
| `stage_a_v3_tool_trace` | Adds an ordered four-tool trace requirement with `drug_id` and `condition_id` arguments for each tool call. It is a tool/query compliance diagnostic, not evidence-label supervision. |
| `stage_a_v4_canonical_json` | Adds the canonical compact JSON envelope and explicitly blocks invalid statuses such as `verified`, `valid`, `sourced`, `mixed`, or `related`. |

Cluster submit templates:

```bash
sbatch --account=<allocation> \
  --export=ALL,WORK=$PWD,HF_HOME=<hf-cache-root>,MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct,PROMPT_CONTRACT=stage_a_v4_canonical_json \
  post_training/run_stage_a_predictions_expanse.sbatch

sbatch --account=<allocation> --partition=<gpu-partition> --gres=gpu:1 \
  --export=ALL,WORK=$PWD,HF_HOME=<hf-cache-root>,MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct,PROMPT_CONTRACT=stage_a_v4_canonical_json \
  post_training/run_stage_a_predictions_cayuga.sbatch

sbatch --account=<allocation> --partition=<gpu-partition> --gres=gpu:1 \
  --export=ALL,WORK=$PWD,HF_HOME=<hf-cache-root>,MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct,CANDIDATE_POLICY=train_observed_pairs \
  post_training/run_stage_a_saved_prediction_candidate_readout_cayuga.sbatch
```

The cluster jobs write raw prediction JSONL and scorer reports under
`post_training/runs/`. Keep those run artifacts ignored unless a compact,
public-safe summary is intentionally curated. Pass `HF_HOME` explicitly when a
cluster job should reuse a site model cache. If the model cache has not been
warmed on the cluster, add `ALLOW_DOWNLOAD=1` to the exported variables for the
first run.

Current real cluster baseline:

```text
Cayuga Qwen/Qwen2.5-1.5B-Instruct hf_chat held-out: 0/5 pass, mean score 0.114
Cayuga Qwen/Qwen2.5-1.5B-Instruct hf_chat strict-contract held-out: 0/5 pass, mean score 0.372
Cayuga strict-contract tiny SFT Qwen/Qwen2.5-0.5B-Instruct: 0/5 pass, mean score 0.000
Cayuga strict-contract tiny SFT Qwen/Qwen2.5-1.5B-Instruct: 0/5 pass, mean score 0.372
Cayuga Qwen/Qwen2.5-0.5B-Instruct v3 tool-trace held-out: 0/5 pass, mean score 0.000; 5 invalid `verified` statuses
Cayuga Qwen/Qwen2.5-0.5B-Instruct v4 canonical JSON held-out: 0/5 pass, mean score 0.000; invalid `verified` status removed, but top-level action/envelope still fails
Cayuga Qwen/Qwen2.5-0.5B-Instruct saved candidate readout: 0/5 pass, mean score 0.657; parse/tool/query gates pass, top pair is `ground`/`supported` in 5/5
Cayuga train-observed saved-candidate gate: trusts 1/5 supported row with 0 unsafe trust; fail-closed strict final correct 2/5
Cayuga all-valid saved-candidate gate: trusts 1/5 supported row with 0 unsafe trust; fail-closed strict final correct 2/5
Cayuga enum_action component tiny SFT Qwen/Qwen2.5-0.5B-Instruct: 0/5 pass, mean score 0.250
Cayuga enum_action candidate-scored tiny SFT Qwen/Qwen2.5-0.5B-Instruct: 1/5 pass, mean score 0.800
Cayuga enum_action full-rank diagnostic: 1/5 pass, mean score 0.800; gold ranks 1, 5, 13, 4, 24
Cayuga enum_action field-rank diagnostic: action top-1 2/5, evidence-status top-1 1/5; invalid-value action rank 6
Cayuga enum_action observed-pair counterfactual: 1/5 top-1; top pair is ground/supported in 5/5
Enum corrective pairs: 20 total, 16 train, 4 held-out; rejected target is ground/supported collapse
Enum action-contrast pairs: 20 total, 16 train, 4 held-out; rejected target preserves status and forces action=ground
Cayuga enum_action action-contrast diagnostic: base 1/4 -> trained 2/4 held-out wins; mean margin -0.079 -> -0.001
Cayuga enum_action action-only diagnostic: base 1/4 -> trained 1/4 held-out wins; mean margin -0.211 -> -0.033
Cayuga enum_action pairwise-margin diagnostic: base 1/4 -> trained 4/4 held-out wins; mean margin -0.211 -> 0.262
Cayuga enum_action full pairwise-margin diagnostic: same-status action contrast base 1/4 -> trained 4/4; ground/supported corrective base 0/4 -> trained 4/4
Cayuga enum_action candidate readout: margins repair to 4/4, but candidate top-1 remains 0/4 for both 30-way and 5-way policies
Cayuga evidence-conditioned enum_action observed-pair diagnostic: 1/5 pass, mean score 0.800; top pair remains ground/supported in 5/5
Cayuga evidence-conditioned routing_after_loop free-form diagnostic: 0/5 pass, mean score 0.200; target-key and enum-validity accuracy 0.0
Cayuga evidence-conditioned routing_after_loop constrained diagnostic: 2/5 pass, mean score 0.850; target-key and enum-validity accuracy 1.0
Cayuga routing action/status contrast margin diagnostic: base 0/3 -> trained 3/3 held-out wins; mean margin -0.117 -> 0.115
Cayuga routing action/status candidate-rank diagnostic: base 0/3 -> trained 2/3 exact top-1; mean gold rank 3.0 -> 1.33
Routing defer-vs-verify boundary pairs: 10 total, 8 train, 2 held-out; rejected targets swap defer/insufficient and verify/insufficient
Cayuga routing defer-vs-verify diagnostic: base 1/2 -> trained 1/2 exact top-1; defer/insufficient remains ranked below verify/insufficient
Routing fail-closed gate diagnostic: trained threshold 0.025 trusts 1/2 rows, with 1/1 trusted correct, 0 unsafe trusted, and 2/2 strict final correct after fail-closed defer routing
Routing evidence-boundary gate: no-model prompt-visible evidence gate reaches 10/10 overall and 2/2 held-out on defer-vs-verify pairs
Routing evidence gate baseline comparison: runtime/oracle are 25/25, ground/supported collapse is 5/25 with 20 unsafe overrides, and citationless routing is 15/25 with 10 citation mismatches
Routing model-readiness gate: best all-family Cayuga readout is 2/5, below citationless routing at 3/5 and runtime gate at 5/5; escalation remains gated
Full-trajectory arbitration scaffold: runtime/hybrid are 25/25, collapse is 5/25 with 20 unsafe overrides, and citationless routing is 15/25 with attribution failures
Saved-prediction readiness gate: best real saved-output compact summary is 0/5 held-out; best fail-closed candidate gate is 2/5 strict final with 0 unsafe trust, below citationless 3/5 and runtime 5/5
Saved-output next decision: targeted_action_status_calibration_probe, with tool_query, DPO/RLVR, HF publication, release tagging, and broad retraining still gated
Saved-output calibration probe: 20 target-vs-ground/supported pairs, split into 16 train-allowed and 4 held-out evaluation-only rows with 0 train/held-out source overlap
Saved-output calibration readout path: dry-run validates train-selected score-gap gating over all 20 probe rows; full model scoring should run on Cayuga/Expanse before interpreting any result
Saved-output calibration readout result: Cayuga Qwen2.5-0.5B has 0/20 exact top-1 and chooses ground/supported in 20/20 probe rows
Saved-output same-model target-format result: full-target flag-focused SFT moves held-out full JSON wins 0/4 -> 4/4 and full flag/invalid_value margin -0.175 -> +0.026; candidate ranking and full-trajectory readiness remain unproven
Saved-output candidate-rank result: trained 5-way candidate top-1 is 1/4; top pair is flag/invalid_value in 4/4 held-out rows despite full teacher-forced margins staying 4/4
Saved-output candidate field diagnostic: trained non-flag rows are both-field failures, so the failure is combined flag/invalid_value over-selection rather than one isolated field miss
Saved-output train-candidate diagnostic: trained train rows top-rank flag/invalid_value in 16/16 cases with 4/16 exact top-1, so the candidate failure is a global focused-objective prior rather than held-out-only noise
Saved-output candidate calibration diagnostic: train-derived pair-mean centering improves held-out candidate top-1 from 1/4 to 2/4, but the train-selected zero-unsafe gate trusts 0/4 held-out rows and reaches only 1/4 strict final with static defer fail-closed routing
Saved-output candidate arbitration: raw candidate top-1 is 1/4, calibrated top-1 is 2/4, train-selected score-gap gate is 1/4, and model-visible evidence/hybrid arbitration are 4/4 on the held-out slice
Saved-output next decision update: targeted calibration is complete; the next model-heavy checkpoint must meet or beat runtime evidence/hybrid arbitration at 4/4 with 0 unsafe candidate trust before `tool_query`, DPO/RLVR, HF publication, release tagging, or broad retraining reopens
Saved-output meet-or-beat gate: current raw, calibrated, and train-selected score-gap candidate policies all fail the reusable 4/4 runtime evidence/hybrid acceptance gate; future compact policies can be built with `build_stage_a_saved_output_policy_summary.py` and passed through `--model-policy-summary` while raw artifacts stay uncommitted
Saved-output intake contract: current compact saved-output bundle passes hash, criteria, future-policy public-safe flag, and adapter provenance checks; next Cayuga policy summary must enter through the adapter and meet-or-beat gate
Saved-output non-flag checkpoint: balanced non-flag oversampling changes the top-pair distribution but remains 1/4 raw and 1/4 calibrated held-out candidate top-1, while runtime evidence/hybrid arbitration stays 4/4
Saved-output candidate-CE checkpoint spec: next model-heavy Cayuga run should use explicit pair+field candidate CE over action/status candidates and still must reach 4/4 held-out exact with zero trusted-candidate incorrect cases before any escalation
Saved-output candidate-CE checkpoint result: explicit pair+field candidate CE shifts raw held-out top pairs to verify/insufficient in 4/4 rows but still remains 1/4 raw and 1/4 calibrated held-out candidate top-1, so runtime evidence/hybrid arbitration remains the baseline
Saved-output post-candidate-CE next decision: both standalone candidate checkpoints remain 1/4 against a 4/4 runtime arbitration reference, so the next checkpoint is an evidence-conditioned saved-output bridge before more standalone SFT, `tool_query`, DPO/RLVR, HF publication, or release tagging
Saved-output evidence bridge: 18/18 compact candidate-failure rows join to prompt-visible evidence-gate rows across 4 unique failed cases; the next data substrate should condition candidate routing on visible evidence features rather than repeat standalone SFT
Saved-output evidence candidate-routing rows: 25 finite-candidate rows preserve the 20/5 split, balance all five action/status pairs, and mark the 4 bridge failure cases as held-out evaluation focus rows before any model-heavy objective
Saved-output evidence candidate-routing readout: runtime evidence routing is 5/5 held-out exact and 4/4 bridge-focus exact, while every static prior is at most 1/5; next prepare a small Cayuga candidate-routing smoke spec
Saved-output evidence candidate-routing smoke spec: next model policy must reach 5/5 held-out exact and 4/4 bridge-focus exact; run the Cayuga dry-run before any full submission
Saved-output evidence candidate-routing smoke runner: dry-run validates the 20/5 split and 4 held-out bridge-focus rows; full model mode requires `--allow-model-load` and writes raw candidates under ignored run paths
Saved-output evidence candidate-routing smoke result adapter: reads compact eval reports only, rejects raw candidate-score/model-text fields, redacts external compact-input paths, and applies the 5/5 held-out plus 4/4 bridge-focus gate
Saved-output evidence candidate-routing Cayuga dry-run checkpoint: mirror dry-run passes at commit 6820498 with 20 train, 5 held-out, 4 bridge-focus held-out rows, and no issues; next decision is explicit full smoke approval or no-submit
Saved-output evidence candidate-routing Cayuga full smoke: Qwen2.5-0.5B selects verify/insufficient for all rows, reaching 4/20 train, 1/5 held-out, and 1/4 bridge-focus exact; freeze this diagnostic slice and build a sealed Stage A evaluation extension
Tool-query placeholder-schema Cayuga smoke: 0/5 held-out pass; all outputs are parseable prompt-schema JSON, but 0/5 contain tool_calls
One-time sealed candidate routing: frozen Qwen2.5-0.5B scores 5/25 and selects verify/insufficient on all 25 rows; static prior is 5/25 and runtime oracle is 25/25
Routing gate arbitration: raw candidate top-1 is 1/2; score-gap fail-closed, evidence-boundary override, and hybrid policies are 2/2 on held-out defer-vs-verify
```

Interpretation: the first real cluster model artifact is a negative baseline.
The strict prompt contract removes parse errors, but the evaluator still
catches missing required tool sequence, missing query fields, evidence/status
routing errors, and terminal-action mismatch. See
`STAGE_A_CAYUGA_HF_CHAT_BASELINE_2026-07-04.md` and
`STAGE_A_CAYUGA_STRICT_CONTRACT_2026-07-04.md` for compact summaries. The first
strict-contract SFT smoke lowered teacher-forced train loss, but did not improve
held-out trajectory gates; see
`STAGE_A_STRICT_SFT_CAYUGA_SMOKE_2026-07-04.md`. The first component-slice
smoke isolates a smaller failure: `enum_action` outputs parse as JSON but use
the disallowed evidence status `valid` and emit an extra `tool` key in all
held-out cases; see `STAGE_A_COMPONENT_ENUM_ACTION_SFT_CAYUGA_2026-07-05.md`.
Finite candidate scoring fixes target-key and enum-validity gates, but exact
enum-pair selection still passes only 1/5; see
`STAGE_A_COMPONENT_ENUM_ACTION_CANDIDATE_SFT_CAYUGA_2026-07-05.md`.
The full-rank rerun keeps all 30 finite candidates and shows the gold pair at
ranks 1, 5, 13, 4, and 24; see
`STAGE_A_COMPONENT_ENUM_ACTION_CANDIDATE_FULLRANK_CAYUGA_2026-07-05.md`.
Field-rank reanalysis shows action top-1 2/5 and evidence-status top-1 1/5.
For invalid-value, the gold pair ranks 24, `flag` action ranks 6, and
`invalid_value` status ranks 2; see
`STAGE_A_COMPONENT_ENUM_ACTION_FIELD_RANK_CAYUGA_2026-07-05.md`.
Restricting the same saved scores to train-observed valid pairs still selects
`ground` / `supported` for all five held-out cases; see
`STAGE_A_COMPONENT_ENUM_ACTION_OBSERVED_PAIR_COUNTERFACTUAL_2026-07-05.md`.
The evidence-conditioned enum/action run repeats that 5-way observed-pair
readout with public-safe evidence state visible in the prompt; it still scores
1/5 and selects `ground` / `supported` for all five held-out cases; see
`STAGE_A_EVIDENCE_ENUM_ACTION_OBSERVED_PAIR_CAYUGA_2026-07-06.md`.
The evidence-conditioned routing-after-loop run exposes observed tool-result
content, but free-form JSON generation still fails all held-out cases with
target-key, enum-validity, target-match, and parse violations; see
`STAGE_A_EVIDENCE_ROUTING_AFTER_LOOP_CAYUGA_2026-07-06.md`.
The constrained routing readout removes schema/enum failure and reaches 2/5
held-out pass with mean score 0.850. The remaining misses are action/status
routing failures for insufficient, verification-needed, and invalid-value
families, not citation-only failures; see
`STAGE_A_EVIDENCE_ROUTING_OBSERVED_PAIR_CAYUGA_2026-07-08.md`.
The follow-up corrective artifact creates train/held-out contrast pairs where
the chosen target is the oracle enum pair and the rejected target is the
observed `ground` / `supported` collapse. This is a data substrate, not a DPO
or RLVR result.
The action-contrast artifact is the next narrower substrate: it preserves the
gold evidence status and changes only the action to `ground`. For the
invalid-value case, the rejected pair is `ground` / `invalid_value`, directly
testing whether the model can learn `flag` action selection.
Use `analyze_stage_a_enum_candidate_scores.py` to publish only compact
gold-rank, margin, and top-candidate bias summaries from ignored raw prediction
artifacts.

Strict-contract component diagnostics:

```bash
python3 post_training/run_stage_a_strict_component_diagnostics.py --compact
```

Current diagnostic baselines:

```text
oracle_full: 5/5 pass, mean score 1.000
invalid_enum_verified: 0/5 pass, mean score 0.000
enum_constrained_from_action: 5/5 pass, mean score 1.000
route_only_correct_no_tools: 0/5 pass, mean score 0.714
ordered_tool_names_only: 0/5 pass, mean score 0.857
tool_loop_with_ground_route: 1/5 pass, mean score 0.714
```

Interpretation: a high partial score is still not a passing trajectory. Ordered
tool names without required query fields reach 0.857 mean score but fail every
held-out case, while a full structured tool loop still fails if the route
collapses to `ground`/`supported`. See
`STAGE_A_STRICT_COMPONENT_DIAGNOSTICS_2026-07-04.md`.

Strict-contract component targets:

```bash
python3 post_training/export_stage_a_strict_component_targets.py
python3 post_training/validate_post_training_data.py
```

These rows turn the diagnostic readout into train/eval slices:

- `enum_action`: predict only `action` and `evidence_status`;
- `tool_query`: predict ordered tool calls with `drug_id` and `condition_id`
  arguments;
- `routing_after_loop`: predict `action`, `evidence_status`, and citations
  after the valid tool loop is present.

Strict component SFT smoke:

```bash
python3 post_training/run_stage_a_strict_component_sft_smoke.py \
  --dry-run \
  --component enum_action
python3 post_training/run_stage_a_strict_component_sft_smoke.py \
  --dry-run \
  --component tool_query
python3 post_training/run_stage_a_strict_component_sft_smoke.py \
  --dry-run \
  --component routing_after_loop
```

The dry-run path validates the selected slice without loading model weights. A
full run requires `--allow-model-load`, trains a tiny last-layer smoke checkpoint
for the selected component, writes held-out component predictions, and scores
exact key, enum, structured tool-query, and target-match gates.

For the repaired `enum_action` path, use
`--decode-mode enum_candidate_score`. This scores the finite set of valid
`(action, evidence_status)` JSON candidates instead of letting the model emit
free-form JSON with extra keys or invalid enum values.
Use `--decode-mode enum_observed_pair_score` to restrict the same scorer to
train-observed valid target pairs. The current counterfactual shows this pruning
alone still collapses to `ground` / `supported`.

For the repaired evidence-conditioned `routing_after_loop` readout, use
`--decode-mode routing_observed_pair_score`. This scores train-observed
`(action, evidence_status)` routing pairs and attaches `cited_source_ids` from
prompt-visible tool-result content only. It is a readout for separating
schema/enum/citation failures from free-form JSON generation, not a hidden-label
repair. After a cluster run, use `analyze_stage_a_routing_candidate_scores.py`
to publish only compact full-target rank, action/status rank, citation-failure,
and top-candidate summaries from ignored raw prediction artifacts.

No-model component saved-prediction baselines:

```bash
python3 post_training/generate_stage_a_component_predictions.py \
  --component routing_after_loop \
  --mode majority_ground_supported \
  --out-dir /tmp/stage_a_component_routing_ground_supported
python3 post_training/generate_stage_a_component_predictions.py \
  --component routing_after_loop \
  --mode routing_no_citations \
  --out-dir /tmp/stage_a_component_routing_no_citations
python3 post_training/generate_stage_a_component_predictions.py \
  --component tool_query \
  --mode tool_names_only \
  --out-dir /tmp/stage_a_component_tool_names_only
```

These baselines write saved prediction JSONL plus component eval reports without
loading model weights. They keep future API or cluster component outputs
comparable to deterministic oracle/collapse/citation/tool-shape gates.

Cluster strict component SFT smoke:

```bash
sbatch --account=<allocation> --partition=<gpu-partition> --gres=gpu:1 \
  --export=ALL,WORK=$PWD,COMPONENT=enum_action,DECODE_MODE=enum_observed_pair_score,MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct \
  post_training/run_stage_a_strict_component_sft_cayuga.sbatch

sbatch --account=<allocation> \
  --export=ALL,WORK=$PWD,COMPONENT=enum_action,DECODE_MODE=enum_observed_pair_score,MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct \
  post_training/run_stage_a_strict_component_sft_expanse.sbatch
```

Run `COMPONENT=enum_action` first, then `tool_query`, then
`routing_after_loop`. Raw reports stay under `post_training/runs/`; only compact
public-safe summaries should be curated into the repository.

Enum corrective SFT/margin smoke:

```bash
python3 post_training/run_stage_a_enum_corrective_sft_smoke.py --dry-run
```

The dry-run path validates the 16 train and 4 held-out enum corrective pairs
without loading model weights. A full run requires `--allow-model-load`, trains
only on chosen enum/action corrective targets, then scores held-out
chosen-vs-rejected likelihood margins. This is an SFT diagnostic against the
observed `ground` / `supported` collapse, not DPO/RLVR.

For the next diagnosis, add `--score-base-margins` to score held-out margins
before training and write `margin_delta_report.json`. Add
`--score-train-margins` when you need to check whether the tiny SFT run learned
the train-pair contrasts before interpreting held-out failures.
Use `--focus-chosen-pairs flag/invalid_value,defer/insufficient --focus-repeat 4`
to oversample the weak enum families during training while keeping train-margin
evaluation on the original 16 unique train pairs. When passing through Slurm
`--export`, use `FOCUS_CHOSEN_PAIRS=flag/invalid_value:defer/insufficient`
because Slurm uses commas to separate exported variables.

Cluster enum corrective SFT/margin smoke:

```bash
sbatch --account=<allocation> --partition=<gpu-partition> --gres=gpu:1 \
  --export=ALL,WORK=$PWD,MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct,SCORE_BASE_MARGINS=1,SCORE_TRAIN_MARGINS=1,FOCUS_CHOSEN_PAIRS=flag/invalid_value:defer/insufficient,FOCUS_REPEAT=4 \
  post_training/run_stage_a_enum_corrective_sft_cayuga.sbatch

sbatch --account=<allocation> \
  --export=ALL,WORK=$PWD,MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct,SCORE_BASE_MARGINS=1,SCORE_TRAIN_MARGINS=1,FOCUS_CHOSEN_PAIRS=flag/invalid_value:defer/insufficient,FOCUS_REPEAT=4 \
  post_training/run_stage_a_enum_corrective_sft_expanse.sbatch
```

Raw `report.json`, `margins.jsonl`, and `margin_report.json` outputs stay under
`post_training/runs/`. Curate only compact public-safe summaries after the
Cayuga result is inspected.

Routing contrast SFT/margin smoke:

```bash
python3 post_training/run_stage_a_routing_contrast_sft_smoke.py \
  --dry-run \
  --pairwise-margin-weight 1 \
  --pairwise-margin 0.05 \
  --score-base-routing-candidates \
  --score-trained-routing-candidates

sbatch --account=<allocation> --partition=<gpu-partition> --gres=gpu:1 \
  --export=ALL,WORK=$PWD,MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct,SCORE_BASE_MARGINS=1,SCORE_TRAIN_MARGINS=1,SCORE_BASE_ROUTING_CANDIDATES=1,SCORE_TRAINED_ROUTING_CANDIDATES=1,PAIRWISE_MARGIN_WEIGHT=1,PAIRWISE_MARGIN=0.05 \
  post_training/run_stage_a_routing_contrast_sft_cayuga.sbatch

sbatch --account=<allocation> \
  --export=ALL,WORK=$PWD,MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct,SCORE_BASE_MARGINS=1,SCORE_TRAIN_MARGINS=1,SCORE_BASE_ROUTING_CANDIDATES=1,SCORE_TRAINED_ROUTING_CANDIDATES=1,PAIRWISE_MARGIN_WEIGHT=1,PAIRWISE_MARGIN=0.05 \
  post_training/run_stage_a_routing_contrast_sft_expanse.sbatch
```

This runner trains on 12 unique routing contrast train pairs and scores 3
held-out chosen-vs-rejected margins. It is the Cayuga diagnostic for
insufficient, verification-needed, and invalid-value action/status routing; it
is not DPO/RLVR and does not score explanation quality. With the routing
candidate flags enabled, it also scores train-observed 5-way routing candidate
ranks before and after training, writing raw candidate JSONL and compact rank
reports under ignored `post_training/runs/`.

Current routing contrast result: base held-out margins were 0/3 wins with mean
margin -0.116919; trained held-out margins were 3/3 wins with mean margin
0.114900 and mean delta 0.231818. This is positive teacher-forced component
evidence, but it does not yet prove finite-candidate routing, free-form JSON
generation, `tool_query`, or full trajectory repair. See
`STAGE_A_ROUTING_CONTRAST_SFT_CAYUGA_2026-07-08.md`.

Current routing candidate-rank result: finite-candidate exact top-1 improves
from 0/3 to 2/3 and mean gold rank improves from 3.0 to 1.333333. The remaining
failure is `defer` / `insufficient`, which moves from rank 5 to rank 2 but
still loses to `verify` / `insufficient`. See
`STAGE_A_ROUTING_CONTRAST_CANDIDATE_CAYUGA_2026-07-08.md`.

Saved-output calibration margin SFT:

```bash
python3 post_training/run_stage_a_saved_output_calibration_margin_sft.py \
  --dry-run \
  --target-format full \
  --score-target-formats full:action_only:action_status_only:status_only \
  --pairwise-margin-weight 1 \
  --pairwise-margin 0.05 \
  --score-base-margins \
  --score-train-margins \
  --score-base-candidates \
  --score-train-candidates \
  --score-trained-candidates \
  --candidate-policy train_observed_plus_rejected

python3 post_training/run_stage_a_saved_output_calibration_margin_sft.py \
  --dry-run \
  --focus-chosen-pairs flag/invalid_value \
  --focus-repeat 4 \
  --focus-only \
  --target-format action_only \
  --pairwise-margin-weight 1 \
  --pairwise-margin 0.05 \
  --score-base-margins \
  --score-train-margins

sbatch --account=<allocation> --partition=<gpu-partition> --gres=gpu:1 \
  --export=ALL,WORK=$PWD,MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct,SCORE_BASE_MARGINS=1,SCORE_TRAIN_MARGINS=1,SCORE_BASE_CANDIDATES=1,SCORE_TRAIN_CANDIDATES=1,SCORE_TRAINED_CANDIDATES=1,CANDIDATE_POLICY=train_observed_plus_rejected,TARGET_FORMAT=full,SCORE_TARGET_FORMATS=full:action_only:action_status_only:status_only,FOCUS_CHOSEN_PAIRS=flag/invalid_value,FOCUS_REPEAT=4,FOCUS_ONLY=1,PAIRWISE_MARGIN_WEIGHT=1,PAIRWISE_MARGIN=0.05 \
  post_training/run_stage_a_saved_output_calibration_margin_sft_cayuga.sbatch

sbatch --account=<allocation> \
  --export=ALL,WORK=$PWD,MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct,SCORE_BASE_MARGINS=1,SCORE_TRAIN_MARGINS=1,SCORE_BASE_CANDIDATES=1,SCORE_TRAIN_CANDIDATES=1,SCORE_TRAINED_CANDIDATES=1,CANDIDATE_POLICY=train_observed_plus_rejected,TARGET_FORMAT=full,SCORE_TARGET_FORMATS=full:action_only:action_status_only:status_only,FOCUS_CHOSEN_PAIRS=flag/invalid_value,FOCUS_REPEAT=4,FOCUS_ONLY=1,PAIRWISE_MARGIN_WEIGHT=1,PAIRWISE_MARGIN=0.05 \
  post_training/run_stage_a_saved_output_calibration_margin_sft_expanse.sbatch
```

This runner trains only on the 16 train-allowed saved-output calibration probe
pairs and evaluates the 4 held-out probe pairs by target-vs-`ground`/`supported`
teacher-forced margin. Optional candidate scoring ranks finite held-out outputs
from train-observed target pairs plus the `ground` / `supported` collapse.
Use `--score-train-candidates` to write train-side candidate scores for
calibration diagnostics without tuning on held-out candidate ranks. Raw
`report.json`, `margins.jsonl`, `train_candidates.jsonl`, `candidates.jsonl`,
trainable state, and scheduler logs stay under ignored `post_training/runs/`;
curate only compact public-safe summaries after Cayuga/Expanse results are
inspected. This is a targeted SFT diagnostic, not DPO/RLVR and not a
release-readiness claim.
Use `analyze_stage_a_saved_output_candidate_fields.py` on ignored
`candidates.jsonl` files when candidate top-1 fails; it reports action/status
field ranks and top-pair bias without committing raw candidate-score tables.
Use `--target-format action_only`, `status_only`, or `action_status_only` to
separate action/status learning from full JSON, citation, and tool-call target
formatting. This target projection changes the scoring surface, so compare
only reports with the same `target_format`.
Use `--score-target-formats full:action_only:action_status_only:status_only`
to score several projections on the same base and trained model. Extra reports
are written as suffixed ignored files such as
`margin_report_action_only.json`; only compact summaries should be curated.

Current saved-output calibration margin result: Cayuga Qwen2.5-0.5B improves
held-out target-vs-collapse margin wins from 0/4 to 1/4 and mean margin from
-0.081001 to -0.045619. All held-out families move in the positive direction,
but only `verify` / `insufficient` crosses zero; train split wins are 4/16. See
`STAGE_A_SAVED_OUTPUT_CALIBRATION_MARGIN_SFT_QWEN05B_CAYUGA_2026-07-10.md`.

Focused follow-up: oversampling the three still-negative non-verify families
improves held-out wins from 0/4 to 3/4 and mean margin from -0.081001 to
-0.002999. This is stronger movement but still not a repair because
`flag` / `invalid_value` remains below `ground` / `supported` on train and
held-out scoring. See
`STAGE_A_SAVED_OUTPUT_CALIBRATION_MARGIN_SFT_FOCUS_NONVERIFY3_QWEN05B_CAYUGA_2026-07-10.md`.

Next target-format diagnostic: run `flag` / `invalid_value` with
`--focus-only --target-format action_only` first. If action-only crosses the
held-out boundary while action+status or full JSON does not, the unresolved
failure is likely target coupling rather than the `flag` action token alone.

Target-format result: `action_only` moves the held-out `flag` margin from
-0.812072 to +0.842364, `action_status_only` moves `flag` / `invalid_value`
from -0.282855 to +0.436716, and `status_only` is already positive for
`invalid_value` before training. The focused full-JSON reference remains
negative for `flag` / `invalid_value` (-0.103098 after SFT), so the remaining
failure is full target coupling rather than isolated action/status label
learning. See
`STAGE_A_SAVED_OUTPUT_TARGET_FORMAT_FLAG_QWEN05B_CAYUGA_2026-07-10.md`.

Current same-model diagnostic: training with `TARGET_FORMAT=full` and
`FOCUS_CHOSEN_PAIRS=flag/invalid_value` repairs the teacher-forced held-out
full-output margins from 0/4 to 4/4, but finite-candidate ranking remains
weak. Held-out candidate exact top-1 is 1/4 and all four held-out rows top-rank
`flag` / `invalid_value`. Train-side candidate scoring shows the same bias:
train exact top-1 is 4/16 and all 16 train rows top-rank
`flag` / `invalid_value`. See
`STAGE_A_SAVED_OUTPUT_TRAIN_CANDIDATE_QWEN05B_CAYUGA_2026-07-10.md`.
Train-derived pair-mean centering partially reduces this prior: held-out exact
top-1 moves from 1/4 to 2/4, but the `flag` / `invalid_value` held-out row is
lost to `ground` / `supported`. This is calibration signal, not a trust-ready
policy. See
`STAGE_A_SAVED_OUTPUT_CANDIDATE_CALIBRATION_QWEN05B_CAYUGA_2026-07-10.md`.
When the threshold is selected from calibrated train rows to avoid unsafe trust,
held-out coverage drops to 0/4 and strict final correctness is only 1/4 with a
static `defer` / `insufficient` fail-closed default. The next useful gate needs
runtime evidence arbitration, not score-gap calibration alone.
Saved-output candidate arbitration confirms that boundary: raw candidate top-1
is 1/4, calibrated top-1 is 2/4, and train-selected score-gap gating is 1/4,
while model-visible evidence and hybrid evidence-then-gate policies are 4/4.
This keeps runtime enforcement as the system baseline before any new optimizer.
The updated next-decision checkpoint therefore closes the targeted calibration
probe loop: future model-heavy Cayuga outputs must meet or beat the 4/4 runtime
evidence/hybrid arbitration baseline with 0 unsafe candidate trust before
`tool_query`, DPO/RLVR, Hugging Face publication, release tagging, or broad
retraining reopens.
The meet-or-beat gate makes that decision executable: current raw,
calibrated, and train-selected score-gap candidate policies all fail, while
the runtime evidence/hybrid policies define the compact acceptance baseline for
future Cayuga summaries. Future compact summaries can be converted with
`build_stage_a_saved_output_policy_summary.py`, then passed through
`--model-policy-summary` only when the summary carries adapter provenance:
dataset, source kind, repo-relative public-manifest source report,
source-report SHA-256, explicit public-safety contract, `policy`, `exact`, `rows`,
`trusted_candidate_incorrect`, and raw-artifact commit flags for raw
predictions, candidate-score JSONL, eval reports, scheduler logs, and model
state. The gate also rejects impossible or non-comparable count summaries:
negative counts, `exact > rows`, trust counts above rows, row counts that do
not match the runtime baseline slice, and JSON count fields encoded as strings,
floats, or booleans. Public QA now also routes the adapter smoke output back
through the meet-or-beat gate, so provenance-contract regressions fail CI
instead of stopping at JSON generation.

Current defer-vs-verify boundary substrate: 10 pairs, with 8 train and 2
held-out examples. Insufficient-evidence rows choose `defer` / `insufficient`
over `verify` / `insufficient`; verification-needed rows choose the reverse.
This is the next targeted Cayuga smoke substrate, not a result.

Current defer-vs-verify Cayuga result: base and trained finite-candidate exact
top-1 both remain 1/2. The `defer` / `insufficient` margin improves from
-0.154711 to -0.023557, but still does not cross the boundary; the trained top
prediction is `verify` / `insufficient` for both held-out cases. See
`STAGE_A_ROUTING_DEFER_VERIFY_CAYUGA_2026-07-08.md`.

Current fail-closed gate diagnostic: over the trained 2-case held-out boundary
slice, threshold 0.025 trusts the high-gap `verify` / `insufficient` row and
fails closed to `defer` / `insufficient` on the low-gap wrong row. This gives
2/2 strict final correctness and 0 unsafe trusted rows on this tiny slice, but
it is not deployment calibration. See
`STAGE_A_ROUTING_DEFER_VERIFY_GATE_TRAINED_2026-07-08.md`.

Current evidence-boundary gate result: a no-model rule using only
model-visible `observed_tool_loop` fields reaches 10/10 overall, 8/8 train, and
2/2 held-out on the defer-vs-verify boundary. It routes rows with no same- or
related-indication evidence to `defer` / `insufficient`, and rows with related
evidence but no same-indication record to `verify` / `insufficient`. See
`STAGE_A_ROUTING_EVIDENCE_BOUNDARY_GATE_2026-07-08.md`.

Current routing arbitration result: raw trained candidate top-1 is 1/2 on the
held-out defer-vs-verify boundary, while score-gap fail-closed,
evidence-boundary override, and hybrid evidence-then-score policies are all
2/2. This makes runtime enforcement the system baseline before any new
training objective. See `STAGE_A_ROUTING_GATE_ARBITRATION_2026-07-08.md`.

Current Cayuga corrective result: 2/4 held-out margin wins, mean margin
-0.020685. The run creates partial signal for contradicted and verification
cases, but insufficient-evidence and invalid-value cases still lose to
`ground` / `supported`. See `STAGE_A_ENUM_CORRECTIVE_SFT_CAYUGA_2026-07-05.md`.

Current margin-delta result: base held-out wins were 0/4, trained held-out wins
were 2/4, and all four held-out families moved in the right direction. The
invalid-value family still fails even on trained train-pair margins. See
`STAGE_A_ENUM_CORRECTIVE_MARGIN_DELTA_CAYUGA_2026-07-05.md`.

Current targeted sampling result: oversampling `flag` / `invalid_value` and
`defer` / `insufficient` improves mean held-out margin from -0.020685 to
-0.008175, but held-out wins drop from 2/4 to 1/4 and invalid-value remains
0/4 on train-pair margins. See
`STAGE_A_ENUM_CORRECTIVE_TARGETED_CAYUGA_2026-07-05.md`.

Current field-rank diagnosis: the invalid-value held-out case is not primarily a
missing status-label exposure problem. `invalid_value` is rank 2 by field, but
`flag` is rank 6, so the next repair should target `flag` action representation
and evidence-conditioned action selection before any DPO/RLVR escalation.

Action-contrast dry run:

```bash
python3 post_training/export_stage_a_enum_action_contrast_pairs.py
python3 post_training/run_stage_a_enum_corrective_sft_smoke.py --dry-run \
  --pairs post_training/stage_a_enum_action_contrast_pairs_v1.jsonl \
  --train-pairs post_training/stage_a_enum_action_contrast_pairs_train_v1.jsonl \
  --heldout-pairs post_training/stage_a_enum_action_contrast_pairs_heldout_v1.jsonl \
  --manifest post_training/stage_a_enum_action_contrast_pairs_manifest.json \
  --focus-chosen-pairs flag/invalid_value --focus-repeat 4
```

Cayuga action-contrast margin smoke:

```bash
sbatch --account=<allocation> --partition=<gpu-partition> --gres=gpu:1 \
  --export=ALL,WORK=$PWD,MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct,SCORE_BASE_MARGINS=1,SCORE_TRAIN_MARGINS=1,PAIRS=post_training/stage_a_enum_action_contrast_pairs_v1.jsonl,TRAIN_PAIRS=post_training/stage_a_enum_action_contrast_pairs_train_v1.jsonl,HELDOUT_PAIRS=post_training/stage_a_enum_action_contrast_pairs_heldout_v1.jsonl,MANIFEST=post_training/stage_a_enum_action_contrast_pairs_manifest.json,FOCUS_CHOSEN_PAIRS=flag/invalid_value,FOCUS_REPEAT=4 \
  post_training/run_stage_a_enum_corrective_sft_cayuga.sbatch
```

Current action-contrast result: held-out wins improve from 1/4 base to 2/4
trained and mean held-out margin improves from -0.079422 to -0.000604, but
`flag` / `invalid_value` and `defer` / `insufficient` still lose to
`ground`. See `STAGE_A_ENUM_ACTION_CONTRAST_CAYUGA_2026-07-05.md`.

Action-only target-format dry run:

```bash
python3 post_training/run_stage_a_enum_corrective_sft_smoke.py --dry-run \
  --pairs post_training/stage_a_enum_action_contrast_pairs_v1.jsonl \
  --train-pairs post_training/stage_a_enum_action_contrast_pairs_train_v1.jsonl \
  --heldout-pairs post_training/stage_a_enum_action_contrast_pairs_heldout_v1.jsonl \
  --manifest post_training/stage_a_enum_action_contrast_pairs_manifest.json \
  --focus-chosen-pairs flag/invalid_value --focus-repeat 4 \
  --target-format action_only
```

Cayuga action-only target-format diagnostic:

```bash
sbatch --account=<allocation> --partition=<gpu-partition> --gres=gpu:1 \
  --export=ALL,WORK=$PWD,MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct,SCORE_BASE_MARGINS=1,SCORE_TRAIN_MARGINS=1,TARGET_FORMAT=action_only,PAIRS=post_training/stage_a_enum_action_contrast_pairs_v1.jsonl,TRAIN_PAIRS=post_training/stage_a_enum_action_contrast_pairs_train_v1.jsonl,HELDOUT_PAIRS=post_training/stage_a_enum_action_contrast_pairs_heldout_v1.jsonl,MANIFEST=post_training/stage_a_enum_action_contrast_pairs_manifest.json,FOCUS_CHOSEN_PAIRS=flag/invalid_value,FOCUS_REPEAT=4 \
  post_training/run_stage_a_enum_corrective_sft_cayuga.sbatch
```

Current action-only result: removing `evidence_status` from the target improves
mean held-out margin from -0.211307 to -0.032852, but held-out wins stay 1/4
and `flag`, `defer`, and `reject` train margins remain below zero. See
`STAGE_A_ENUM_ACTION_ONLY_CAYUGA_2026-07-05.md`.

Pairwise-margin action objective dry run:

```bash
python3 post_training/run_stage_a_enum_corrective_sft_smoke.py --dry-run \
  --pairs post_training/stage_a_enum_action_contrast_pairs_v1.jsonl \
  --train-pairs post_training/stage_a_enum_action_contrast_pairs_train_v1.jsonl \
  --heldout-pairs post_training/stage_a_enum_action_contrast_pairs_heldout_v1.jsonl \
  --manifest post_training/stage_a_enum_action_contrast_pairs_manifest.json \
  --focus-chosen-pairs flag/invalid_value --focus-repeat 4 \
  --target-format action_only \
  --pairwise-margin-weight 1 --pairwise-margin 0.05
```

Cayuga pairwise-margin action diagnostic:

```bash
sbatch --account=<allocation> --partition=<gpu-partition> --gres=gpu:1 \
  --export=ALL,WORK=$PWD,MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct,SCORE_BASE_MARGINS=1,SCORE_TRAIN_MARGINS=1,TARGET_FORMAT=action_only,PAIRWISE_MARGIN_WEIGHT=1,PAIRWISE_MARGIN=0.05,PAIRS=post_training/stage_a_enum_action_contrast_pairs_v1.jsonl,TRAIN_PAIRS=post_training/stage_a_enum_action_contrast_pairs_train_v1.jsonl,HELDOUT_PAIRS=post_training/stage_a_enum_action_contrast_pairs_heldout_v1.jsonl,MANIFEST=post_training/stage_a_enum_action_contrast_pairs_manifest.json,FOCUS_CHOSEN_PAIRS=flag/invalid_value,FOCUS_REPEAT=4 \
  post_training/run_stage_a_enum_corrective_sft_cayuga.sbatch
```

Current pairwise-margin result: action-only held-out margins improve from 1/4
base to 4/4 trained wins, with mean held-out margin improving from -0.211307 to
0.261516. This repairs the action-only chosen-over-`ground` slice, not the full
action-plus-status enum component. See
`STAGE_A_ENUM_PAIRWISE_MARGIN_CAYUGA_2026-07-05.md`.

Current full pairwise-margin result: full action-plus-status held-out margins
improve from 1/4 to 4/4 wins on same-status action contrasts and from 0/4 to
4/4 wins on the broader `ground` / `supported` corrective contrast. This is a
teacher-forced component-margin result, not a free-generation or full
trajectory pass. See `STAGE_A_ENUM_FULL_PAIRWISE_MARGIN_CAYUGA_2026-07-05.md`.

Finite-candidate pairwise-margin readout dry run:

```bash
python3 post_training/run_stage_a_enum_corrective_sft_smoke.py --dry-run \
  --focus-chosen-pairs flag/invalid_value --focus-repeat 4 \
  --target-format full \
  --pairwise-margin-weight 1 --pairwise-margin 0.05 \
  --candidate-ce-weight 1 --candidate-ce-mode field \
  --score-base-enum-candidates --score-enum-candidates \
  --enum-candidate-policy pair_observed_outputs
```

Cayuga finite-candidate selection objective:

```bash
sbatch --account=<allocation> --partition=<gpu-partition> --gres=gpu:1 \
  --export=ALL,WORK=$PWD,MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct,SCORE_BASE_MARGINS=1,SCORE_TRAIN_MARGINS=1,SCORE_BASE_ENUM_CANDIDATES=1,SCORE_ENUM_CANDIDATES=1,ENUM_CANDIDATE_POLICY=pair_observed_outputs,TARGET_FORMAT=full,PAIRWISE_MARGIN_WEIGHT=1,PAIRWISE_MARGIN=0.05,CANDIDATE_CE_WEIGHT=1,CANDIDATE_CE_MODE=field,FOCUS_CHOSEN_PAIRS=flag/invalid_value,FOCUS_REPEAT=4 \
  post_training/run_stage_a_enum_corrective_sft_cayuga.sbatch
```

This next run trains directly on the finite-candidate enum selection failure.
Raw candidate-score tables stay under ignored `post_training/runs/`.

Current candidate readout result: pairwise-margin SFT improves rank but not
top-1 selection. The all-valid 30-way policy stays 0/4 top-1 while mean gold
rank improves from 15.0 to 7.0; the pair-observed 5-way policy stays 0/4 top-1
while mean gold rank improves from 3.5 to 2.75. See
`STAGE_A_ENUM_CANDIDATE_READOUT_CAYUGA_2026-07-05.md`.

Current candidate-CE result: adding the 5-way finite-candidate CE objective
keeps teacher-forced margin wins at 4/4 and improves candidate top-1 from 0/4
to 1/4, with mean gold rank 3.5 -> 2.5. This is still negative/partial method
evidence: keep `tool_query`, DPO, and RLVR gated until candidate calibration or
a constrained candidate gate repairs top-1 more decisively. See
`STAGE_A_ENUM_CANDIDATE_CE_PAIR_OBSERVED_CAYUGA_2026-07-05.md`.

Current candidate gate result: score-gap gating over the candidate-CE run cannot
trust any held-out row without risking false trust. Thresholds that trust rows
also trust incorrect top-1 outputs; the zero-false-trust threshold fails closed
on 4/4. See `STAGE_A_ENUM_CANDIDATE_CE_GATE_CAYUGA_2026-07-05.md`.

Next enum repair hook: `--candidate-ce-mode field` trains action and
`evidence_status` marginals separately over the finite candidate set. This is a
supervised component-slice probe for the observed `both_field_failure` pattern,
not preference optimization or RLVR.

Current field-CE result: factorized action/status CE preserves 4/4
teacher-forced margin wins but does not improve candidate top-1 beyond 1/4 or
produce useful zero-false-trust coverage. See
`STAGE_A_ENUM_FIELD_CE_PAIR_OBSERVED_CAYUGA_2026-07-05.md`.

Strict-contract training artifact:

```bash
python3 post_training/export_stage_a_strict_contract_data.py
python3 post_training/validate_post_training_data.py
python3 post_training/run_stage_a_strict_contract_sft_smoke.py --dry-run
```

The strict target does not repair model output after generation. It creates a
trainable target family for the exact compact JSON contract used by API/HPC
saved-prediction runs, with rejected pairs that isolate the observed
`verify`/`supported` collapse from the Cayuga strict baseline.

Cluster strict-contract SFT smoke:

```bash
sbatch --account=<allocation> --partition=<gpu-partition> --gres=gpu:1 \
  --export=ALL,WORK=$PWD,MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct \
  post_training/run_stage_a_strict_sft_cayuga.sbatch

sbatch --account=<allocation> \
  --export=ALL,WORK=$PWD,MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct \
  post_training/run_stage_a_strict_sft_expanse.sbatch
```

These jobs train a tiny last-layer smoke checkpoint on
`stage_a_strict_contract_sft_train_v1.jsonl`, generate held-out saved
predictions for `stage_a_strict_contract_sft_heldout_v1.jsonl`, and score them
with `evaluate_stage_a_predictions.py`. Raw job outputs stay under
`post_training/runs/`.

## Current Artifact

Dataset: `negbiodb_ct_native_trajectory_v1`

Generated from:

```bash
python3 negbiodb_ct/export_post_training_data.py \
  --runner negbiodb_ct/agent_sonnet_n40_mixedfix.json
```

Inputs:

- `negbiodb_ct/tasks_pilot.jsonl`
- ignored live runner artifact: `negbiodb_ct/agent_sonnet_n40_mixedfix.json`
- local NegBioDB-CT SQLite database used to reconstruct tool observations

Outputs:

| path | rows | role |
| --- | ---: | --- |
| `negbiodb_ct_native_sft_v1.jsonl` | 40 | Supervised full native tool trajectories: prompt, tool calls, tool observations, final `submit_decision`. |
| `negbiodb_ct_native_sft_train_v1.jsonl` | 30 | Deterministic stratified SFT train split: 6 examples per action class. |
| `negbiodb_ct_native_sft_heldout_v1.jsonl` | 10 | Deterministic stratified held-out split: 2 examples per action class. |
| `negbiodb_ct_native_sft_split_manifest.json` | 1 | Split seed, counts, class balance, and task IDs. |
| `cv/` | 8 files | Four deterministic stratified train/held-out folds over the same 40 native SFT examples. |
| `negbiodb_ct_native_sft_cv4_manifest.json` | 1 | Cross-validation seed, fold counts, held-out coverage, and task IDs. |
| `negbiodb_ct_oracle_sft_v1.jsonl` | 400 | Larger deterministic-oracle SFT artifact over the full CT pilot task set. |
| `negbiodb_ct_oracle_sft_manifest.json` | 1 | Oracle SFT provenance, boundary, class counts, and skipped rows. |
| `negbiodb_ct_oracle_sft_balanced_v1.jsonl` | 700 | Class-balanced oracle SFT artifact for the next pressure-SFT run. |
| `negbiodb_ct_oracle_sft_balanced_manifest.json` | 1 | Balanced oracle SFT provenance, seed, and class counts. |
| `pressure/` | 4 files | Native CV pressure train folds with `flag` and `verify` oversampled. |
| `negbiodb_ct_native_sft_cv4_pressure_manifest.json` | 1 | Native pressure CV fold train paths, multipliers, and unchanged held-out paths. |
| `curriculum/` | 4 files | Native CV curriculum train folds with ordered contrast-family blocks. |
| `negbiodb_ct_native_sft_cv4_curriculum_manifest.json` | 1 | Curriculum CV fold train paths, contrast families, and unchanged held-out paths. |
| `curriculum_v2/` | 4 files | Targeted curriculum-v2 train folds with persistent-failure boosts. |
| `negbiodb_ct_native_sft_cv4_curriculum_v2_manifest.json` | 1 | Curriculum-v2 fold train paths, target weights, and unchanged held-out paths. |
| `boundary_rationale/` | 4 files | Boundary-rationale paired train folds with base rows plus prompt-side rationale rows. |
| `negbiodb_ct_native_sft_cv4_boundary_rationale_manifest.json` | 1 | Boundary-rationale fold train paths, boundary negatives, and unchanged held-out paths. |
| `boundary_rationale_heldout_ablation/` | 4 files | Held-out oracle-rationale ablation folds for eval-only rationale-at-inference tests. |
| `negbiodb_ct_native_sft_cv4_boundary_rationale_heldout_ablation_manifest.json` | 1 | Held-out oracle-rationale ablation fold paths and provenance. |
| `boundary_rationale_heldout_evidence_ablation/` | 4 files | Held-out evidence-rationale ablation folds generated from visible tool observations. |
| `negbiodb_ct_native_sft_cv4_boundary_rationale_heldout_evidence_ablation_manifest.json` | 1 | Held-out evidence-rationale ablation fold paths, mismatch counts, and provenance. |
| `negbiodb_ct_native_sft_evidence_rationale_v1.jsonl` | 40 | Full native SFT artifact with deployable evidence-derived rationale messages. |
| `negbiodb_ct_native_sft_evidence_rationale_manifest.json` | 1 | Evidence-rationale layer provenance, action counts, and match/mismatch counts. |
| `negbiodb_ct_oracle_sft_evidence_rationale_v1.jsonl` | 400 | Full pilot oracle-policy SFT artifact with evidence-derived rationale messages for stress-testing. |
| `negbiodb_ct_oracle_sft_evidence_rationale_manifest.json` | 1 | Pilot-400 evidence-rationale stress-test provenance and match counts. |
| `generative_rationale/` | 1 file | Fold0 held-out artifact that targets generated evidence rationale plus final decision JSON. |
| `negbiodb_ct_oracle_sft_generative_rationale_v1.jsonl` | 400 | Full pilot oracle-policy SFT artifact with target-side generated evidence rationales. |
| `negbiodb_ct_oracle_sft_generative_rationale_manifest.json` | 1 | Generative evidence-rationale train artifact provenance and match counts. |
| `negbiodb_ct_native_sft_cv4_generative_rationale_fold0_manifest.json` | 1 | Fold0 generative evidence-rationale held-out artifact provenance and match counts. |
| `negbiodb_ct_oracle_boundary_preferences_v1.jsonl` | 620 | Evidence-boundary preference pairs over fixed native CT tool observations. |
| `negbiodb_ct_oracle_boundary_preferences_manifest.json` | 1 | Boundary preference pair counts, chosen/rejected action counts, and validation summary. |
| `negbiodb_ct_oracle_boundary_preferences_hard_v1.jsonl` | 240 | Negative-margin hard-mode subset of the boundary preference pairs. |
| `negbiodb_ct_oracle_boundary_preferences_hard_manifest.json` | 1 | Hard-mode preference subset counts, selected modes, and validation summary. |
| `negbiodb_ct_oracle_boundary_preferences_hard_train_v1.jsonl` | 208 | Deterministic train split of hard boundary preference pairs. |
| `negbiodb_ct_oracle_boundary_preferences_hard_heldout_v1.jsonl` | 32 | Deterministic held-out split of hard boundary preference pairs. |
| `negbiodb_ct_oracle_boundary_preferences_hard_split_manifest.json` | 1 | Hard preference split seed, source IDs, action counts, and failure-mode balance. |
| `negbiodb_ct_native_preferences_v1.jsonl` | 64 | DPO-style pairs comparing a clean trajectory/final decision against deterministic failure modes. |
| `negbiodb_ct_native_manifest.json` | 1 | Counts, source runner summary, and provenance. |
| `validate_post_training_data.py` | - | Local validator for counts, hidden-key leakage, final decision shape, and pair pass/fail direction. |
| `split_sft_data.py` | - | Builds deterministic stratified train/held-out splits from the tracked SFT examples. |
| `build_sft_cv_splits.py` | - | Builds deterministic stratified cross-validation folds from the tracked SFT examples. |
| `export_oracle_sft_data.py` | - | Exports full-pilot deterministic-oracle SFT examples using native CT tool observations. |
| `build_sft_pressure_data.py` | - | Builds pressure SFT artifacts from the row-level failure diagnosis. |
| `analyze_sft_pressure_failures.py` | - | Analyzes pressure-run row-level failures and candidate ranks. |
| `build_sft_curriculum_data.py` | - | Builds contrast-family curriculum SFT artifacts from the pressure-failure diagnosis. |
| `build_sft_curriculum_v2_data.py` | - | Builds targeted curriculum-v2 artifacts from persistent curriculum-v1 failures. |
| `build_sft_boundary_rationale_data.py` | - | Builds paired boundary-rationale artifacts without further class oversampling. |
| `evidence_rationale.py` | - | Shared evidence-derived boundary-rationale rules for native CT tool observations. |
| `apply_evidence_rationale.py` | - | Applies the deployable evidence-rationale layer to native SFT JSONL artifacts. |
| `build_sft_generative_rationale_data.py` | - | Builds SFT artifacts where the target text contains the evidence rationale plus final decision JSON. |
| `build_boundary_preference_data.py` | - | Builds terminal-action preference pairs for evidence boundary confusions. |
| `build_boundary_preference_hard_modes.py` | - | Filters negative-margin boundary preference pairs into a hard-mode subset. |
| `split_boundary_preference_hard_modes.py` | - | Builds deterministic train/held-out splits for hard boundary preference pairs. |
| `run_boundary_preference_margin.py` | - | Scores chosen vs rejected boundary preference actions by model likelihood margin. |
| `run_boundary_preference_dpo_smoke.py` | - | Runs a reference-free DPO-style pairwise smoke loop over boundary preference pairs. |
| `run_boundary_preference_candidate_eval.py` | - | Scores all legal final-decision candidates for boundary preference prompts. |
| `run_boundary_preference_candidate_ce_smoke.py` | - | Runs an all-candidate cross-entropy smoke loop over hard boundary preference prompts, with boundary scheduling and optional eval-gated checkpoint selection. |
| `evaluate_evidence_guardrail.py` | - | Evaluates evidence-derived override as a guardrail on held-out model outputs. |
| `analyze_sft_boundary_rationale_failures.py` | - | Analyzes row-level boundary-rationale failures and defer-vs-verify candidate margins. |
| `run_sft_boundary_rationale_ablation.py` | - | Runs eval-only held-out oracle/evidence rationale ablations using existing boundary-rationale fold states. |
| `summarize_sft_pressure_runs.py` | - | Summarizes ignored pressure-run artifacts into tracked JSON/Markdown result anchors. |
| `summarize_sft_curriculum_run.py` | - | Summarizes ignored curriculum-run artifacts into tracked JSON/Markdown result anchors. |
| `summarize_sft_curriculum_v2_run.py` | - | Summarizes ignored curriculum-v2 run artifacts into tracked JSON/Markdown result anchors. |
| `summarize_sft_boundary_rationale_run.py` | - | Summarizes ignored boundary-rationale run artifacts into tracked JSON/Markdown result anchors. |
| `summarize_sft_boundary_rationale_ablation.py` | - | Summarizes ignored held-out rationale ablation outputs into tracked result anchors. |
| `analyze_sft_curriculum_failures.py` | - | Analyzes curriculum-run row-level failures and persistent strict/constrained errors. |
| `run_sft_cv_sweep.py` | - | Orchestrates repeated SFT train/loss/strict/constrained eval across CV folds. |
| `run_sft_oracle_warmstart.py` | - | Trains on the oracle SFT artifact and evaluates against native held-out CV folds. |
| `analyze_sft_sweep_failures.py` | - | Aggregates row-level SFT sweep failures, confusion matrices, recurrent failures, and constrained candidate ranks. |
| `run_sft_smoke.py` | - | Minimal local SFT smoke loop over the tracked SFT examples. |
| `evaluate_sft_loss.py` | - | Teacher-forced loss comparison before and after loading an SFT smoke state. |
| `run_sft_decision_eval.py` | - | Reloads an SFT smoke state and evaluates final-decision generation on the same tracked examples. |
| `run_sft_constrained_eval.py` | - | Scores legal final-decision candidates by model likelihood to remove JSON parse failures from the diagnostic. |
| `SFT_SWEEP_RESULTS_2026-06-26.md` | - | Tracked compact summary of the full CV and oracle-400 SFT execution. |
| `SFT_FAILURE_ANALYSIS_2026-06-26.md` | - | Tracked row-level failure analysis and next formulation diagnosis. |
| `SFT_PRESSURE_ARTIFACTS_2026-06-26.md` | - | Tracked summary of pressure artifacts and next full-run commands. |
| `SFT_PRESSURE_RUN_RESULTS_2026-06-26.md` | - | Tracked compact summary of the full pressure CV and balanced-oracle rerun. |
| `SFT_PRESSURE_FAILURE_ANALYSIS_2026-06-26.md` | - | Tracked row-level failure analysis for the pressure rerun. |
| `SFT_CURRICULUM_ARTIFACTS_2026-06-26.md` | - | Tracked summary of the curriculum artifact and full-run follow-up. |
| `SFT_CURRICULUM_RUN_RESULTS_2026-06-26.md` | - | Tracked compact summary of the full curriculum CV rerun. |
| `SFT_CURRICULUM_FAILURE_ANALYSIS_2026-06-26.md` | - | Tracked row-level failure analysis for the curriculum rerun. |
| `SFT_CURRICULUM_V2_RUN_RESULTS_2026-06-26.md` | - | Tracked compact summary of the targeted curriculum-v2 CV rerun. |
| `SFT_BOUNDARY_RATIONALE_ARTIFACTS_2026-06-26.md` | - | Tracked summary of the boundary-rationale artifact and fold0 sanity run. |
| `SFT_BOUNDARY_RATIONALE_RUN_RESULTS_2026-06-26.md` | - | Tracked compact summary of the full boundary-rationale CV rerun. |
| `SFT_BOUNDARY_RATIONALE_FAILURE_ANALYSIS_2026-06-26.md` | - | Row-level boundary-rationale failure analysis and defer-vs-verify diagnosis. |
| `SFT_BOUNDARY_RATIONALE_HELDOUT_ABLATION_2026-06-27.md` | - | Held-out oracle-rationale ablation result using existing boundary-rationale fold states. |
| `SFT_BOUNDARY_RATIONALE_HELDOUT_EVIDENCE_ABLATION_2026-06-27.md` | - | Held-out evidence-rationale ablation result using visible tool-derived rationale prompts. |
| `EVIDENCE_RATIONALE_LAYER_2026-06-27.md` | - | Reusable deployable evidence-rationale layer artifact and interpretation. |
| `EVIDENCE_RATIONALE_PILOT400_STRESS_2026-06-27.md` | - | Full pilot stress-test of the evidence-rationale rule on deterministic-oracle SFT data. |
| `EVIDENCE_RATIONALE_GUARDRAIL_EVAL_2026-06-27.md` | - | Guardrail override evaluation on normal boundary-rationale held-out model outputs. |
| `SFT_EVIDENCE_DISTILL_FOLD0_DIAGNOSTIC_2026-06-27.md` | - | Fold0 diagnostic for distilling the evidence-rationale rule into model weights. |
| `SFT_EVIDENCE_PROMPTED_UPPER_BOUND_FOLD0_2026-06-27.md` | - | Fold0 prompted-rule upper-bound using the evidence-rationale distillation checkpoint. |
| `SFT_GENERATIVE_RATIONALE_ARTIFACTS_2026-06-27.md` | - | Generative evidence-rationale artifact and smoke-plumbing checkpoint. |
| `SFT_GENERATIVE_RATIONALE_FOLD0_DIAGNOSTIC_2026-06-27.md` | - | Fold0 diagnostic for generative evidence-rationale SFT decision transfer. |
| `BOUNDARY_PREFERENCE_ARTIFACTS_2026-06-27.md` | - | Evidence-boundary preference artifact after negative generative-rationale SFT. |
| `BOUNDARY_PREFERENCE_MARGIN_BASE_2026-06-27.md` | - | Base-model preference-margin diagnostic over the boundary preference artifact. |
| `BOUNDARY_PREFERENCE_HARD_MODE_ARTIFACTS_2026-06-27.md` | - | Hard-mode preference subset and reference-free DPO-style smoke checkpoint. |
| `BOUNDARY_PREFERENCE_HARD_SPLIT_DPO_2026-06-27.md` | - | Heldout-aware hard-mode preference split and DPO-style margin smoke checkpoint. |
| `BOUNDARY_PREFERENCE_CANDIDATE_EVAL_2026-06-28.md` | - | Held-out all-candidate final-decision eval for the hard-split DPO state. |
| `BOUNDARY_PREFERENCE_CANDIDATE_CE_SMOKE_2026-06-28.md` | - | All-candidate CE smoke following the negative pairwise-DPO candidate eval. |
| `sft_failure_analysis_2026-06-26.json` | - | Machine-readable row-level failure analysis. |
| `sft_pressure_run_summary_2026-06-26.json` | - | Machine-readable pressure-run result summary. |
| `sft_pressure_failure_analysis_2026-06-26.json` | - | Machine-readable pressure-run row-level failure analysis. |
| `sft_curriculum_run_summary_2026-06-26.json` | - | Machine-readable curriculum-run result summary. |
| `sft_curriculum_failure_analysis_2026-06-26.json` | - | Machine-readable curriculum-run row-level failure analysis. |
| `sft_curriculum_v2_run_summary_2026-06-26.json` | - | Machine-readable targeted curriculum-v2 run result summary. |
| `sft_boundary_rationale_smoke_summary_2026-06-26.json` | - | Machine-readable boundary-rationale fold0 smoke and evalfast summary. |
| `sft_boundary_rationale_run_summary_2026-06-26.json` | - | Machine-readable boundary-rationale full CV result summary. |
| `sft_boundary_rationale_failure_analysis_2026-06-26.json` | - | Machine-readable boundary-rationale row-level failure analysis. |
| `sft_boundary_rationale_heldout_ablation_summary_2026-06-27.json` | - | Machine-readable held-out oracle-rationale ablation summary. |
| `sft_boundary_rationale_heldout_evidence_ablation_summary_2026-06-27.json` | - | Machine-readable held-out evidence-rationale ablation summary. |
| `evidence_rationale_guardrail_eval_2026-06-27.json` | - | Machine-readable evidence-rationale guardrail override evaluation. |
| `sft_evidence_distill_fold0_summary_2026-06-27.json` | - | Machine-readable fold0 evidence-rationale distillation diagnostic. |
| `sft_evidence_prompted_upper_bound_fold0_summary_2026-06-27.json` | - | Machine-readable fold0 prompted-rule upper-bound comparison. |
| `sft_generative_rationale_smoke_summary_2026-06-27.json` | - | Machine-readable generative evidence-rationale artifact and smoke summary. |
| `sft_generative_rationale_fold0_diagnostic_2026-06-27.json` | - | Machine-readable fold0 generative-rationale SFT diagnostic. |
| `boundary_preference_margin_base_summary_2026-06-27.json` | - | Machine-readable base-model boundary preference margin summary. |
| `boundary_preference_hard_mode_summary_2026-06-27.json` | - | Machine-readable hard-mode subset and DPO-style smoke summary. |
| `boundary_preference_hard_split_dpo_summary_2026-06-27.json` | - | Machine-readable heldout-aware hard-mode DPO-style margin smoke summary. |
| `boundary_preference_candidate_eval_summary_2026-06-28.json` | - | Machine-readable all-candidate final-decision eval summary for the hard-split DPO state. |
| `boundary_preference_candidate_ce_smoke_summary_2026-06-28.json` | - | Machine-readable all-candidate CE smoke summary. |

Preference failure modes:

```text
missing_attribution: 16
mixed_endpoint_over_grounding: 8
self_answering_without_tools: 40
```

The prompt shown to the learner does not include hidden scoring keys. Gold
labels and NCT IDs appear only in metadata or as returned/cited tool evidence.

SFT split:

```bash
python3 post_training/split_sft_data.py
```

Current split:

```text
seed = 20260626
train_examples = 30
heldout_examples = 10
train_by_class = defer 6, flag 6, ground 6, reject 6, verify 6
heldout_by_class = defer 2, flag 2, ground 2, reject 2, verify 2
```

Cross-validation splits:

```bash
python3 post_training/build_sft_cv_splits.py
```

Current CV split:

```text
folds = 4
seed = 20260626
source_examples = 40
source_by_class = defer 8, flag 8, ground 8, reject 8, verify 8
each_fold_train_examples = 30
each_fold_heldout_examples = 10
each_fold_heldout_by_class = defer 2, flag 2, ground 2, reject 2, verify 2
heldout_coverage_unique_examples = 40
heldout_coverage_min_count = 1
heldout_coverage_max_count = 1
```

Larger deterministic-oracle SFT artifact:

```bash
python3 post_training/export_oracle_sft_data.py
```

Current oracle artifact:

```text
dataset = negbiodb_ct_oracle_sft_v1
source_runner = deterministic_oracle_policy
sft_examples = 400
by_class = defer 120, flag 40, ground 140, reject 40, verify 60
skipped = []
boundary = Deterministic oracle-policy SFT data; not live runner behavior.
```

CV fold sweep runner:

```bash
python3 post_training/run_sft_cv_sweep.py \
  --out-dir post_training/runs/qwen_sft_cv4_schema_action_80
```

Larger oracle warm-start runner:

```bash
python3 post_training/run_sft_oracle_warmstart.py \
  --out-dir post_training/runs/qwen_oracle400_warmstart_cvheldout
```

Current runner smoke checks:

```text
cv_sweep_smoke = fold1, max_steps 1, loss-only eval
cv_sweep_train_first_loss = 2.2059
cv_sweep_train_teacher_forced_loaded_loss = 1.2253
cv_sweep_heldout_teacher_forced_loaded_loss = 1.2144

oracle_warmstart_smoke = train_limit 2, max_steps 1, fold0 loss-only eval
oracle_warmstart_train_first_loss = 1.6071
oracle_warmstart_fold0_heldout_teacher_forced_loaded_loss = 1.3513
```

These were execution-path smoke checks only. The full CV fold sweep and
oracle-400 warm-start comparison results are recorded below.

Full execution result:

```text
result_anchor = post_training/SFT_SWEEP_RESULTS_2026-06-26.md
failure_analysis_anchor = post_training/SFT_FAILURE_ANALYSIS_2026-06-26.md
native_cv_strict_action_accuracy_mean = 0.475
native_cv_constrained_loaded_accuracy_mean = 0.400
native_cv_parse_failures_total = 0
oracle400_strict_action_accuracy_mean = 0.450
oracle400_constrained_loaded_accuracy_mean = 0.400
oracle400_parse_failures_total = 0
```

Interpretation: SFT now reliably produces parse-stable final-decision JSON, but
neither the 4-fold native SFT sweep nor the oracle-400 warm start robustly
learns all action classes. The oracle artifact is larger, but did not improve
native held-out accuracy in this first formulation. Row-level analysis shows
the dominant failures are `verify -> defer` and `flag -> ground`.

Next pressure artifacts:

```text
pressure_anchor = post_training/SFT_PRESSURE_ARTIFACTS_2026-06-26.md
native_pressure_train_examples_per_fold = 54
native_pressure_train_by_class = defer 6, flag 18, ground 6, reject 6, verify 18
oracle_balanced_examples = 700
oracle_balanced_by_class = defer 140, flag 140, ground 140, reject 140, verify 140
```

Full pressure rerun result:

```text
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
```

Interpretation: native pressure is a mixed diagnostic result, not a clear win.
It slightly improves constrained loaded accuracy, but strict generation trades
the previous `ground`/`defer` behavior for reliable `verify`. Balanced-oracle
warm-start is a negative result: it overfits the balanced teacher artifact and
collapses native held-out behavior toward `ground`.

Curriculum artifacts:

```text
pressure_failure_anchor = post_training/SFT_PRESSURE_FAILURE_ANALYSIS_2026-06-26.md
curriculum_anchor = post_training/SFT_CURRICULUM_ARTIFACTS_2026-06-26.md
curriculum_manifest = post_training/negbiodb_ct_native_sft_cv4_curriculum_manifest.json
curriculum_train_examples_per_fold = 72
curriculum_train_by_class = defer 12, flag 18, ground 18, reject 12, verify 12
curriculum_train_by_family = base 30, ground_flag 12, reject_override 18, verify_defer 12
```

Interpretation: this is not another global class-balance pass. It preserves the
native base examples and adds ordered contrast blocks for `ground` vs `flag`,
`verify` vs `defer`, and mixed-endpoint `reject` override behavior.

Full curriculum rerun result:

```text
curriculum_result_anchor = post_training/SFT_CURRICULUM_RUN_RESULTS_2026-06-26.md
curriculum_result_json = post_training/sft_curriculum_run_summary_2026-06-26.json
native_curriculum_strict_action_accuracy_mean = 0.475
native_curriculum_constrained_loaded_accuracy_mean = 0.425
native_curriculum_strict_parse_failures_total = 0
native_curriculum_strict_class_accuracy = defer 3/8, flag 6/8, ground 3/8, reject 4/8, verify 3/8
native_curriculum_constrained_loaded_class_accuracy = defer 3/8, flag 7/8, ground 2/8, reject 2/8, verify 3/8
```

Interpretation: curriculum restores the original native CV strict mean and
improves `flag`, but does not produce a clear aggregate win over the earlier
native baseline or pressure run. This remains an SFT-formulation bottleneck; run
row-level curriculum-failure analysis before DPO/RLVR.

Curriculum failure analysis:

```text
curriculum_failure_anchor = post_training/SFT_CURRICULUM_FAILURE_ANALYSIS_2026-06-26.md
curriculum_failure_json = post_training/sft_curriculum_failure_analysis_2026-06-26.json
curriculum_persistent_failure_count = 20
curriculum_constrained_failure_pair_counts = defer->verify 5, flag->reject 1, ground->flag 5, ground->reject 1, reject->flag 6, verify->defer 5
```

Interpretation: `flag` is mostly learned, but the model still confuses
`defer`/`verify`, clean efficacy `ground`/suspicious `flag`, and mixed-endpoint
`reject`/`flag`. The next step is curriculum-v2 or targeted prompt/SFT work, not
DPO/RLVR yet.

Targeted curriculum-v2 result:

```text
curriculum_v2_manifest = post_training/negbiodb_ct_native_sft_cv4_curriculum_v2_manifest.json
curriculum_v2_result_anchor = post_training/SFT_CURRICULUM_V2_RUN_RESULTS_2026-06-26.md
curriculum_v2_result_json = post_training/sft_curriculum_v2_run_summary_2026-06-26.json
curriculum_v2_train_examples_per_fold = 102, 109, 103, 103
curriculum_v2_targeted_examples_per_fold = 30, 37, 31, 31
curriculum_v2_strict_action_accuracy_mean = 0.375
curriculum_v2_constrained_loaded_accuracy_mean = 0.300
curriculum_v2_strict_parse_failures_total = 0
curriculum_v2_constrained_loaded_class_accuracy = defer 3/8, flag 4/8, ground 3/8, reject 0/8, verify 2/8
```

Interpretation: targeted persistent-failure oversampling is a negative
diagnostic. It lowers accuracy relative to curriculum-v1 and does not recover
`reject`. The next formulation should change the supervision representation,
for example boundary-rationale or paired contrast SFT, rather than adding more
row duplication.

Boundary-rationale artifact:

```text
boundary_rationale_manifest = post_training/negbiodb_ct_native_sft_cv4_boundary_rationale_manifest.json
boundary_rationale_anchor = post_training/SFT_BOUNDARY_RATIONALE_ARTIFACTS_2026-06-26.md
boundary_rationale_result_json = post_training/sft_boundary_rationale_smoke_summary_2026-06-26.json
boundary_rationale_train_examples_per_fold = 60
boundary_rationale_train_by_class = defer 12, flag 12, ground 12, reject 12, verify 12
boundary_rationale_train_by_role = base 30, rationale 30
boundary_rationale_fold0_strict_action_accuracy = 0.500
boundary_rationale_fold0_constrained_loaded_action_accuracy = 0.500
boundary_rationale_fold0_strict_parse_failures = 0
```

Interpretation: boundary-rationale SFT changes the supervision representation
without further oversampling. The rationale is prompt-side context inserted
before the final `submit_decision`; the target remains the final decision JSON.
The fold0 sanity run is trainable and parse-stable, but its class pattern is
uneven (`verify`/`reject` stronger than `defer`/`ground`). Run the full 4-fold
boundary-rationale CV before making an improvement claim or moving to DPO/RLVR.

Full boundary-rationale rerun:

```text
boundary_rationale_run_anchor = post_training/SFT_BOUNDARY_RATIONALE_RUN_RESULTS_2026-06-26.md
boundary_rationale_run_json = post_training/sft_boundary_rationale_run_summary_2026-06-26.json
boundary_rationale_strict_action_accuracy_mean = 0.500
boundary_rationale_constrained_loaded_accuracy_mean = 0.500
boundary_rationale_strict_parse_failures_total = 0
boundary_rationale_strict_class_accuracy = defer 0/8, flag 3/8, ground 3/8, reject 6/8, verify 8/8
boundary_rationale_constrained_loaded_class_accuracy = defer 0/8, flag 3/8, ground 4/8, reject 5/8, verify 8/8
```

Interpretation: boundary-rationale SFT is the best native-SFT aggregate so far
in this small run family, but it does not solve the action-boundary problem.
`verify` is recovered and `reject` is strong, while every held-out `defer`
example is still predicted as `verify`. The next step should be row-level
boundary-failure analysis or a held-out prompt-side rationale ablation, not
another simple oversampling pass.

Boundary-rationale failure analysis:

```text
boundary_rationale_failure_anchor = post_training/SFT_BOUNDARY_RATIONALE_FAILURE_ANALYSIS_2026-06-26.md
boundary_rationale_failure_json = post_training/sft_boundary_rationale_failure_analysis_2026-06-26.json
defer_failure_count = 8
all_defer_failures_predicted_verify = True
all_defer_observations_empty = True
heldout_defer_prompt_has_boundary_rationale = False
mean_defer_minus_verify_mean_nll = 0.3479
```

Interpretation: the remaining `defer` failure is not a parse/decoding artifact
or a noisy evidence case. Every true-defer held-out row has empty evidence
(`search_failures=[]`, `failures_for_other_indications=0`), yet both strict
generation and constrained scoring choose `verify`. The next test should be a
held-out prompt-side rationale ablation using the existing boundary-rationale
fold states.

Held-out oracle-rationale ablation:

```text
heldout_ablation_anchor = post_training/SFT_BOUNDARY_RATIONALE_HELDOUT_ABLATION_2026-06-27.md
heldout_ablation_json = post_training/sft_boundary_rationale_heldout_ablation_summary_2026-06-27.json
heldout_ablation_strict_action_accuracy_mean = 1.000
heldout_ablation_constrained_loaded_accuracy_mean = 1.000
heldout_ablation_strict_parse_failures_total = 0
heldout_ablation_strict_class_accuracy = defer 8/8, flag 8/8, ground 8/8, reject 8/8, verify 8/8
```

Interpretation: this is not a deployable evaluation condition because the
held-out prompt includes an oracle boundary rationale and final-action hint.
It shows that the trained fold states can follow the rationale at inference
time, and that the normal held-out `defer -> verify` collapse is due to missing
or weakly internalized rationale conditioning rather than an inability to emit
`defer`.

Held-out evidence-rationale ablation:

```text
heldout_evidence_ablation_anchor = post_training/SFT_BOUNDARY_RATIONALE_HELDOUT_EVIDENCE_ABLATION_2026-06-27.md
heldout_evidence_ablation_json = post_training/sft_boundary_rationale_heldout_evidence_ablation_summary_2026-06-27.json
heldout_evidence_ablation_manifest = post_training/negbiodb_ct_native_sft_cv4_boundary_rationale_heldout_evidence_ablation_manifest.json
heldout_evidence_ablation_strict_action_accuracy_mean = 1.000
heldout_evidence_ablation_constrained_loaded_accuracy_mean = 1.000
heldout_evidence_ablation_strict_parse_failures_total = 0
heldout_evidence_ablation_strict_class_accuracy = defer 8/8, flag 8/8, ground 8/8, reject 8/8, verify 8/8
heldout_evidence_ablation_action_mismatches = 0, 0, 0, 0
```

Interpretation: a deterministic rationale generated only from visible held-out
tool observations also fully rescues `defer`. This is still a rule-preprocessor
condition, but it is much closer to a deployable guardrail/routing layer than
the oracle ablation.

Reusable evidence-rationale layer:

```text
evidence_rationale_anchor = post_training/EVIDENCE_RATIONALE_LAYER_2026-06-27.md
evidence_rationale_artifact = post_training/negbiodb_ct_native_sft_evidence_rationale_v1.jsonl
evidence_rationale_manifest = post_training/negbiodb_ct_native_sft_evidence_rationale_manifest.json
evidence_rationale_examples = 40
evidence_rationale_by_action_class = defer 8, flag 8, ground 8, reject 8, verify 8
evidence_rationale_by_evidence_action = defer 8, flag 8, ground 8, reject 8, verify 8
evidence_rationale_matches = 40
evidence_rationale_mismatches = 0
```

Interpretation: the evidence rule is now a reusable preprocessor/distillation
artifact rather than code embedded inside the ablation runner. It uses the
`Evidence-derived final action` wording to mark the action as rule-derived, not
oracle-provided.

Evidence-rationale pilot-400 stress test:

```text
evidence_pilot400_anchor = post_training/EVIDENCE_RATIONALE_PILOT400_STRESS_2026-06-27.md
evidence_pilot400_artifact = post_training/negbiodb_ct_oracle_sft_evidence_rationale_v1.jsonl
evidence_pilot400_manifest = post_training/negbiodb_ct_oracle_sft_evidence_rationale_manifest.json
evidence_pilot400_examples = 400
evidence_pilot400_by_action_class = defer 120, flag 40, ground 140, reject 40, verify 60
evidence_pilot400_by_evidence_action = defer 120, flag 40, ground 140, reject 40, verify 60
evidence_pilot400_matches = 400
evidence_pilot400_mismatches = 0
```

Interpretation: this stress source is still deterministic-oracle data, not live
runner behavior. It confirms the evidence-rationale rule is not n=40-specific
and covers the full 400-task pilot surface without action mismatches.

Evidence-rationale guardrail evaluation:

```text
evidence_guardrail_anchor = post_training/EVIDENCE_RATIONALE_GUARDRAIL_EVAL_2026-06-27.md
evidence_guardrail_json = post_training/evidence_rationale_guardrail_eval_2026-06-27.json
strict_model_action_accuracy = 0.500
strict_guardrail_action_accuracy = 1.000
strict_model_defer = 0/8
strict_guardrail_defer = 8/8
strict_rescued_errors = 20
strict_introduced_errors = 0
constrained_model_action_accuracy = 0.500
constrained_guardrail_action_accuracy = 1.000
constrained_model_defer = 0/8
constrained_guardrail_defer = 8/8
constrained_rescued_errors = 20
constrained_introduced_errors = 0
```

Interpretation: as an override on the normal boundary-rationale held-out model
outputs, the evidence-rationale guardrail fully rescues the observed failures
without introducing new errors. This supports treating the layer as a concrete
guardrail/routing candidate, not only as a training-data transformation.

Evidence-rationale distillation fold0 diagnostic:

```text
evidence_distill_anchor = post_training/SFT_EVIDENCE_DISTILL_FOLD0_DIAGNOSTIC_2026-06-27.md
evidence_distill_json = post_training/sft_evidence_distill_fold0_summary_2026-06-27.json
evidence_distill_train_sft = post_training/negbiodb_ct_oracle_sft_evidence_rationale_v1.jsonl
evidence_distill_eval = post_training/cv/negbiodb_ct_native_sft_cv4_v1_fold0_heldout.jsonl
evidence_distill_train_examples = 400
evidence_distill_train_loss_delta = -1.5084
evidence_distill_strict_action_accuracy = 0.200
evidence_distill_strict_parse_failures = 4
evidence_distill_constrained_loaded_action_accuracy = 0.200
evidence_distill_constrained_loaded_class_accuracy = defer 0/2, flag 0/2, ground 2/2, reject 0/2, verify 0/2
```

Interpretation: direct SFT distillation is a negative fold0 diagnostic. Training
on the 400-row evidence-rationale teacher artifact fits the teacher rows but
does not internalize the boundary rule for unprompted native held-out prompts.
The next distillation attempt should change the formulation before spending on
a full 4-fold rerun.

Evidence-rationale prompted-rule upper-bound:

```text
evidence_prompted_upper_bound_anchor = post_training/SFT_EVIDENCE_PROMPTED_UPPER_BOUND_FOLD0_2026-06-27.md
evidence_prompted_upper_bound_json = post_training/sft_evidence_prompted_upper_bound_fold0_summary_2026-06-27.json
same_checkpoint = post_training/runs/qwen_oracle400_evidence_distill_base_cvheldout/train/trainable_state.pt
base_prompt_strict_action_accuracy = 0.200
base_prompt_constrained_loaded_action_accuracy = 0.200
prompted_strict_action_accuracy = 1.000
prompted_constrained_loaded_action_accuracy = 1.000
prompted_strict_parse_failures = 0
```

Interpretation: the distillation checkpoint can follow the rule when the
evidence-rationale message is present, but it does not retrieve the rule from
the unprompted native base prompt. This favors external guardrail/routing or a
stronger generative-rationale/preference distillation formulation.

Generative evidence-rationale artifact:

```text
generative_rationale_anchor = post_training/SFT_GENERATIVE_RATIONALE_ARTIFACTS_2026-06-27.md
generative_rationale_json = post_training/sft_generative_rationale_smoke_summary_2026-06-27.json
generative_rationale_train_artifact = post_training/negbiodb_ct_oracle_sft_generative_rationale_v1.jsonl
generative_rationale_train_examples = 400
generative_rationale_train_matches = 400
generative_rationale_fold0_artifact = post_training/generative_rationale/negbiodb_ct_native_sft_cv4_generative_rationale_v1_fold0_heldout.jsonl
generative_rationale_fold0_examples = 10
generative_rationale_fold0_matches = 10
oracle_encode_dry_length_range = 422..751
fold0_encode_dry_length_range = 424..570
generative_rationale_smoke_loss = 2.3865
```

Interpretation: this is an artifact/plumbing checkpoint, not a held-out
performance result. It creates the next distillation formulation: the model must
generate the evidence rationale and then the final decision JSON from tool
observations.

Generative-rationale fold0 diagnostic:

```text
generative_rationale_diagnostic_anchor = post_training/SFT_GENERATIVE_RATIONALE_FOLD0_DIAGNOSTIC_2026-06-27.md
generative_rationale_diagnostic_json = post_training/sft_generative_rationale_fold0_diagnostic_2026-06-27.json
train_examples = 400
train_loss_delta = -2.3574
generative_fold0_strict_action_accuracy = 0.300
generative_fold0_strict_parse_failures = 0
native_base_fold0_strict_action_accuracy = 0.000
native_base_fold0_strict_parse_failures = 10
native_base_fold0_constrained_loaded_action_accuracy = 0.400
native_base_fold0_constrained_loaded_class_accuracy = defer 0/2, flag 0/2, ground 2/2, reject 0/2, verify 2/2
```

Interpretation: the checkpoint fits the 400-row generative-rationale train
artifact by loss, but this formulation does not yet transfer the boundary rule
to held-out generation or unprompted native/base decisions. Keep the external
evidence-rationale guardrail as the strongest deployable path while testing a
different contrastive/preference distillation objective.

Evidence-boundary preference artifact:

```text
boundary_preference_anchor = post_training/BOUNDARY_PREFERENCE_ARTIFACTS_2026-06-27.md
boundary_preference_artifact = post_training/negbiodb_ct_oracle_boundary_preferences_v1.jsonl
boundary_preference_manifest = post_training/negbiodb_ct_oracle_boundary_preferences_manifest.json
source_examples = 400
preference_pairs = 620
chosen_passed = 620
rejected_passed = 0
failure_modes = boundary_defer_over_verify 120, boundary_verify_over_defer 60, boundary_ground_over_flag 140, boundary_ground_over_reject 140, boundary_flag_over_ground 40, boundary_flag_over_reject 40, boundary_reject_over_ground 40, boundary_reject_over_flag 40
```

Interpretation: this is the next contrastive objective candidate. Chosen and
rejected responses share the same visible tool evidence and differ only in the
terminal `submit_decision`, isolating boundary action selection from tool-use
formatting.

Boundary preference base-margin diagnostic:

```text
boundary_preference_margin_anchor = post_training/BOUNDARY_PREFERENCE_MARGIN_BASE_2026-06-27.md
boundary_preference_margin_json = post_training/boundary_preference_margin_base_summary_2026-06-27.json
boundary_preference_margin_raw = post_training/runs/qwen_boundary_preference_margin/base_full.json
model = Qwen/Qwen2.5-0.5B-Instruct
n = 620
mean_win_rate = 0.615
sum_win_rate = 0.453
mean_margin_mean = 0.4442
hard_negative_modes = boundary_defer_over_verify 0.008/-0.1125, boundary_flag_over_ground 0.000/-0.1656, boundary_reject_over_ground 0.000/-2.6783, boundary_reject_over_flag 0.000/-2.5255
easy_or_aligned_modes = boundary_verify_over_defer 1.000/0.1042, boundary_ground_over_flag 1.000/0.1784, boundary_ground_over_reject 1.000/2.6621, boundary_flag_over_reject 1.000/2.4932
```

Interpretation: the full preference artifact is not uniformly hard. The next
DPO/preference smoke should begin with negative-margin boundary modes rather
than all 620 pairs uniformly, because the base model is already aligned on the
easy contrast families.

Boundary preference hard-mode subset and DPO-style smoke:

```text
hard_mode_anchor = post_training/BOUNDARY_PREFERENCE_HARD_MODE_ARTIFACTS_2026-06-27.md
hard_mode_json = post_training/boundary_preference_hard_mode_summary_2026-06-27.json
hard_mode_artifact = post_training/negbiodb_ct_oracle_boundary_preferences_hard_v1.jsonl
hard_mode_manifest = post_training/negbiodb_ct_oracle_boundary_preferences_hard_manifest.json
hard_mode_pairs = 240
hard_mode_failure_modes = boundary_defer_over_verify 120, boundary_flag_over_ground 40, boundary_reject_over_ground 40, boundary_reject_over_flag 40
dpo_smoke_condition = reference_free_dpo_style_limit8_overfit
dpo_smoke_pre_win_rate = 0.000
dpo_smoke_pre_mean_margin = -1.4004
dpo_smoke_post_win_rate = 1.000
dpo_smoke_post_mean_margin = 6.9414
dpo_smoke_loss_delta = -0.2598
```

Interpretation: this is a plumbing/overfit success, not a held-out improvement
claim. The reference-free pairwise objective can move hard-pair margins in the
right direction on 8 selected pairs. The next safe experiment is a full
hard-subset run with pre/post margins by failure mode and preferably a hard-pair
held-out split.

Boundary preference hard split and heldout-aware DPO-style smoke:

```text
hard_split_anchor = post_training/BOUNDARY_PREFERENCE_HARD_SPLIT_DPO_2026-06-27.md
hard_split_json = post_training/boundary_preference_hard_split_dpo_summary_2026-06-27.json
hard_split_train = post_training/negbiodb_ct_oracle_boundary_preferences_hard_train_v1.jsonl
hard_split_heldout = post_training/negbiodb_ct_oracle_boundary_preferences_hard_heldout_v1.jsonl
hard_split_manifest = post_training/negbiodb_ct_oracle_boundary_preferences_hard_split_manifest.json
hard_split_train_pairs = 208
hard_split_heldout_pairs = 32
hard_split_overlap_source_ids = 0
dpo_split_condition = reference_free_dpo_style_hard_split_steps48
dpo_split_train_eval_pairs = 32
dpo_split_eval_pairs = 32
dpo_split_pre_eval_win_rate = 0.000
dpo_split_pre_eval_mean_margin = -1.3701
dpo_split_post_eval_win_rate = 1.000
dpo_split_post_eval_mean_margin = 39.5352
dpo_split_loss_delta = -0.6113
```

Interpretation: this is stronger than the limit8 smoke because the held-out
hard pairs also flip to positive margins across all four selected hard
failure-mode families. It is still a margin-only diagnostic over deterministic
oracle-derived terminal-action pairs, not final live tool-use trajectory
accuracy and not an RLVR deployment claim.

Boundary preference all-candidate final-decision eval:

```text
candidate_eval_anchor = post_training/BOUNDARY_PREFERENCE_CANDIDATE_EVAL_2026-06-28.md
candidate_eval_json = post_training/boundary_preference_candidate_eval_summary_2026-06-28.json
candidate_eval_preferences = post_training/negbiodb_ct_oracle_boundary_preferences_hard_heldout_v1.jsonl
candidate_eval_state = post_training/runs/qwen_boundary_preference_dpo_hard_split_steps48/trainable_state.pt
base_action_accuracy = 0.000
base_exact_candidate_accuracy = 0.000
dpo_loaded_action_accuracy = 0.250
dpo_loaded_exact_candidate_accuracy = 0.250
dpo_loaded_defer_over_verify = 8/8
dpo_loaded_flag_over_ground = 0/8, pred defer 8
dpo_loaded_reject_over_flag = 0/8, pred defer 8, expected rank 2 for 8/8
dpo_loaded_reject_over_ground = 0/8, pred defer 8, expected rank 2 for 8/8
```

Interpretation: pairwise margin success does not yet imply all-candidate
decision success. The DPO-loaded state fixes `defer_over_verify`, but it
collapses top-1 decisions to `defer` for the `flag` and `reject` hard modes.
The next objective should expose all legal negatives or multiple hard negatives
per prompt before making a decision-level improvement claim.

Boundary preference all-candidate CE smoke:

```text
candidate_ce_anchor = post_training/BOUNDARY_PREFERENCE_CANDIDATE_CE_SMOKE_2026-06-28.md
candidate_ce_json = post_training/boundary_preference_candidate_ce_smoke_summary_2026-06-28.json
candidate_ce_script = post_training/run_boundary_preference_candidate_ce_smoke.py
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

Interpretation: direct all-candidate CE can move top-1 behavior on held-out
hard slices, and round-robin boundary scheduling improves the current small
eval from 0.333 to 0.583. The tradeoff is still unresolved: source-order
training solves `flag`, while round-robin recovers `defer`/`reject` and loses
`flag` to `reject`. A repeated-`flag` rehearsal and stop-after-reject variant
both fail to beat round-robin. The next step is eval-gated checkpoint selection
or a Pareto/multi-objective boundary selector before full held-out hard-split
all-candidate eval. The selector plumbing now exists and records the unresolved
action-family collapse instead of hiding it. On the balanced a8/eval4 slice, it
selects a stronger 0.667 round-robin checkpoint, but still leaves one action
family at 0/4. Phase-batch updates plus eval-gated selection produce the first
tested small-slice checkpoint without a zero-accuracy action family: held-out
0.917 with `defer` 4/4, `flag` 4/4, and `reject` 3/4. Do not treat this as a
broad post-training claim yet. A seeded candidate-set replicate still avoids
zero-family collapse at 0.750, but `flag` remains weak at 1/4. Naive flag-only
loss weights fix held-out `flag` 4/4, but collapse `reject` to 0/4 at both
`flag=1.25` and `flag=2.0`, so they are negative robustness results rather than
scaling answers. Joint low `flag`/`reject` weighting preserves `reject` 4/4 but
collapses `flag` to 0/4, so naive action weighting is not sufficient.
Action-level CE recovers `defer`/`flag` 4/4 but still collapses `reject -> flag`
at 0/4. An explicit `action_floor` selector can select a checkpoint satisfying
nonzero `flag` and `reject` floors on the seed 11 slice, recovering `defer` 4/4,
`reject` 4/4, and `flag` 1/4 at action accuracy 0.750. Treat this as selector
hardening rather than a solved behavior. A one-sided `flag>reject` margin
penalty is active and recovers `flag` 4/4, but it again collapses `reject` 0/4.
A two-sided `flag>reject` plus `reject>flag` margin probe preserves `reject`
partly at the selected checkpoint, but leaves `flag` 0/4 and still fails the
explicit floor gate. The next robustness work should use a genuinely
floor-aware training objective or multi-negative action objective before full
hard-split scaling.

Validate with:

```bash
python3 post_training/validate_post_training_data.py
```

Current validator result:

```text
issues = []
sft_examples = 40
preference_pairs = 64
boundary_preference_pairs = 620
hard_boundary_preference_pairs = 240
hard_boundary_preference_failure_modes = boundary_defer_over_verify 120, boundary_flag_over_ground 40, boundary_reject_over_ground 40, boundary_reject_over_flag 40
hard_boundary_preference_train_pairs = 208
hard_boundary_preference_heldout_pairs = 32
oracle_sft_examples = 400
oracle_sft_by_class = defer 120, flag 40, ground 140, reject 40, verify 60
```

## Prompt-Only Baseline

The same 40 packet IDs were evaluated without tools using Sonnet:

```bash
python3 negbiodb_ct/run_prompt_only.py \
  --task-ids-from post_training/negbiodb_ct_native_sft_v1.jsonl \
  --out negbiodb_ct/agent_prompt_only_sonnet_n40.json

python3 examples/analyze_negbiodb_ct_prompt_only.py \
  negbiodb_ct/agent_prompt_only_sonnet_n40.json
```

Current result:

```text
action_accuracy = 0.150
mean_reward = 0.150
native_profile_mean_score = 0.450
tool_call_rate = 0.000
by_class = defer 6/8, flag 0/8, ground 0/8, reject 0/8, verify 0/8
pred_actions = defer 31, reject 3, verify 3, parse failure 2, flag 1
top failure bucket = conservative_defer_wrong_without_tools 25
```

Interpretation: the prompt-only model mostly chooses conservative `defer`, and
only succeeds on a subset of true-defer cases. This provides the first direct
comparison showing why the native tool trajectory data matters before moving to
SFT/DPO.

## Open-Model Baseline

The same packet set also runs through a local Hugging Face causal-LM adapter:

```bash
python3 negbiodb_ct/run_open_model_prompt_only.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --task-ids-from post_training/negbiodb_ct_native_sft_v1.jsonl \
  --out negbiodb_ct/agent_open_model_qwen05_n40.json

python3 examples/analyze_negbiodb_ct_prompt_only.py \
  negbiodb_ct/agent_open_model_qwen05_n40.json
```

Current local Qwen 0.5B result:

```text
action_accuracy = 0.200
mean_reward = 0.200
native_profile_mean_score = 0.467
tool_call_rate = 0.000
by_class = verify 8/8, defer 0/8, flag 0/8, ground 0/8, reject 0/8
pred_actions = verify 40
top failure buckets = missed_positive_or_invalid_evidence 16, oververify_without_evidence 8, missed_mixed_endpoint_contradiction 8
```

Method decision: start with SFT, not DPO/RLVR. The trainable open model first
needs to learn the basic task schema, action distribution, and when evidence
requires tools. DPO pairs are useful after SFT reduces the all-`verify` collapse.

## SFT Smoke Loop

The first learner-side smoke test runs one masked-label training step over a
small subset of the tracked native trajectories:

```bash
python3 post_training/run_sft_smoke.py \
  --limit 2 \
  --max-steps 1 \
  --batch-size 1 \
  --max-length 512 \
  --train-last-layers 1 \
  --device auto \
  --out-dir post_training/runs/qwen_sft_smoke
```

Current local result:

```text
model = Qwen/Qwen2.5-0.5B-Instruct
device = mps
examples = 2
max_steps = 1
train_last_layers = 1
trainable_params = 14913280
loss = 4.4349
trainable_state = post_training/runs/qwen_sft_smoke/trainable_state.pt
```

Boundary: this proves the SFT data/model plumbing path only. It does not claim
downstream accuracy improvement.

Teacher-forced loss comparison on all 40 tracked examples:

```bash
python3 post_training/evaluate_sft_loss.py \
  --state post_training/runs/qwen_sft_smoke/trainable_state.pt \
  --limit 40 \
  --batch-size 2 \
  --max-length 512 \
  --device auto \
  --out post_training/runs/qwen_sft_loss_compare.json
```

Current local result:

```text
base_loss = 2.4869
loaded_loss = 2.4281
loss_delta = -0.0589
target_tokens = 536
loaded_tensors = 13
unexpected_tensors = []
```

The saved state reloads and can be evaluated on the exact same 40 tracked
examples:

```bash
python3 post_training/run_sft_decision_eval.py \
  --state post_training/runs/qwen_sft_smoke/trainable_state.pt \
  --limit 40 \
  --device auto \
  --out post_training/runs/qwen_sft_decision_eval_n40.json
```

Current strict-generation result:

```text
loaded_tensors = 13
unexpected_tensors = []
action_accuracy = 0.000
mean_reward = 0.000
generic_mean_score = 0.433
tool_call_rate = 1.000
parse_failures = 40
```

A JSON-prefill diagnostic produced the same parse-failure count:

```text
action_accuracy = 0.000
parse_failures = 40
```

Constrained candidate scoring removes parsing from the evaluation by scoring
legal final-decision JSON candidates:

```bash
python3 post_training/run_sft_constrained_eval.py \
  --limit 40 \
  --max-length 512 \
  --device auto \
  --score-mode mean \
  --out post_training/runs/qwen_sft_constrained_base_n40.json

python3 post_training/run_sft_constrained_eval.py \
  --state post_training/runs/qwen_sft_smoke/trainable_state.pt \
  --limit 40 \
  --max-length 512 \
  --device auto \
  --score-mode mean \
  --out post_training/runs/qwen_sft_constrained_loaded_n40.json
```

Current result:

```text
base constrained action_accuracy = 0.275
loaded constrained action_accuracy = 0.275
changed_rows = 0
parse_failures = 0
by_class = ground 8/8, verify 3/8, defer 0/8, flag 0/8, reject 0/8
pred_actions = ground 24, reject 11, verify 5
```

Longer same-40 schema/action SFT diagnostic:

```bash
python3 post_training/run_sft_smoke.py \
  --limit 40 \
  --max-steps 80 \
  --batch-size 2 \
  --max-length 512 \
  --train-last-layers 2 \
  --lr 5e-5 \
  --device auto \
  --out-dir post_training/runs/qwen_sft_schema_action_80
```

Current local result:

```text
train_last_layers = 2
trainable_params = 29825664
first_loss = 2.2059
last_loss = 0.0293
teacher_forced_base_loss = 2.4869
teacher_forced_loaded_loss = 0.0546
strict_generation_action_accuracy = 0.725
strict_generation_parse_failures = 0
constrained_action_accuracy = 0.725
generic_mean_score = 0.908
by_class = defer 4/8, flag 3/8, ground 6/8, reject 8/8, verify 8/8
```

Interpretation: save/reload/eval plumbing is now present. A one-step smoke state
reduces teacher-forced target loss but does not change legal-candidate ranking.
The longer same-40 diagnostic shows that the current SFT formulation can learn
both the JSON output contract and action discrimination when trained harder.
This is an overfit-style substrate check, not a held-out generalization claim.

Train/held-out schema/action SFT diagnostic:

```bash
python3 post_training/run_sft_smoke.py \
  --sft post_training/negbiodb_ct_native_sft_train_v1.jsonl \
  --limit 30 \
  --max-steps 80 \
  --batch-size 2 \
  --max-length 512 \
  --train-last-layers 2 \
  --lr 5e-5 \
  --device auto \
  --out-dir post_training/runs/qwen_sft_train30_schema_action_80
```

Current held-out result:

```text
train_teacher_forced_loaded_loss = 0.0422
heldout_teacher_forced_loaded_loss = 0.0857
heldout_strict_generation_action_accuracy = 0.700
heldout_strict_generation_parse_failures = 0
heldout_constrained_base_action_accuracy = 0.300
heldout_constrained_loaded_action_accuracy = 0.500
heldout_strict_by_class = defer 1/2, flag 1/2, ground 2/2, reject 2/2, verify 1/2
```

Interpretation: this is still a tiny diagnostic split, but it moves beyond
same-40 overfit. The train-fold SFT lowers held-out teacher-forced loss, emits
valid JSON on held-out examples, and improves held-out constrained scoring over
the base model. The next technical step is to repeat the diagnostic across all
CV folds and compare a larger oracle-SFT warm start against the native held-out
folds before using DPO or RLVR. The orchestration scripts for those two runs are
now present, with smoke tests proving the execution path but not yet claiming
full-run performance.

## Boundary

The native n=40 artifacts are live native CT runner outputs, not the full
four-step NullAtlas training target. The oracle n=400 artifact is larger, but it
is deterministic teacher data from the scoring key plus native CT tool
observations, not live runner behavior. Together they are enough for first-pass
SFT/DPO plumbing and failure-mode preference tests. A later artifact should
export the stricter `nullatlas_full` trajectory form once the runner emits that
expanded tool loop.

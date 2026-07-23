# Project Status

Last updated: 2026-07-23

## Current Thesis

Biology tool-use agents should be evaluated on trainable
tool -> evidence packet -> terminal action trajectories, with runtime evidence
arbitration and fail-closed gates. Stage A remains a benchmark and diagnosis
substrate first; DPO/RLVR, real-query expansion, and Hugging Face publication
stay closed until source-separated policies beat deterministic runtime
baselines. The existing v0.1.0 GitHub release is a reproducibility snapshot,
not evidence that the scientific gate passed.

## Current Result

The missing `tool_query` component diagnostic and the one-time sealed
candidate-routing evaluation are complete.

- `tool_query` placeholder-schema SFT: 0/5 held-out pass, mean score 0.250,
  target-key accuracy 0.0, and tool-query-shape accuracy 0.0;
- all five generated outputs are parseable JSON with prompt-schema fields, but
  none contains `tool_calls`;
- the current `tool_query` target has one shared four-tool sequence with
  literal `<drug_id>` / `<condition_id>` arguments, so this result does not
  evaluate identifier resolution or live tool execution;
- frozen candidate-routing policy: the saved July 10 trainable state is loaded
  directly, with no retraining before sealed evaluation;
- source-separated sealed routing: 5/25 exact, selecting
  `verify/insufficient` on all 25 rows;
- best static single-pair prior: 5/25;
- deterministic runtime oracle gate: 25/25;
- incorrect `ground/supported` predictions: 0, but useful evidence-family
  discrimination is also absent.

The sealed result independently reproduces the exposed-development collapse.
It does not beat the static prior and remains far below runtime evidence
arbitration. The one-time lock is complete; do not tune on or rescore these 25
sealed rows.

Stage A saved-output diagnosis is active. The latest compact Cayuga summaries
show that teacher-forced full-target scoring can be repaired on the 4-case
held-out calibration slice, but finite-candidate ranking still fails the
runtime baseline. The follow-up non-flag balanced checkpoint completed and
changed the collapse pattern without meeting the gate:

- teacher-forced full-target margin: 4/4 held-out wins;
- finite-candidate top-1: 1/4 held-out exact, collapsing to
  `flag/invalid_value`;
- train-derived calibration: 2/4 held-out exact;
- non-flag balanced candidate top-1: 1/4 raw held-out exact and 1/4 calibrated
  held-out exact, shifting top pairs toward `verify/insufficient`,
  `ground/supported`, and `reject/contradicted` but not improving exact score;
- non-flag field diagnostic: 1/4 action top-1, 2/4 evidence-status top-1, and
  only 1/4 exact pair top-1;
- candidate-CE pair+field checkpoint: 1/4 raw held-out exact and 1/4
  calibrated held-out exact, shifting all raw held-out top pairs to
  `verify/insufficient` without improving exact score;
- runtime evidence and hybrid arbitration: 4/4 held-out exact with zero trusted
  candidate incorrect cases;
- meet-or-beat gate: fails because no current model/candidate policy reaches
  the runtime arbitration baseline;
- evidence-conditioned candidate-routing full smoke: 4/20 train exact, 1/5
  held-out exact, and 1/4 bridge-focus exact; the model selects
  `verify/insufficient` for all 20 train and all 5 held-out rows.

The compact checkpoint is
`post_training/stage_a_saved_output_checkpoint_diagnosis_2026-07-10.json` with
the companion report
`post_training/STAGE_A_SAVED_OUTPUT_CHECKPOINT_DIAGNOSIS_2026-07-10.md`.

The next Cayuga checkpoint spec is
`post_training/stage_a_saved_output_next_checkpoint_spec_2026-07-10.json` with
the companion report
`post_training/STAGE_A_SAVED_OUTPUT_NEXT_CHECKPOINT_SPEC_2026-07-10.md`.

The completed non-flag checkpoint result is
`post_training/stage_a_saved_output_nonflag_checkpoint_result_2026-07-10.json`
with the companion report
`post_training/STAGE_A_SAVED_OUTPUT_NONFLAG_CHECKPOINT_RESULT_2026-07-10.md`.

The next candidate-routing checkpoint spec is
`post_training/stage_a_saved_output_candidate_ce_checkpoint_spec_2026-07-10.json`
with the companion report
`post_training/STAGE_A_SAVED_OUTPUT_CANDIDATE_CE_CHECKPOINT_SPEC_2026-07-10.md`.

The completed candidate-CE checkpoint result is
`post_training/stage_a_saved_output_candidate_ce_checkpoint_result_2026-07-10.json`
with the companion report
`post_training/STAGE_A_SAVED_OUTPUT_CANDIDATE_CE_CHECKPOINT_RESULT_2026-07-10.md`.

The post-candidate-CE next-decision checkpoint is
`post_training/stage_a_saved_output_post_candidate_ce_next_decision_2026-07-10.json`
with the companion report
`post_training/STAGE_A_SAVED_OUTPUT_POST_CANDIDATE_CE_NEXT_DECISION_2026-07-10.md`.

The evidence-conditioned saved-output bridge is
`post_training/stage_a_saved_output_evidence_bridge_2026-07-10.json` with the
companion report
`post_training/STAGE_A_SAVED_OUTPUT_EVIDENCE_BRIDGE_2026-07-10.md`. It joins
18/18 compact candidate-failure rows to prompt-visible evidence-gate rows
across 4 unique held-out failure cases.

The evidence-conditioned candidate-routing substrate is
`post_training/stage_a_saved_output_evidence_candidate_routing_rows_v1.jsonl`
with train/held-out splits and manifest in `post_training/`. It contains 25
finite-candidate routing rows, preserves the 20/5 Stage A split, balances all
five action/status pairs at 5 rows each, and marks the 4 bridge failure cases as
held-out evaluation focus rows rather than training rows.

The evidence-conditioned candidate-routing readout is
`post_training/stage_a_saved_output_evidence_candidate_routing_readout_2026-07-10.json`
with the companion report
`post_training/STAGE_A_SAVED_OUTPUT_EVIDENCE_CANDIDATE_ROUTING_READOUT_2026-07-10.md`.
The runtime evidence gate is 5/5 held-out exact and 4/4 bridge-focus exact,
while every static single-pair prior is at most 1/5 held-out exact.

The evidence-conditioned candidate-routing smoke spec is
`post_training/stage_a_saved_output_evidence_candidate_routing_smoke_spec_2026-07-10.json`
with the companion report
`post_training/STAGE_A_SAVED_OUTPUT_EVIDENCE_CANDIDATE_ROUTING_SMOKE_SPEC_2026-07-10.md`.
It sets the next model policy gate at 5/5 held-out exact and 4/4 bridge-focus
exact before any escalation.

The smoke runner is
`post_training/run_stage_a_saved_output_evidence_candidate_routing_smoke.py`
with Cayuga wrapper
`post_training/run_stage_a_saved_output_evidence_candidate_routing_smoke_cayuga.sbatch`.
Local dry-run validates the 20/5 split and 4 held-out bridge-focus rows without
loading model weights.

The compact smoke-result adapter is
`post_training/evaluate_stage_a_saved_output_evidence_candidate_routing_smoke_result.py`.
It reads only compact `eval_report.json` outputs, fails closed if raw candidate
scores or raw model text appear, redacts external compact-input paths, and
applies the 5/5 held-out plus 4/4 bridge-focus gate.

The Cayuga mirror dry-run checkpoint is
`post_training/stage_a_saved_output_evidence_candidate_routing_cayuga_dry_run_2026-07-10.json`
with the companion report
`post_training/STAGE_A_SAVED_OUTPUT_EVIDENCE_CANDIDATE_ROUTING_CAYUGA_DRY_RUN_2026-07-10.md`.
It records 20 train rows, 5 held-out rows, 4 bridge-focus held-out rows, an
empty issue list, and `ready_for_full_mode=true` at mirror commit `6820498`.

The approved Cayuga full-smoke result is
`post_training/stage_a_saved_output_evidence_candidate_routing_smoke_result_qwen05b_cayuga_2026-07-10.json`
with companion report
`post_training/STAGE_A_SAVED_OUTPUT_EVIDENCE_CANDIDATE_ROUTING_SMOKE_RESULT_QWEN05B_CAYUGA_2026-07-10.md`.
It fails the acceptance gate at 1/5 held-out exact and 1/4 bridge-focus exact,
does not beat the 1/5 static prior, and remains below the 5/5 runtime evidence
gate. Raw candidate scores, model state, and scheduler logs remain uncommitted.

The sealed evaluation commitment is
`post_training/stage_a_sealed_extension_commitment_2026-07-10.json` with
companion report
`post_training/STAGE_A_SEALED_EXTENSION_COMMITMENT_2026-07-10.md`. It commits
to 25 private rows balanced at 5 per action family. The source-task,
split-group, and normalized-claim overlap counts against the declared public
task/manifest exclusions are all zero. The private candidate pool and selected
manifest remain outside the repository with row-level labels and source IDs
unpublished.

## Source Changes

A focused verifier/benchmark refresh strengthens, rather than changes, the
thesis. BioAgent Bench and SciAgentGym motivate perturbation and horizon-aware
tool-use slices. The Art of Building Verifiers motivates separate process and
outcome signals plus controllable/uncontrollable failure attribution.
Plan-RewardBench supports keeping unaudited LLM judges out of deterministic
RLVR gates because trajectory reward models degrade on difficult long-horizon
traces.

## Next Decision

Proceed with
`prospective_real_query_slice_and_runtime_hybrid_before_new_post_training`.

Do not repeat the completed sealed evaluation or use its rows for training,
threshold selection, model selection, or prompt repair. The next Stage A
research substrate should be prospectively generated and should replace
placeholder arguments with actual model-visible query values or an explicit
entity-resolution interface. Pre-register source separation, mutation tests,
and risk/coverage metrics before scoring.

In parallel, evaluate the deployment claim directly: compare the frozen model,
the deterministic evidence gate, and a runtime-enforced hybrid under missing
attribution, stale source, contradictory evidence, invalid numeric values,
partial queries, wrong tools, and unavailable-tool perturbations. Keep
DPO/RLVR closed until a new development substrate shows a component-specific
repair and a separate future sealed set confirms it. Explanation quality
remains preference/process-supervision territory. Raw predictions, candidate
scores, scheduler logs, model state, private manifests, and one-time lock files
remain uncommitted.

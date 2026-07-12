# NegBioDB-CT tool-use trajectory post-training — experiment design

*Design (pre-execution). The realized "first concrete direction" of this project, instantiated on the
NegBioDB / NullAtlas negative-evidence tool. Not the SFM ipTM-trust track (bio_sfm_designer) — that is the
complementary structure-predictor half of the two-specialist thesis. Aligned with JK over 2026-06-25.*

## 0. One-line aim
Does **SFT on expert NullAtlas tool-use trajectories** install a valid, complete, attribution-correct,
calibrated-defer **negative-evidence tool-use loop** in a *weak open model*, closing the gap to the
frontier ceiling — **without from-base RLVR** (which starves) and **without training the validity
judgment** (which the deterministic tool already does better)?

## 1. Question · hypothesis · predicted null
- **H+**: SFT installs the tool-use loop mechanics → "post-training installs the tool-use *policy*
  (SFT works, isomorphic to the report's SFT leg); facts stay in retrieval, validity stays the
  deterministic tool." Clean, novel, honest.
- **Predicted null**: prompting-only already saturates → no headroom (extends the report's cross-vendor
  grounding to open models). Pre-empted by Gate 1.

## 2. The conceptual frame: a 3-way decomposition (this is what pre-empts the predictable failures)
| What | Belongs to | Why (already proven) |
|---|---|---|
| **Facts** (which trial failed) | **retrieval** (NegBioDB-CT) | two-walls: from-base RLVR starves; not trainable in |
| **Validity judgment** (is this record's value possible?) | **deterministic tool** (`check_value_validity`) | provenance-over-trust: the gate beats the model 22×; post-training the judgment stays manipulable |
| **Trajectory mechanics** (which tool, in what order; attribution; calibrated defer/reject/flag) | **← the only legitimate SFT target** | the tool and retrieval *don't* do these |

Targeting the facts (starves) or the validity judgment (the tool already wins) = a predictable failure.
The decomposition forecloses both; the SFT target is **the loop mechanics only**.

## 3. Environment: NegBioDB-CT + the NullAtlas 6-tool trajectory
- **DB**: `data/negbiodb_ct.db` (229 MB; 132,925 failure results / 216,987 trials; AACT/ClinicalTrials.gov).
- **Tool**: the NullAtlas MCP (`src/nullatlas_mcp/server.py`), already wired to CT, exposing a real
  multi-step loop:
  `nullatlas_survey_prior_failures` (search) → `nullatlas_verify_trial_claims` (attribution) →
  `nullatlas_check_value_validity` (validity gate) → `nullatlas_negative_evidence_completeness`
  (coverage → defer) → terminal action.
- Why CT (not another NegBioDB domain): the NullAtlas tool-chain is most developed on CT; clean
  deterministic rewards (NCT existence · attribution-match · validity · completeness); directly connected
  to the report's priors; the loop is genuinely multi-step (not degenerate). Multi-domain (tool-SELECTION
  over dti/gex/gwas/hpa/ppi — the retriever supports them) is a later extension, not the first run.

## 4. Trajectory schema (NegBioDB-native; reuses the trust_cue_attribution interface)
Action / tool loop the model orchestrates, given a claim "did drug X fail for indication Y?":
- decision: `answer_self` **vs** `call_nullatlas` (the autonomous-uptake decision)
- tool steps: `survey_prior_failures` → `verify_trial_claims` (attribution) → `check_value_validity`
  → `negative_evidence_completeness`
- terminal: `ground_with_attribution` | `defer_insufficient` | `reject_contradicted` | `flag_invalid`
- the **validity judgment** is the tool's output, not the model's call (the model learns *when to call it*).
Trajectory/observation/reward records follow `LLM_SFM_interpretability/experiments/trust_cue_attribution/SCHEMA.md`
(hidden truth scored ONLY after the action; observation never contains truth/labels).

## 5. Task construction (balanced-action · held-out · non-circular · anti-leak)
- Claims keyed on `intervention_id × condition_id` (drug × indication) from `trial_failure_results`.
- **Held-out truth source (resolved)**: `exports/ct/negbiodb_ct_splits.csv` + `negbiodb_ct_m1_*.parquet`
  (regenerate via `scripts_ct/export_ct_ml_dataset.py`, seeds 42/43/44). The tool retrieves from the
  **train** split; eval items + gold come from a **held-out** split → calling the tool cannot trivially
  return the eval answer (non-circular).
- **Balanced across actions** so a constant policy fails: (a) recorded negative in train → ground+attribution;
  (b) no record (held-out / absent) → defer; (c) contradicts a recorded success/different-outcome → reject;
  (d) fuzzy near-match → verify; (e) impossible value injected → flag (calls `check_value_validity`).
- gold = `source_record_id` (NCT) + `primary_endpoint_met`/`failure_category`. No gold/action in the
  observation (SCHEMA enforces) + a leak-guard test.

## 6. Reward (deterministic · non-circular · anti-reward-hack)
`reward = correct_terminal + attribution_bonus − λ·verify_cost − defer_penalty·wrong_defer − fabrication_penalty`
- **attribution-match**, not existence (the 78/9 lesson): cite the NCT for *this* drug+indication.
- **defer_penalty > 0** AND penalize defer when a record exists (kills the A70 abstain-attractor).
- **fabrication_penalty**: citing an NCT absent from the registry (the report's fabrication metric).
- scored against held-out gold, never against the tool's own returned text (non-circular).

## 7. Conditions + baselines (all run *before* training — A6 baseline-before-reasoning)
1. **regex / feature-row router** (observation→action). If it hits the trajectory score → confounded, redesign (G2).
2. **deterministic policy baseline** (reuse `policy::signal_gated_verify`). If it gets the loop right → the model isn't needed.
3. **prompting-only** (the weak open model, 0/few-shot) = headroom.
4. **frontier reference** (Sonnet, API) = ceiling (report priors).
5. **SFT** (weak open model) — the target.
6. **format-only control** (SFT on format-correct / policy-wrong trajectories) → isolate policy learning from format.

## 8. Model panel (large models OK; headroom is the constraint)
| Role | Model | Why |
|---|---|---|
| **Primary (SFT)** | **Llama-3.3-70B** | direct continuity with two-walls/R2 (same 70B, QLoRA infra reused, ~3 GPU-h); open → has headroom |
| **Mid (SFT)** | **Qwen2.5-14B (or a Qwen-Coder variant)** | BFCL trainable-open band + code/tool inductive bias; clearest lift; gives the lift-vs-size gradient |
| **Open ceiling** (no train) | Qwen3.x-mid / DeepSeek-V3.2 | brackets where strong open models saturate |
| **Frontier ceiling** (no train) | Sonnet (API) | report-priors ceiling |
Key output = the **lift-vs-size gradient** ("SFT lift large for weak models, small for strong" =
generalizes the report's *force-the-tool-for-weak* into *train-the-trajectory-for-weak*). MoE giants
(DeepSeek V4 / Kimi K2.6 / GLM-5.1) are NOT trained — they saturate + QLoRA-immature.

## 9. ⚠️ Failure-mode pre-emption (the heart — "no predictable failures")
| Predictable failure | Source | Design control |
|---|---|---|
| from-base RLVR starves | two-walls / Phase 8-9 | **SFT-first**; RLVR/DPO only after a stable evaluator + warm-start SFT (R2) |
| training the validity judgment loses to the tool | provenance-over-trust / phase4 | validity stays `check_value_validity`; DV = loop mechanics only |
| circular reward | refuted decision-value | hidden truth scored AFTER action; never reward "matches the tool's output" |
| reward-hack (existence only) | 78/9 | reward = attribution-match, not existence |
| abstain-attractor (always-defer) | A70 starvation | defer_penalty > 0 + penalize defer when a record exists |
| leakage (label/keyword) | sham-leak / CT-keyword | no truth in observation (SCHEMA) + leak-guard test + regex baseline first |
| format-only artifact | Phase 8 "format" | format-only control isolates policy learning |
| single-seed / underpower | throughout | multiple seeds · held-out split · bootstrap CI |
| by-construction circularity | report 0.82-caveat | eval truth = held-out CT split; attribution vs held-out gold |
| weak-model floor | autonomous-uptake | model-band pilot (Gate 1) |

## 10. Go/no-go gates (must pass before the full run)
- **G1 headroom**: prompting-only clearly below ceiling AND above floor. Else re-pick model/tasks.
- **G2 baseline confound**: the regex/feature-row baseline must NOT already hit the trajectory score.
- **G3 evaluator stability**: leak-guarded deterministic scorer + the CT split source (✅ resolved:
  `exports/ct/negbiodb_ct_splits.csv`) before any training.

## 11. Reuse map (no new engine — PROJECT_BRIEF rule)
- trajectory/reward/preference/feature-row schema + scorer → `LLM_SFM_interpretability/experiments/trust_cue_attribution/`
- deterministic policy / shuffled-inverted controls / conformal → `bio-sfm-trust-core`
- the tool → `Negative_result_DB/src/nullatlas_mcp` (CT-wired, 6 tools)
- held-out truth → `Negative_result_DB/exports/ct/negbiodb_ct_splits.csv` (+ `export_ct_ml_dataset.py`)
- SFT QLoRA loop → NegBioRL `Negative_result_DB/scripts_rl/` (R2 / Phase-7)
- manifest/experiment pattern → `bio_sfm_designer`
- **New (small): a NegBioDB-CT adapter (claim/record → standardized evidence_packet) + a CT scorer
  (held-out registry truth) + the balanced-action task set. That is all.**

## 12. Scope / honesty
- Positive = "SFT installs the tool-use loop mechanics (generalizes the report's SFT leg to the trajectory
  axis); facts = retrieval, validity = the deterministic tool" — the NegBioDB half of a two-specialist
  thesis (the structure-predictor track carries the ipTM-trust half).
- Honest limits: SFT is *distillation* of frontier trajectories (not "the model learned it itself"); the
  validity judgment belongs to the tool, not training; single-CT does not test tool-SELECTION (→ multi-domain
  extension) nor fallible-specialist numeric trust (→ the structure track). Do NOT claim ipTM-style trust-routing.

## 13. Executable sequence
1. Build the NegBioDB-CT adapter + the balanced-action task set on the held-out split (no API/GPU).
2. Build/port the deterministic scorer + leak-guard tests (G3).
3. **Gate-1 pilot** (no training): prompting-only on the candidate weak open models + the regex/feature-row
   + deterministic-policy baselines + the frontier reference → locate the headroom band; check G1, G2.
4. Only if G1–G3 pass: generate expert trajectories → SFT (Llama-70B + Qwen-14B) → eval → the lift-vs-size gradient.
5. (Later) warm-start DPO/RLVR; multi-domain tool-selection; the structure-track complement.

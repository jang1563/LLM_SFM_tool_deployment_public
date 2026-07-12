# NullAtlas post-train RL ENVIRONMENT — design (2026-06-26)

This design turns the NullAtlas multi-DB tool-use trajectory into a **post-train RL
environment** whose verifiable reward (computed from the negative-results DBs) rewards reaching a
correct, grounded answer with **fewer failures / less trial-and-error**. Researched via an 8-agent
workflow (6 grounded research dims + synthesis + an adversarial critic). The critic returned
**GO-WITH-FIXES**; this doc is the design **with the blocking fixes baked in** — do not build the
pre-fix version.

## 0 · Honest scope (read first — this is NOT a decision-accuracy or fact-learning project)
- **Decision accuracy is already maxed.** CT G1: a cheap open 14B does the 5-action trajectory at
  **0.94 ≈ Sonnet 0.95** prompting-only with an oracle tool; a **no-LLM rule beat the agent 0.927 vs
  0.847**. R2: RL-on-SFT only *holds* the SFT ceiling (0.301≈0.29), never exceeds it.
- **Facts don't train in (info wall):** long-tail provenance forced-MCC ~0 at every scale. Facts stay
  in the tool at inference; we train *behavior*, never memorize NCTs.
- **What this env actually trains (the only honest headroom):** (1) **efficiency** — fewer/cheaper tool
  calls, no redundant retries at equal MCC (ARPO: equal accuracy at *half* the tool budget is
  trainable); (2) **robustness** — one prompt-free policy that calls `resolve_entity` and does NOT
  over-defer under fuzzy names (where even frontier models false-deprioritize 70–83%); (3) **knowing
  when to stop** (completeness without over-querying).
- **The deliverable claim:** *"post-training reduces tool-call cost and false-deprioritization at equal
  decision MCC, in a single prompt-free policy."* If the efficiency Δ is below the eval noise floor, the
  honest outcome is a **documented null on a pre-registered, cold, ≥500-item, multi-seed efficiency
  eval** — itself a publishable two-walls-consistent finding. Novelty (cite-and-distinguish Biomni-R0):
  first RLVR env rewarding *efficient multi-DB grounding of NEGATIVE/absent evidence with abstention
  reward-neutral*.

## 1 · Framework — TRL `environment_factory` (+ verifiers as the paper artifact)
Build on **TRL `environment_factory`** (OpenEnv contract) + QLoRA(4-bit nf4) + vLLM colocate, single
Expanse node (`gpus=h100:2`, account <allocation>, `<gpu-partition>`). It is the only option that supports
QLoRA-GRPO + the multi-turn tool loop with no new orchestrator/Ray/SGLang.
- ⚠️ **Fix (framework over-claim):** the repo's GRPO leg (`70b_phase9A_a70_grpo_forced.py`) is
  **single-turn**; the multi-turn `environment_factory` path is **NET-NEW code**. Build it with its own
  unit + a **CPU replay harness** (scripted policy through `NullAtlasEnv`) validating token-masking,
  tool-return inlining, and `StopEpisode` **before any GPU**; budget a separate 2–3 GPU-h
  colocate+QLoRA-70B+env_factory memory/throughput smoke.
- **Fallback / paper artifact:** re-express the env as a `verifiers` `vf.StatefulToolEnv` + `vf.Rubric`
  (mount the FastMCP NullAtlas server) and publish to the Environments Hub for citeability — but do NOT
  use the verifiers RL trainer for QLoRA-GRPO (LoRA there is SFT/DPO-only, DP-only as of early 2026).
  veRL/SkyRL = scale-out only (Ray/SGLang we don't run); ART = dark-horse if the loop is cleaner as
  free-form Python (keep RULER for SOFT axes only, never the fact reward).

## 2 · Action space — the 6 existing NullAtlas tools (in-process), 5 learnable + 1 gate
Wrap the python funcs in `src/nullatlas_mcp/server.py` **directly** (do NOT route the hot rollout loop
through MCP stdio). Learnable tool methods on `NullAtlasEnv`:
1. `resolve_entity(name)` — fuzzy synonym/entity-resolution (the trained sub-skill; the fuzzy-finding lever).
2. `survey_prior_failures(drug, indication)` — grounding/retrieve; **`as_of_date` auto-injected from
   `self` inside the method** (never a model arg → no-look-ahead enforced deterministically).
3. `get_trial_design(nct)` — drill-down (served from **pre-cache**, see §6 fix).
4. `verify_claims(claims)` — the only cross-DB router (CT + ChEMBL/DTI + ClinGen/VP-slice + PubMed).
5. `check_completeness(drug, indication, cited_ncts)` — coverage/stop-check (held-out, see §3 fix).
6. `submit_decision(action, cited_ncts)` — TERMINAL.

**Deterministic env-side gate (NOT learnable):** `check_value_validity` (`_validity.py`) auto-applied to
every cited record; a final answer citing a value-invalid record is auto-penalized. **Honest scope: 5 of
6 tools are CT-only; multi-DB reach is ONLY via `verify_claims`. depmap.db is 0 bytes; DC/PPI/GE/CP/MD
have negative-results tables but no survey tool. Ship CT-primary + a cross-DB claim-verification
minority; do NOT claim a wired 8-domain env.**

## 3 · Reward — 5 terms, MO-GRPO per-term-normalized, with the BLOCKING fixes
Compose: `R = Σ_i w_i · groupnorm_i(R_i)`, **each term MO-GRPO-normalized within the GRPO group
separately before summing** (arXiv:2509.22047 — else the high-variance efficiency term dominates →
always-defer collapse). None-valued terms exclude that sample from the term's group-stats (assert the
remaining sub-group is not size-1).

- **R_decision** (w≈1.0 but see the rebalance fix) — ternary à la TruthRL (arXiv:2509.25760): correct
  grounded-negative = +1, **correct abstain/defer = 0 NEUTRAL** (never penalized → prevents GRPO
  re-learning the 47/91% fabrication), fabricated/false-deprioritized = −1.
  - ⚠️ **BLOCKING FIX 1 — real action vocabulary.** The 5-action narrative (ground/reject/defer/verify/
    flag) does **NOT exist** in `ct_guardrails._ACTION_BY_STATE` (which is
    {pause_or_redirect, stop_same_context, request_more_evidence, no_negative_record_found,
    continue_standard_due_diligence, continue_with_context_review, review_candidate_records,
    ask_for_specific_pair}). **Re-key `submit_decision`, R_decision, R_false_deprior to these REAL
    strings before any code.** Map the 5 narrative classes → the real vocab explicitly.
  - ⚠️ **FIX — oracle-as-set:** `_ACTION_BY_STATE[state]` is a LIST. Define +1 iff action == the
    per-state **PRIMARY** (first) action; the rest = "acceptable, 0". Give **no credit** to
    `request_more_evidence` unless the state requires it (it covers 3/6 states → kills the generic
    always-request exploit).
- **R_completeness** (graded dense [0,1], the anti-starvation lever) — registry-grounded coverage from
  `completeness.py`.
  - ⚠️ **BLOCKING FIX 2 — citation-copy collapse.** As written, completeness scores cited NCTs against
    the **same survey payload the agent already saw** → the dominant collapse is "call survey once,
    paste all NCTs, submit modal action" (string-copy, not skill; maxes completeness + citation +
    efficiency at once). **Hold out a random subset of registry NCTs from the survey return; reward
    completeness ONLY against the held-out set** (forces a better-resolved / `by_class` second call to
    recover them). If that's too fragile, demote completeness to **eval-only**.
- **R_citation** (decoupled, deterministic vs DB; R1-Searcher/ToolRL) — +1 if cited record EXISTS in
  NegBioDB AND its entity-pair matches the resolved query; −1 if non-existent/wrong-pair
  (misattribution). **Do NOT reward DB-routing by claim-shape** (it's a deterministic regex router in
  `claim_guard.verify_one` → the keyword-confound trap of `ct-utility-provable-claims-pivot`).
- **R_false_deprioritization** (asymmetric, penalty-only, w≈0.8 > a miss) — fires when the model
  deprioritizes off a fuzzy **wrong-pair** match (an expensive deployment error).
  - ⚠️ **BLOCKING FIX 3 — fuzzy-tier null makes defer safe.** completeness=None on the synonym/fuzzy
    tiers AND the ~30% no-record class → the lowest-variance safe equilibrium is **over-deferring on
    fuzzy entities = the false-deprioritization the env exists to prevent.** Replace the null with an
    explicit **POSITIVE reward for calling `resolve_entity` and converting resolution→'ok'**, and an
    explicit **penalty for deferring/empty-handling without having attempted resolution.** Make
    *resolve-then-survey* the rewarded path on fuzzy tiers.
- **R_efficiency** (w ramped 0→0.1 via curriculum AFTER grounding stabilizes) + `R_redundancy`
  (−β·duplicate calls, β≈0.3, deterministic).
  - ⚠️ **FIX — group-relative → absolute per-class budget.** Group-relative (sigmoid vs group mean)
    rewards the minimum-call equilibrium with no gradient to climb out (if the group converges to 1
    call, everyone scores ~0.5). Use a **per-class ABSOLUTE minimal-budget** reference from
    `required_tools_for_action` (defer/verify need 2, others 1) so the gradient survives group
    convergence.
- **HiPRAG gating** (arXiv:2510.07794): efficiency/redundancy pay only when decision correct + format
  ok → can't earn the efficiency bonus by never calling the DB.
  - ⚠️ **FIX — decision-moot contradiction.** With w_dec=1.0 + HiPRAG gating everything behind decision,
    the dominant gradient IS the maxed decision. **Rebalance:** keep the gate (correctness is a floor)
    but make the *post-gate* gradient dominated by efficiency + false-deprioritization (the real
    headroom), and report the headline on those, not decision MCC.

## 4 · Two-walls-aware training recipe (3 stages, ~15–20 GPU-h/cycle)
**STEP 1 — SFT-trace generation (~0 GPU).** From-base GRPO STARVES here (frac_reward_zero_std=1,
grad_norm=0 — base confidently wrong). Generate gold multi-turn traces by running Sonnet 4.6 AND/OR a
deterministic ORACLE teacher through the SAME `NullAtlasEnv`; format `{messages:[system, user-proposal,
assistant tool-call, tool-result, assistant-final]}` like `phase9C_calib_sft.py` / `84_hint_r2_*`.
- ⚠️ **Open risk (teacher mismatch):** the oracle teacher is minimal-call but off-distribution for a
  70B; Sonnet is on-distribution but over-calls under fuzziness (the G1 finding). Neither is ideal for an
  *efficiency* objective → pilot both, pick by post-SFT trajectory naturalness + call-count.

**STEP 2 — SFT cold-start (mandatory, ~1.5 GPU-h).** Clone `phase9C_calib_sft.py` (TRL SFTTrainer +
LoraConfig). Installs sampleable trajectory behavior (base split_rate 0.033 → SFT ~0.90).

**STEP 3 — GRPO refine on the SFT adapter (~8–12 GPU-h).** Clone `70b_phase9A_a70_grpo_forced.py`.
- ⚠️ **FATAL bug to avoid:** NEVER `merge_and_unload()` on a 4-bit base when warm-starting from the SFT
  adapter (lossy silent no-op = the exact bug that starved the first R2 warm-start to 0.026). Use
  `PeftModel.from_pretrained(..., is_trainable=True)` after `prepare_model_for_kbit_training`,
  `peft_config=None`.
- Swap the single binary reward for `nullatlas_reward` (reuse the `src/negbiorl/rewards.py`
  DEFAULT_REWARD_FUNCS/weights pattern). `GRPOConfig(use_vllm=True, vllm_mode='colocate',
  num_generations=8, max_completion_length≥4096` (caps TOTAL tokens across all turns — else episodes
  truncate before submit_decision`), scale_rewards='group')`. **StarPO-S** (arXiv:2504.20073): drop
  zero-std groups, keep KL/entropy, prefer outcome-at-final-state (avoid the multi-turn Echo Trap).
- **GATE-BEFORE-SPEND:** 40-step smoke first; grep `[reward]` for `batch_std>0` /
  `frac_reward_zero_std≈0`. **Pre-register** the fallback (DAPO dynamic-sampling + `A=2r−1`), don't
  discover it at smoke time.

**GPU budget (1,800-hr cap):** ~0 trace-gen + 1.5 SFT + 1.5 smoke + 8–12 GRPO + 1.5 eval ≈ **15–20
GPU-h/cycle; 3–4 reward-shaping cycles ≈ 60–80 GPU-h (<5% of allocation).**

## 5 · Eval (pre-registered, cold, with the no-LLM floor)
- ⚠️ **Info-wall leak fix:** SFT traces bake specific NCT tokens → memorization on seen entities (won't
  transfer, forced-MCC~0). **Entity-disjoint split** (`export.py` cold_compound + cold_target) between
  SFT/GRPO/eval; report the headline Δ **ONLY on the cold split**; pre-register that seen-entity gains
  are expected and discarded.
- ⚠️ **Missing no-LLM floor:** the eval MUST include the **scripted minimal-call rule-based policy** as
  the floor (the no-LLM rule beat the agent 0.927 vs 0.847). If RL doesn't beat the script on
  Δ-efficiency-at-equal-MCC, **there is no result.**
- ⚠️ **Power calc:** compute the minimum detectable Δ-tool-calls at 80% power given per-item call-count
  variance BEFORE sizing; ≥500 items, multiple seeds, bootstrap CI. (The old n=120–300 binary-MCC evals
  had ±0.11 CIs — the efficiency Δ may be smaller.)
- Metrics: decision MCC (held equal), **mean n_tool_calls, redundant-call rate,
  false_deprioritization_rate, completeness_pct.** Headline = Δ-efficiency + Δ-false-deprior at equal MCC.

## 6 · Build roadmap (order; CPU-validate before GPU)
1. `src/negbiorl/phase10/` task builder (clone `81_hint_r2_build.py:to_records`): emit
   `{system, proposal, as_of_date, oracle_state, oracle_actions, registry_failure_ncts,
   gold_completeness_set (HELD-OUT subset), gold_cited_ncts, resolver_tier}` from CT DB at gold/silver
   tier; CT proportions {ground.35/defer.30/verify.15/reject.10/flag.10} **mapped to the real
   `_ACTION_BY_STATE` vocab**; leak-guard assert.
2. `nullatlas_env.py`: `NullAtlasEnv` (reset + 6 tool methods wrapping in-process funcs; as_of_date from
   self; accumulate n_calls/redundant/used_synonym/cited; submit_decision terminal; env-side validity gate).
3. `nullatlas_reward(environments, **kwargs)`: the 5 fixed terms (real-vocab ternary; **held-out**
   completeness; decoupled citation; **resolve-then-survey** fuzzy term; **absolute per-class**
   efficiency + redundancy); MO-GRPO per-term norm; HiPRAG gate (rebalanced). **Unit-test on hand-built
   trajectories (correct / fabricated-NCT / false-deprior-off-fuzzy / redundant / citation-copy) before
   any GPU.**
4. **Pre-cache** get_trial_design + PubMed for the task set into sqlite; **a cache miss RAISES** (fail
   loud), never falls through to live API during rollout; assert full coverage.
5. **CPU replay harness** — scripted policy through `NullAtlasEnv`; validate token-masking, tool-return
   inlining, StopEpisode. (net-new multi-turn code.)
6. SFT traces (Sonnet + oracle, pick by naturalness/call-count) → SFT cold-start.
7. GRPO refine (env_factory + nullatlas_reward + INIT_ADAPTER no-merge; StarPO-S + DAPO fallback) →
   40-step smoke gate (batch_std>0) → full 1200-step.
8. Curriculum: ramp w_eff 0→0.1 + resolver_tier synonym→fuzzy→exact AFTER grounding stabilizes.
9. Cold-split eval (≥500, seeds, bootstrap CI, **no-LLM floor**, pre-registered metrics); headline =
   Δ-efficiency + Δ-false-deprior at equal MCC; documented null if below noise floor.

## 7 · Reuse (exact files)
`src/nullatlas_mcp/`: `server.py:53-170` (6 tools = action space), `completeness.py:completeness()`
(dense term), `verifier.py:TrialClaimVerifier`+FLAG set (hard-flag ground truth), `claim_guard.py:
BiomedicalClaimGuard.verify_one` (multi-DB router), `_retriever.py:search_prior_failures` (fuzzy hook),
`_validity.py:validity_failures()` (env-side gate), `moa.py:_POS=6.0/_NEG=4.5` (DTI oracle).
`scripts_rl/`: `81_hint_r2_build.py` (builder), `phase9C_calib_sft.py`/`84_*` (SFT), `70b_phase9A_a70_
grpo_forced.py` (GRPO, the no-merge adapter pattern), `rewards.py` (reward API).
`slurm_nairr/run_hint_r2_grpo_expanse.sbatch` (SLURM). From `LLM_SFM_tool_deployment/negbiodb_ct/`:
`baselines.score_decision` + `tasks_*.jsonl` (the verifiable scoring + task seed).

## 8 · The 3 blocking fixes (pre-code checklist — do NOT write code until done)
1. **Real action vocabulary** — re-key submit_decision + R_decision + R_false_deprior to the actual
   `_ACTION_BY_STATE` strings; map the 5 narrative classes explicitly; per-state PRIMARY action for +1.
2. **Held-out completeness** — hold registry NCTs out of the rollout-visible survey return; reward
   completeness only vs held-out (or demote to eval-only). Kills the citation-copy collapse.
3. **Anti-starvation that survives fuzzy/no-record** — absolute per-class efficiency budget +
   resolve-then-survey positive reward on fuzzy tiers (not a null that makes defer safe). Pre-register
   the DAPO + A=2r−1 fallback. Else the 40-step smoke re-hits grad_norm=0.

Verdict (adversarial critic): **GO-WITH-FIXES** — honest, two-walls-consistent, defensible novelty;
worst case is a publishable documented null. Proceed only after the 3 blocking spec fixes.

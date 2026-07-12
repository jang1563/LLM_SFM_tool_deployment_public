# Band-Reranker Post-Training Environment — Design (2026-06-28, corrected)

The **defensible** version of the NullAtlas/NegBioDB post-training environment the original rejected idea was
not. This file now reflects the de-leaked A2 reranker results: SFT is the working deployable recipe; from-base
GRPO is a useful diagnostic but did not install the hard in-band disambiguation.

Research+design+adversarial workflow -> **BUILD-WITH-CHANGES (descoped)**.

## Why this is the defensible post-training target
The original RL-env (train the policy to do the negative-evidence trajectory/calibration) was killed because:
hidden lever (the retriever the policy can't see) · deterministic-ceiling decision · SFT captured all gains ·
training self-refuted "training unnecessary". The **band reranker** inverts all four:
1. **Verifiable reward** = exact MONDO-id match (not a hidden retriever's behavior).
2. **The policy IS the decision-maker** — it picks among already-retrieved candidate ids → clean credit assignment.
3. **Dense reward** — every band case has a gold (no abstain-attractor void).
4. **Infra already built** — `band_cases.json` = task instances; gold = reward; band precision = metric.

## Measured starting point (gold=MONDO id, de-leaked)
The held-out query originally re-entered the displayed synonym list for the gold candidate, inflating all model
scores. After removing the query from candidate synonym displays, the honest band_cases pick numbers are:

| method | de-leaked band_cases |
|---|---:|
| string-match floor | 0.285 |
| SapBERT top-1 | 0.750 |
| Qwen2.5-7B zero-shot rerank | 0.810 |
| Qwen2.5-1.5B SFT rerank | 0.875 |
| Claude-haiku reference | 0.900 |
| gold-in-top8 ceiling | 0.970 |

Band pool = 5,427 ambiguous rows (3,798 sapbert_high + 1,629 sapbert_med) over ~2,272 distinct MONDO ids,
supporting concept-disjoint splits at scale.

## The scientific hook (falsifiable two-walls test)
Does **from-base GRPO/RLVR work HERE** (verifiable reward, clean credit, policy=decider) where it starved on the
original sparse/hidden-lever task? The de-leaked answer was narrower than the leaked draft suggested:

- SFT installs the useful reranker: band_cases 0.785 -> 0.875.
- From-base GRPO does **not** install the hard in-band disambiguation: band_cases 0.785 -> 0.755.
- GRPO does learn part of the full task: band_eval 0.402 -> 0.710, mainly by learning the abstain gate.

So the corrected result supports the original practical lesson: retrieve/SFT the grounded decision; do not claim
from-base RL solved the real disambiguation without stronger controls.

## Prior art (verified)
- **BioELQA** (2402.15189): T5 over SapBERT candidates as MCQA → 93.5/94.5/85.2, beats bi-encoder + GPT-3.5 zero-shot (52.2).
- **BeLink** (2605.22501, real): Qwen3-8B set-wise instruction-tune over SapBERT top-20 + explicit NIL, +3-24%.
- **Rank-R1** (2503.06034) / **REARANK** (2505.20046): GRPO listwise rerankers; Rank-R1 matched SFT on 18% of data.
- **Spurious Rewards** (2506.10947): Qwen RLVR gains can be pretraining-elicitation artifacts → **non-Qwen control mandatory**.
- **Small-Model Learnability Gap** (2502.12143): ≤3B students degrade on long-CoT distillation → **distill the PICK only, no long CoT**.

## Plan (descoped, 2 tracks)
### Track A — PRODUCT (ships regardless of RL outcome, ~20-40 GPU-hr)
A tiny (0.5-3B) **zero-API** MONDO band reranker via SFT on gold MONDO labels.
- **Step 0** env scaffold `src/negbiorl_band/{band_env.py, reward.py}`; reuse the `band_cases.json` harness.
- **Step 1** data `scripts_ct/build_band_rl_dataset.py`: from 5,427 band rows → (query + SapBERT top-8 distinct
  MONDO + synonym sets, gold). Hard distractors from gold's is_a neighborhood. NIL cases = remove gold from
  held-out concepts. **CONCEPT-DISJOINT + NEIGHBORHOOD-DISJOINT split** (no eval concept's is_a parent/sibling
  may appear as a train gold OR distractor); report train/eval synonym token-overlap; show test precision stable
  id-disjoint vs neighborhood-disjoint (if it drops >3-5pp, gains were ontology memorization). Freeze the 200-case
  `band_cases.json` as untouched eval.
- **Step 2** target labels: use the gold MONDO id directly; no teacher-label distillation is needed.
- **Step 3** SFT LoRA on 0.5B/1.5B/3B; eval frozen-200 + concept-disjoint test-500.
- Realistic bar: tiny SFT near **0.875** is a win for zero-API deploy; claiming a 0.95 match is retracted.

### Track B — SCIENCE (single diagnostic probe, ~60-100 GPU-hr)
One tiny student, three legs: **from-base GRPO vs SFT vs FORMAT-ONLY/spurious-reward control**.
- **MANDATORY de-confounder:** re-run from-base GRPO on ONE **non-Qwen** student (Llama-3.2-1B / OLMo2-1B) —
  without it the two-walls conclusion is family-confounded.
- **3 seeds** on the decisive GRPO legs (small-scale RLVR is noisy).
- **Reward honesty:** exact-MONDO is the only +1; on REAL text the exact positive rate is only ~45% (realtext.log:
  0.85-0.95 band CORRECT 0.45-0.46, +hier 0.90-0.93) — either keep strictly exact (and admit it's less dense than
  claimed) or call the tiered version a hierarchy-distance reward (don't sell it as the clean dense contrast).
- **Reward-hacking guards:** abstain-frequency floor + NIL/calibration as a CO-PRIMARY metric; add
  "always-pick-SapBERT-top1" (0.750) and "always-pick-A" degenerate baselines.

## First experiment (~2-4 GPU-hr, completed)
Zero-shot and SFT band precision for Qwen2.5 family members fixed the real headroom after de-leaking. The key
current numbers are Qwen2.5-7B zero-shot 0.810, Qwen2.5-1.5B SFT 0.875, Claude-haiku 0.900, ceiling 0.970.

## Kill criteria
- tiny already >0.92 zero-shot OR 7B <0.90 → STOP (no headroom; ship retrieval/API deploy).
- neighborhood-disjoint test drops >3-5pp → ontology memorization; generalization claim void.
- GRPO gain < spurious-control +3pp OR doesn't replicate on non-Qwen → Qwen elicitation artifact; no two-walls claim.
- GRPO reward-hacks (abstain/NIL collapses) → null it.
- from-base GRPO frac_reward_zero_std≥0.9 / grad_norm→0 → the PREDICTED null (a result, not a bug — report it).

## Compute
Track A ~20-40 GPU-hr; Track B ~60-100 GPU-hr. A clean full grid is 300-700 GPU-hr, so keep the public claim to
the completed small diagnostic unless a broader multi-seed/non-Qwen sweep is explicitly run.

## Framing
Lead with the **product**: a tiny zero-API reranker that closes most of the gap to a proprietary reference on the
ambiguous band. Position GRPO as a small **diagnostic probe** whose partial/null result strengthens the
retrieve/SFT-over-from-base-RL framing.

# From Entity Resolution to a Post-Training RL Environment: the A2 / Band-Reranker Report

*NegBioDB / NullAtlas — negative-evidence grounding for clinical-trial failures. 2026-06-28.*
*All numbers below are measured (non-circular) unless labeled "projected". Honesty is the point.*

---

## 0. One-paragraph summary

We set out to turn a NullAtlas negative-evidence trajectory into a post-training RL environment. An adversarial
hyper-review killed the naïve version (it would have trained a metric the policy can't move), so we pivoted to
the real bottleneck — **free-text drug/disease → NegBioDB entity resolution** — and built it into a working,
open-model, zero-proprietary-dependency system. Along the way the *reranker* that fixes the resolver's one
weak spot turned out to be the **defensible** post-training target the original idea was not, and SFT yielded
both a **product** (a 1.5B open reranker that closes most of the proprietary-reference gap on a laptop) and a
**falsifiable science finding** about what from-base GRPO does and does not learn. This report synthesizes the
whole arc.

---

## 1. Problem & thesis

The program's published headline is a **two-walls** result: for negative-evidence *calibration*, fine-tuning is
INSUFFICIENT (from-base GRPO starves) AND UNNECESSARY (inference-time tool access recovers it) → **"you can't
train it in; you retrieve it."** But retrieval only works if a messy real drug/disease name can be **matched**
to the right NegBioDB entity. If matching fails, an agent wrongly concludes "no prior failure" — the dangerous
**false-deprioritization** that greenlights a doomed trial. Entity resolution *is* the deployment bottleneck.

---

## 2. A2 — free-text entity resolution (the retriever)

**The crux:** diseases have no clean entity id in the CT DB (do_id / icd / canonical all 0% populated; mesh too
coarse — one "Cardiovascular Diseases" mesh lumps >1,000 distinct conditions). We linked to **MONDO** (clean
per-disease id + is_a hierarchy).

- **Disease leg** (`resolve_disease.py`): dictionary-NER over MONDO labels + generic-root blocklist +
  obsolete-filter → **SapBERT** (`cambridgeltl/SapBERT-from-PubMedBERT-fulltext`) semantic match on the residue
  + antonym-veto + two-threshold abstain; grounding via Q ∪ descendants(≤2) ∪ parent with EXACT/SUBTYPE/ANCESTOR
  tags. Handles lay ("heart attack"→MI), abbrev ("STEMI"), typo ("Alzheimers"); abstains on non-disease.
- **Drug leg** (`resolve_drug.py`): dosage/salt-normalize (with a bare-cation guard) → exact in-registry chembl
  → **ChEMBL-alias KB** (113,739 name→chembl from `chembl_37` molecule_synonyms) → SapBERT → abstain.
- **Unified** (`resolve_a2.py`): a **3-way verdict** — `grounded` (recorded failures + NCT + relation tag),
  `no_recorded_failure` (both legs resolved, no record), `abstain` (a leg couldn't resolve) — which *never*
  conflates "couldn't resolve" with "no failure". Target-side crosswalk links are tier-tagged so a `grounded`
  verdict can't silently rest on a 50%-precise link.
- **DB artifact:** `migrations_ct/005_conditions_mondo.sql` + `scripts_ct/load_mondo_crosswalk.py` materialize
  conditions.mondo_id (60.4% of conditions; 73% of failure-bearing).

**Measured (leave-one-out on real registry surface variation, gold = chembl / MONDO id):**

| leg | naïve exact | + normalize | + SapBERT | + curated KB | resolver |
|---|---|---|---|---|---|
| **drug** | 0.000 | 0.533 | 0.609 | 0.690 | **0.745** (random-holdout 0.745, robust) |
| **disease** | 0.000 | fuzzy 0.668 | **recall@1 0.766** (random 0.729) | — | — |

**Precision (real CT.gov input, LLM-judged):** the ≥0.95 SapBERT tier is **95% correct** on real strings
(the recalibration validated beyond the ontology-synonym proxy); the honest false-ground (wrong disease) is
3–10% across tiers. **Embedder bake-off:** SapBERT **0.766 > BioLORD-2023-C 0.694** on our MONDO task — the
curated KB, not a fancier embedder, is the drug lever (`build_chembl_aliases.py`).

**Hyper-review (20-agent adversarial):** *no conclusion flipped.* Two top worries (chembl id-space, BioLORD
replication) were false alarms. Four MED defects were confirmed and **fixed** (NIL-masking → 0.95 gate is
~10% not 7.5% false-ground; target-side tiering; reranker underpowering; proxy-vs-real framing) + two deployment
bugs (salt→cation, substring antonym-veto). The rigor *strengthened* the result.

---

## 3. The band reranker (the precision lever)

The resolver's one weak spot is the disease SapBERT **0.85–0.95 ambiguous band** (~29% of resolutions, 50–66%
precise = sibling/parent/subtype confusion). A **closed-set reranker** — SapBERT's top-8 candidate concepts,
each shown with its full synonym set, pick the same-disease one or abstain — fixes it:

*(De-LEAKED numbers. A 2nd hyper-review found the held-out query re-entered each gold candidate's displayed
synonym list, so a no-model string-match scored 0.925 — matching every model. After stripping the query from the
displayed synonyms, the honest band_cases (pick) numbers are:)*

| method | de-leaked band_cases | (leaked, retracted) |
|---|---|---|
| string-match floor (no model) | 0.285 | 0.925 |
| SapBERT top-1 (retrieval, no rerank) | 0.750 | 0.750 |
| Qwen2.5-7B (open) rerank | 0.810 | 0.950 |
| **SFT-1.5B rerank** | **0.875** | 0.950 |
| Claude-haiku rerank | 0.900 | 0.965 |
| ceiling (gold-in-top8) | 0.970 | 0.970 |

It only *re-ranks retrieved ids* (never generates) — consistent with "retrieve, don't hallucinate". The reranker
is **real (+12.5-15pp over retrieval, far above the 0.285 string-match floor)** but the margin is smaller than
the leaked +20pp, and **the open models do NOT quite match the proprietary one** (SFT-1.5B 0.875, Qwen-7B 0.810
vs Claude 0.900). The genuine result is the **distillation win: SFT-1.5B (0.875) > Qwen-7B zero-shot (0.810)** —
a trained small model beats a bigger untrained one, so the stack still deploys API-free at a real (if smaller) cost.

---

## 4. The defensible RL environment (post-training the reranker)

The band reranker is the RL target the original idea was not: (1) **verifiable** reward = exact MONDO-id match;
(2) the **policy IS the decision-maker** (picks among retrieved ids → clean credit assignment); (3) **dense**
reward; (4) infra already built. Data: `build_band_rl_dataset.py` → 23,696 train / 3,189 eval, concept-disjoint,
gold-in-top8 0.55 on the *full* band (harder than the curated sample; task = pick-**or-abstain**).

### Track A — product (SFT on gold MONDO labels; de-leaked)

| student (SFT LoRA) | band_cases (pick) | band_eval (full) |
|---|---|---|
| Qwen2.5-1.5B | 0.790 → **0.875** | 0.371 → 0.773 |

**A SFT'd 1.5B reaches 0.875 (de-leaked) — a real +12.5pp over SapBERT (0.750) and above Qwen-7B zero-shot
(0.810), but slightly below Claude (0.900), so the old full-parity wording is withdrawn.** It is SFT on the
gold MONDO id (not teacher distillation — a misnomer in an earlier draft). Wired into `BandReranker(backend="local")`
for a zero-API laptop deploy (the N=30 MPS check is a functional smoke test, CI [0.83,0.99], not a precision
number). → the stack is zero-proprietary-dependency at a real, honest cost (SFT-1.5B 0.875 vs Claude 0.900).

### Track B — science (does from-base GRPO work here?)

Manual single-token GRPO (`grpo_train.py`). **De-leaked (Qwen2.5-1.5B), splitting the two sub-tasks:**

| task | base | SFT | from-base GRPO |
|---|---|---|---|
| band_cases (in-band **disambiguation**) | 0.785 | **0.875** | **0.755 (no lift, ↓)** |
| band_eval (full = disambiguation + **abstain-gate**) | 0.402 | 0.773 | **0.710 (gate only)** |

**Findings (de-leaked — this CORRECTS the earlier leaked draft, which over-read "the wall is base-samplability"):**
1. **SFT installs the reranker** (band_cases 0.785→0.875) — the real, deployable result.
2. **From-base GRPO does NOT install the disambiguation.** It leaves band_cases flat/down (0.785→0.755) and
   improves only band_eval (0.402→0.710) — i.e. it learns the trivial **abstain-gate** (abstain when the gold
   isn't in the top-8), not the hard in-band pick. The earlier "GRPO installs (+41/+70pp)" was (a) leak-inflated
   and (b) a gate-vs-disambiguation conflation, exactly as the 2nd hyper-review flagged.
3. **So the corrected result is CLOSER to the original two-walls, not a refutation of it:** from-base
   verifiable-reward RL is *insufficient* for the real closed-set decision (learns only the easy gate); SFT
   installs it. The "base-samplability wall" framing is withdrawn. (The Spurious-Rewards control and multi-seed
   sweep the design pre-registered were not run, so no strong RL claim is made either way.)

**Corrected answer to "post-train RL env viability":** the band reranker is a valid SFT target (0.875, real);
**SFT is the recipe that works; from-base GRPO does not install the disambiguation** — consistent with the
program's original "you retrieve/SFT it, you don't from-base-RL it."

---

## 5. Honest caveats

- Headline recalls are held-out-**synonym / registry-surface proxies**, not real CT.gov/PubMed query logs; the
  drug LOO applies to the 17% of chembls with ≥3 surface forms. Real-text precision was separately LLM-judged
  (§2). "Naïve 0.000" is a single-copy-holdout floor, not the deployed ~13% exact-first baseline.
- Earlier `grpo_train.py` output text over-read the *saturated* band_cases result. The corrected interpretation is
  split by task: band_cases tests hard disambiguation, while band_eval also rewards the easier abstain gate.
- from-base GRPO here is a *single-seed* probe; the family control (OLMo) is the load-bearing de-confounder, but
  multi-seed would harden it further. NAIRR's 1,800-hr cap was deliberately NOT spent on this.

---

## 6. Deployment & what remains

**Deployable now, API-free:** `A2Resolver(rerank=True)` with `BandReranker(backend="local", adapter_dir=…)` —
SapBERT retriever + SFT-distilled 1.5B reranker + ChEMBL/dict KBs, runs on any GPU or a laptop.

**RxNorm tier — measured, NEGATIVE (honest science):** UMLS license approved 2026-06-28; the public RxNav API
resolves brand→ingredient (Vidaza→azacitidine, Tylenol→acetaminophen, verified). But on the 9,221 failure-bearing
drug residue the deterministic tiers miss, **RxNorm recovers only ~1–3%** — ChEMBL's TRADE_NAME synonyms had
already captured the standard brand→generic mappings, so RxNorm just overlaps. Aggressive normalization
(strip route/schedule/formulation) adds ~6% via the *existing* ChEMBL KB, and combo-detection ("plus"/"with")
~5% (→ correct abstain). **The remaining ~89% of the residue is genuinely non-chembl-able** — cell/gene
therapies, vaccines, experimental biologics, drug classes, phrases — so the resolver's abstain there is *correct*,
not a coverage gap. **→ the RxNorm tier was NOT built (no value); the lever for the recoverable residue is
normalization, not a new KB.** The projected "~85% with RxNorm" was falsified by measurement.

**Bottom line:** "cheap open model + retrieval, no proprietary API" is viable all the way to the
entity-resolution endpoint. Where post-training *does* help (the ambiguous band), a tiny open SFT model closes
most of the gap to the proprietary reference on a laptop (0.875 vs Claude-haiku 0.900), while from-base GRPO
learns the abstain gate rather than the hard disambiguation.

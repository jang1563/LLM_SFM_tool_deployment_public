# Free-text / out-of-registry entity resolution (A2) — design (2026-06-26)

The real open bottleneck the hyper-review redirected to: map **free-text** drug + disease names
(brand/abbrev/INN/research-code/misspelled/lay) to the right NegBioDB entity so recorded trial
**failures** can be retrieved — **without false-deprioritizing** (silently answering "no prior failure"
because resolution missed) and **without false-grounding** (matching an unrelated entity). A
retrieval/embedding problem, **NOT training/GRPO** (settled: the retriever is the scale-independent
lever). Researched via a 6-agent workflow (4 grounded dims + synthesis + adversarial critic).
Critic verdict: **GO-WITH-FIXES** — design below has the 7 fixes baked in.

## 0 · Problem + honest scope (3 mandatory outputs — never collapse)
The resolver sits **in front of** the existing D1 retriever (`scripts_rl/150` structured + `151` dense)
and must emit **three distinct** outcomes: **(a) resolved + linked**, **(b) resolved-but-no-failure-
record**, **(c) UNRESOLVED / abstain**. *Conflating (b) and (c) IS the false-deprioritization bug.*
- **Verified this session** (`data/negbiodb_ct.db`): interventions 176,741 rows, **chembl_id on only
  36,413 (21%)** → drugs are NOT "mostly solved by chembl" (the 0.48→0.91 fuzzy win lives entirely
  inside that 21%); conditions 55,915 rows, **do_id/icd/canonical_name 0% populated**, mesh_id 74% but
  **coarse** (one "Adenocarcinoma" mesh = 2,147 distinct condition_names; 14,719 rows have NO mesh).
  And `151_d1_dense_retriever.py:run_eval` uses `query = mat[i]` (the doc's own embedding) → its
  **0.973/0.997 recall is self-retrieval, CIRCULAR, do not cite**.
- **In-scope (solvable):** in-registry drug cross-forms (demonstrated); out-of-registry/lay/misspelled
  drugs (external alias index); disease normalization to a real concept id + a directional subtype rule.
- **Honest residual (abstain, a feature):** context-free ambiguous abbreviations (MI = myocardial
  infarction vs mitral insufficiency) → carry-both-then-abstain; child→parent failure transferability
  (no literature — validate, default OFF); the ~46% of CT.gov strings + 14,719 mesh-less rows that map
  to no ontology concept (permanent long tail → abstain, don't force-link); KB version rot (pin).

## 1 · Drug resolution — RxNorm spine + 4-tier cascade (each tier can abstain)
Canonical spine = **RxNorm ingredient RXCUI**, bridged to the existing `interventions.chembl_id` failure
index via **UNII**, so the proven chembl-keyed retrieval is reused unchanged.
- **T0 research-code/UNII:** local `drug_synonyms.json` (41,160 name→molecule keys; maps PF-/code names)
  + ChEMBL `molecule_synonyms` research_code. (RxNorm has NO research codes — omitting T0 silently
  false-deprioritizes the newest compounds.)
- **T1 brand/INN/lay/misspelled:** **offline** RxNorm full release + RXNCONSO `getApproximateMatch`
  (handles "lipitorr"→Lipitor, "hctz"→hydrochlorothiazide) → RXCUI → ingredient (IN) RXCUI → UNII →
  chembl_id. Offline (not the live RxNav API) so the eval is reproducible.
- **T2 embedding fallback** (the new piece, fires on the ~79% no-chembl + lay names): **SapBERT** ANN
  over an enriched alias index (RxNorm TTYs + ChEMBL synonyms + intervention surface forms) → chemical
  id → route via the `scripts_rl/155` **audited dedup GROUPS** (`d1_drug_groups.json`) so the over-merge
  audit (the "PF-04691502" research-code collision guard) still blocks false-grounding.
- **T3 abstain.** Deterministic front-end (`src/negbiodb_ct/drug_resolver.py` dosage/salt strip + exact
  synonym) runs FIRST (preserves the in-registry win at zero cost).
- ⚠️ **FIX (per-tier abstain):** T0/T1/T2 each need a **margin gate** (score floor + gap-to-second-best).
  RxNorm approx-match over-matches short/ambiguous 3-letter tokens/salts/combos; a tier below margin
  **falls through** to the next, never returns a confident id. ⚠️ **FIX (UNII bridge):** RXCUI→chembl
  via UNII is not 1:1 and UNII is missing for many research-code/biologic compounds → assert coverage;
  where UNII is absent, route via the dedup group, don't drop (else re-introduces false-deprioritization).

## 2 · Disease resolution — THE CRUX: MONDO + a directional, antonym-guarded hierarchy
The NegBioDB condition string **cannot** be the link target (no clean id; coarse mesh over-matches).
**Link BOTH the user free text AND each condition_name into MONDO** (the only ontology with a clean
per-disease concept id + an `is_a` hierarchy + curated SSSOM equivalence xrefs to MeSH/DOID/UMLS/
Orphanet/OMIM/ICD-11; UMLS-CUI rejected = not 1:1/licensed; mesh-tree rejected = billing-coarse).
- **Offline crosswalk (the real engineering):** materialize `conditions.mondo_id` via a NEW migration
  (`migrations_ct/NNN_conditions_mondo_crosswalk.sql` + join table). Per condition_name: normalize via
  the existing d1 condition signature (strip Stage IV/grade/recurrent/"(Part 1)") → exact/lexical MONDO
  label hit → misses to SapBERT ANN (UNION a BGE-M3 lexical leg) top-20 → hybrid rerank (cosine +
  rapidfuzz JaroWinkler) → τ_high accept / τ_low abstain / **middle band → human spot-check** (reuse the
  d1 kappa packet). Pin the MONDO release. The 74% mesh_id rows seed candidates free via MONDO's MeSH xref.
- **Hierarchy / subtype rule (directional — make-or-break against false-grounding):** a query at MONDO
  node Q grounds on a failure at F only if F ∈ {Q} ∪ descendants(Q) ∪ {immediate is_a parent of Q} —
  **NEVER siblings, NEVER >1 level up, NEVER naive is_a closure** (which recreates the mesh over-match).
  Relation tags: **EXACT** (direct grounding) · **SUBTYPE/DESCENDANT** (a STEMI failure for an MI query →
  advisory) · **ANCESTOR** (an MI failure for a STEMI query → lower-confidence advisory). Carry
  biomarker/negation tokens (HER2±, squamous/non-squamous) as HARD non-merge guards.
- ⚠️ **FIX (antonym over-match — the #1 disease risk):** SapBERT pulls hypo/**hyper**thyroidism, type
  1/**2** diabetes, **Hodgkin/non-Hodgkin**, RCC/renal-failure close (1-token/negation differences).
  Extend the biomarker guard to a **disease negation/antonym blocklist** (hypo/hyper, non-, -free,
  refractory-vs-naive, type 1/2, acute/chronic, benign/malignant): inject these as MANDATORY hard
  distractors in the reranker candidate set AND as a **post-ANN veto** (if the top candidate differs from
  the query only by a negation/antonym token → **force abstain**, never ground).
- ⚠️ **FIX (depth bound):** replace the arbitrary `depth≤2` with an **information-content / LCA-distance**
  bound (Resnik/Lin) — MONDO subtree density varies (cancer deep, cardio shallow), so a fixed edge-depth
  over/under-includes.
- ⚠️ **FIX (transferability default):** child→parent failure transfer has **no gold** → ship
  **EXACT-only grounding as the DEFAULT**; SUBTYPE/ANCESTOR advisories ship **DISABLED** until a
  pre-registered held-out hierarchy slice (2nd annotator) validates them; if validation is
  inconclusive they stay `advisory-experimental`, never counted as grounding.

## 3 · Embedding pipeline — two encoders, two indices (different jobs)
- **Entity-name index (new):** **SapBERT** (`cambridgeltl/SapBERT-from-PubMedBERT-fulltext`, 110M) —
  self-aligned on 4M UMLS synonym pairs → surface-variant/abbrev/typo robust, which a passage embedder
  lacks. ADD a **BGE-M3** leg (sparse/ColBERT) for research-codes/lexical overlap; **union** the top-k.
  Do NOT use BGE-M3 dense-only on short names (it blurs "Aspirin" vs "Aspirin DL-Lysine").
- **Record-passage index (already built):** keep **BGE-M3** for the failure-document retrieval
  (`scripts_rl/151`, 38,338 docs cached) — the NAIRR-earmarked job. SapBERT = short string→concept;
  BGE-M3 = passage retrieval.
- **Resolve → retrieve → rerank → abstain** per mention: (1) **normalize** (scispaCy AbbreviationDetector
  / Ab3P; for ambiguous abbrevs carry ALL expansions; dosage/salt strip; exact fast-path FIRST). (2)
  **retrieve** SapBERT+BGE-M3 ANN top-k≈20 (brute-force `(N,1024)` matmul — reuse `151 dense_topk`, no
  FAISS at a few-hundred-k strings). (3) **rerank** top 5-10 with a **closed multiple-choice Haiku call
  with an explicit None/abstain option** (NEVER an open "is X relevant to Y?" judge — D1 measured human
  κ=0.33 for open judging; MC-over-retrieved-set is the easier, calibration-robust task; the candidate
  set deliberately includes salt/sibling/abbrev-collision distractors). (4) **threshold/abstain**
  (`rrf_fuse` from 151).

## 4 · Precision / recall / abstain — the asymmetric cost IS the design
False-grounding (precision) is the **worse** error; false-deprioritization (recall) is recoverable;
**abstain costs only recall, never precision** (the guardrail treats absence as advisory). Two
separately-tuned thresholds on a held-out set that **plants out-of-registry/NIL queries**: τ_resolve
(bi-encoder floor → reranker) and τ_abstain (reranker-margin floor → abstain), τ_abstain set
conservatively to hold false-grounding ≈0. Report the full **precision / recall / abstain-rate curve**
(a single operating point hides the tradeoff), drug/disease split, Wilson CIs, validated across ≥2
rerank models (temp-0 API nondeterminism shifts numbers).

## 5 · Eval — non-circular, four streams, a baseline ladder
The D1 0.973 is circular (forbidden). Gold must NOT be the resolver's own KB/crosswalk.
- **A · perturbation** (scale, in-registry gold): reuse `build_fuzzy_tasks.query_form` (most-dissimilar
  sibling + no-leak assert) + injected misspell/abbrev/lay; gold = held-out chembl_id. ⚠️ **FIX:**
  this is SYNTHETIC noise from our own model → **not the headline**; report as robustness-to-known-
  perturbations only.
- **B · raw CT.gov condition strings** (real free text, zero construction): "NSTEMI - Non-ST Segment
  Elevation MI", "SLE (Systemic Lupus)". ⚠️ **FIX:** gold = mondo_id **only from human-adjudicated
  middle-band crosswalk rows**, evaluated on a **DISJOINT** set of strings the resolver did not see
  crosswalked (else circular at the disease leg).
- **C · human-curated hard set** (~100-200, 2nd-annotator κ): lay drugs (Tylenol/ASA/PF-06439535) +
  subtype-vs-parent disease + **hard NIL negatives** (true "no recorded failure" — must not over-match)
  + **sibling distractors** (STEMI must not ground on hemorrhagic-stroke). The only non-circular
  hierarchy gold; report its wide CIs honestly.
- ⚠️ **FIX · D · external non-synthetic stream:** harvest real drug+disease mentions from PubMed
  abstracts / CT.gov `brief_summary` free text (or real agent logs), human gold → **THIS is the
  deployment false-deprioritization headline**, not synthetic A.
- **Metric:** precision (1−false-grounding) and recall (1−false-deprioritization) reported **separately**
  per abstain threshold, stratified by bucket (brand/INN/research-code/misspelled · lay/abbrev/
  paraphrase/subtype), abstain-coverage curve; disease hierarchy uses **LCA partial-credit** (Kosmopoulos
  2015), never flat accuracy. **Baseline ladder** (each component's marginal Δ): regex → +fuzzy →
  +synonym(chembl/MONDO) → +SapBERT → +BGE-M3 → +rerank → +hierarchy. **Run the regex baseline BEFORE any
  embedding claim. 0.73 false-deprioritization is the number to beat.**

## 6 · THE BUILD GATE — measure crosswalk coverage FIRST (S2 before the architecture)
⚠️ **The single number that decides whether the disease leg is worth building** is the crosswalk
abstain rate: real MONDO coverage of the 55,915 condition strings is **unmeasured**. CT.gov strings
include healthy-volunteer / procedure / device / symptom / "Part 1" basket strings with NO MONDO
concept. **Run S2 (the conditions→MONDO crosswalk) as a standalone measurement first; report
mapped / no-concept / mesh-less stratified coverage. If abstain >~40% on common-disease queries, the
disease leg's value claim must be re-scoped BEFORE committing the architecture.** (Drug leg proceeds
regardless — RxNorm+SapBERT coverage is well-trodden.)

> **✅ GATE MEASURED 2026-06-26 — PASSED (disease leg VIABLE, no re-scope).** Downloaded MONDO
> (`mondo.obo`, 60,843 terms / 33,768 MONDO classes / 94,983 synonyms / 24,833 MeSH xrefs). Findings on
> `negbiodb_ct.db` conditions: **exact MONDO-label match only ~19%** — BUT the crux is the CT "condition"
> field is **messy free text (arm/population/intervention), NOT clean disease names** ("1 Hz rTMS to
> Pre-SMA", "0.5-14 Year Old Children With Nephroblastoma", "Medico-Economic Aspects"). A **longest-
> MONDO-label-substring extraction (a dictionary NER) + a generic-root blocklist** (drop "disease /
> disorder / syndrome / cancer / …" standalone + `MONDO:0000001`) resolves **61.9% of FAILURE-bearing
> conditions to a SPECIFIC, high-precision mondo_id** (spot-check 14/14 correct: "Stage IV Adrenal Cortex
> Carcinoma"→adrenal cortex carcinoma, "Fibroid"→leiomyoma, "Leigh Disease"→Leigh syndrome, typo-robust
> "Alzheimers"→Alzheimer disease). **Abstain 38.1%** = (a) SapBERT-recoverable abbrev/typo/novel/synonym
> ("2019-nCoV", "714leukemia", "STEMI", "- HIV") + (b) genuine non-disease (vaccines, procedures, economic).
> The blocklist traded ~7pp coverage (68.7→61.9%) for a large precision gain (kills the "→disease"/
> "→syndrome" root false-matches) — the right trade (precision ≫ recall here). **DESIGN MODIFICATIONS
> (now baked into §2):** (i) add a **mention-extraction front-end** (longest-MONDO-substring dictionary
> NER — the strings are arm/population free text, so normalize→match alone undercounts at 19%); (ii) the
> **generic-label blocklist** is mandatory; (iii) SapBERT semantic match is the residue-recovery tier on
> the 38%, not the primary. The 62%-floor → SapBERT-residue → honest-abstain split is the disease leg.
>
> **✅ SapBERT RESIDUE MEASURED 2026-06-26 (disease leg COMPLETE, ~72% coverage).** Ran SapBERT
> (`cambridgeltl/SapBERT-from-PubMedBERT-fulltext`, CLS, MPS) — embedded the 93,060 MONDO labels (cached
> `a2_freetext/mondo_sapbert.npy`) + a 250-sample of the failure-bearing dict-abstain residue, cosine ANN.
> **Sim distribution:** ≥0.90 **26%** (correct recovery — the plural/word-order/synonym variants the
> dictionary's exact-match missed: "Cytomegalovirus Infections"→cytomegalovirus infection, "Breast
> Neoplasms"→breast neoplasm, "Bacterial Infections"→bacterial infectious disease); 0.85–0.90 6%; **<0.80
> 60% — correctly ABSTAINS** on non-disease ("Imatinib Mesylate" the DRUG, "Racial Bias", "Adolescent
> Health", arm fragments). The **<0.80 threshold cleanly separates** real-disease recovery from non-disease.
> **→ COMBINED disease coverage ≈ dictionary 61.9% + SapBERT(≥0.90) 26%×residue = ~72% confident
> high-precision MONDO id, with ~26% honest abstain** (drug names / procedures / arm fragments /
> out-of-MONDO). The "impossible — no entity id" disease crux is SOLVED. Operating point (τ≈0.85–0.90) to
> be tuned on the eval set; the SapBERT tier also subsumes a lemmatization-improvable slice of the
> dictionary tier. Persistent artifacts: `a2_freetext/{mondo.obo, mondo_sapbert.npy, sapbert_residue.py}`.

## 7 · Build steps + GPU plan (<<50 GPU-h, mostly off-allocation, NO training)
S0 pin & download offline KB versions (MONDO/RxNorm+RXNCONSO/ChEMBL) → S1 cheap SapBERT-vs-BGE-M3
ablation on a held-out abbrev/typo/brand slice (confirm SapBERT wins on short names) → **S2 (GATE) the
conditions→MONDO crosswalk + coverage report** → S3 build the enriched drug alias index + MONDO name
index, embed both (SapBERT + BGE-M3, reuse `151 embed_texts` chunked cache) → S4 implement
`src/negbiodb_ct/resolver_freetext.py` (normalize → fast-path → ANN → Haiku MC-with-None rerank →
two-threshold three-output API; per-tier abstain; antonym veto; directional MONDO hierarchy with relation
tags, EXACT-default) → S5 build eval A+B+C+D (disjoint, human gold, NIL+sibling, assert query≠key) → S6
baseline ladder + precision/recall/abstain curves + LCA partial-credit + Wilson CIs across ≥2 rerank
models → S7 tune τ to hold false-grounding ≈0, lock the operating point, write the honest residual into
the report → S8 wire the resolver as the **query-side front-end** to 150/151 (do NOT rebuild the doc
index). **GPU:** SapBERT name index ~minutes on one H100 (off-allocation, even Cayuga); BGE-M3
entity-name leg via `slurm_nairr/d1_bge_m3_embed_expanse.sbatch` with a `--target entity_dict` mode;
one-time batch <<50 GPU-h; rerank = Haiku API (no GPU). Bulk of the 1,800-hr cap intact. Acknowledge
"NAIRR Pilot and SDSC Expanse AI."

## 8 · The 7 fixes (pre-build checklist)
1. **Disease antonym/negation blocklist** + post-ANN veto (force abstain on negation-only differences).
2. **Per-tier abstain** (margin gates on T0/T1/T2, not just T3).
3. **Crosswalk-coverage GATE** — run S2 first; re-scope if disease abstain >~40%.
4. **Disease gold human-only** + disjoint string set (kill the source-B self-grading circularity).
5. **External non-synthetic eval stream D** (PubMed/CT.gov free text) as the deployment headline.
6. **Child→parent transferability: EXACT-only DEFAULT**; advisories disabled until validated.
7. **IC/LCA-distance hierarchy bound** (not fixed depth) + **UNII-bridge coverage** assert/fallback.

## 9 · Reuse
`scripts_rl/151` (embed_texts/dense_topk/rrf_fuse harness — feed entity NAMES), `150` (the structured
retriever the id feeds), `fuzzy_resolvers.py` (exact/fuzzy/synonym ladder = drug front-end + the
baseline to beat), `src/negbiodb_ct/drug_resolver.py` (normalizer), `d1_retrieval/drug_synonyms.json`
(41,160 T0 keys) + `d1_drug_groups.json` (155 audited dedup = routing target) + `d1_condition_groups
.json`/`d1_condition_dedup_summary.json` (condition normalizer), `build_fuzzy_tasks.query_form` (eval A),
`195_end_to_end_fuzzy.py` + `E2E_FUZZY_RESULT` (free-text harness for eval D), `slurm_nairr/d1_bge_m3_
embed_expanse.sbatch` + `bge_m3_cache/`, the D1 human-kappa packet (eval C + crosswalk spot-check).

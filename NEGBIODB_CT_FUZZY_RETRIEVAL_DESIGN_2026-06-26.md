# NegBioDB-CT Fuzzy-Retrieval Experiment — Design (2026-06-26)

Follow-up to the G1 tool-use trajectory result ([[ct-trajectory-tooluse-g1]]): under **oracle retrieval**
(tools query the task's known intervention_id/condition_id) a cheap open model does the 5-action
negative-evidence trajectory honestly and accurately (Qwen-14B 0.94 ≈ Sonnet 0.95). The one open honest
axis is the **oracle caveat** — real deployment must *resolve a drug name to the right records*, and the
registry records one drug under many names. This experiment tests whether the recipe survives that.

## 1. The realistic fuzziness source (no synthetic noise)
Probe of `data/negbiodb_ct.db` (2026-06-26):
- `intervention_name` == `canonical_name` (0% divergence — normalized). DEAD as a source.
- **6,020 `chembl_id`s carry >1 distinct surface form** — the goldmine, e.g.
  - `CHEMBL1201583` (117 forms): Bevacizumab · **Avastin** · PF-06439535 · IBI310 · "Bevacizumab Injection"
  - `CHEMBL428647` (74): Paclitaxel · **Abraxane** · nab-paclitaxel · CAPOX
  - `CHEMBL1098` (99): bupivacaine · Bupivacain · Bupivacaine HCl · **EXPAREL**
- This is the REAL name-resolution problem (brand/biosimilar-code/salt/dose/abbrev), immune to a
  "you fabricated the noise" critique. Ground truth = same `chembl_id` ⇒ same drug.

## 2. Key insight — brand↔generic is NOT fuzzy-string-solvable
`Avastin`↔`Bevacizumab` are unrelated strings → string similarity fails. Bridging them needs **entity
resolution: name → chembl_id → all surface forms.** That map is free in the DB (each intervention row
has name + chembl_id). So the discriminator is the *retriever tier*, not string fuzz.

## 3. Three resolver tiers (the experiment's main axis)
For a query drug NAME (condition_id held oracle), find the failure records:
1. **exact** — `lower(name)==lower(intervention_name)` → that id's failures. (Naive.)
2. **fuzzy-string** — rapidfuzz top-k over intervention_name → those ids' failures. (Catches Bupivacain→Bupivacaine, dose variants; **fails brand→generic**.)
3. **synonym (chembl_id)** — fuzzy-find the query id(s) → their chembl_id(s) → ALL ids of those chembl_ids → all their failures. (Catches Avastin→Bevacizumab via chembl expansion.)

## 4. Task construction (cross-form queries)
Reuse `tasks_pilot.jsonl`. For each task whose drug `chembl_id` has ≥2 surface forms, set the QUERY name
to a **different** form than the one the failure is recorded under (`query_id ≠ recorded_id`, same
chembl_id). Failure stays under `recorded_id`. condition_id stays oracle. Tasks whose drug has 1 form →
kept as **clean controls** (query == recorded). Augment each record: `query_name`, `query_id`,
`recorded_id`, `chembl_id`, `is_cross_form`.

## 5. Two failure modes to measure (vs oracle 0.94)
- **False-deprioritization (dangerous):** gold=ground but the resolver returns ∅ → model asserts "no prior
  failure" / defers → a real prior failure MISSED (would greenlight a doomed trial). The N6 mode, concretized.
  Measured on cross-form ground/flag/reject tasks per (resolver tier × model).
- **False-grounding:** resolver returns a wrong-drug match → model grounds on it (misattribution).

Separate **retriever recall** (did the right record surface? — a property of the tier, measured deterministically)
from **model behavior given imperfect retrieval** (does it hedge/defer appropriately, or overclaim?).

## 6. Conditions
- Models: Qwen-14B-bf16, Qwen-32B-AWQ, Sonnet (same panel as G1).
- Resolver tiers: exact / fuzzy-string / synonym. (3×3; may trim to the informative cells.)
- **One variable isolated:** fuzz the DRUG name only; condition_id oracle. (Condition fuzz = later.)

## 7. Hypotheses
- H1 (retriever-is-the-lever): exact/fuzzy-string → recall collapses on cross-form ground tasks → model
  false-deprioritizes; synonym tier recovers recall → accuracy returns. ⇒ **the bottleneck is the retriever
  (entity-resolution layer), not the model/training** (confirms the G1 deployment thesis empirically).
- H2 (model calibration): the danger is whether the model, given ∅ retrieval, *overclaims* "no failure"
  vs *hedges* ("not found — may exist under another name"). A model/scale effect here = the only place
  training could add value.

## 8. Build order (validate before HPC — "no predictable failures")
1. `build_fuzzy_tasks.py` — cross-form task set + clean controls.
2. `fuzzy_resolvers.py` — the 3 tiers + `search_failures_via(tier,...)`.
3. **Deterministic experiment-validity gate** (local, no model): per tier, retriever recall on cross-form
   ground tasks. MUST show exact/fuzzy≈0, synonym≈1 — else the experiment is null/broken. Plus the
   deterministic-policy action-accuracy under each tier (the ceiling per tier).
4. `run_agent_open_fuzzy.py` — runner whose drug tool = resolver(tier); condition oracle.
5. Sonnet reference (native) + HPC open models, reuse the Expanse infra.

## 9. Honest caveats (state in any writeup)
- Query forms are drawn from in-DB surface forms ⇒ tests the *in-vocabulary cross-form* case; true
  free-text (out-of-registry) is harder (synonym tier would also miss → defer is then correct).
- Condition resolution held oracle ⇒ isolates drug-name resolution; a full system fuzzes both.

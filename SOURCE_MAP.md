# Source Map

Last source audit: 2026-06-25.

This file is deliberately compact. It records the source landscape without
reproducing full source text.

## Reading Status Key

- `prior full-read`: the source was read in full during an earlier research pass.
- `target-verified`: targeted lines or official pages were rechecked.
- `scan target`: identified as relevant but not read deeply in the latest pass.

## Anthropic Public Direction

| source | status | deployment relevance |
| --- | --- | --- |
| Anthropic, `Claude for Life Sciences`, published 2025-10-20, current official page checked 2026-06-25 | target-verified | Product direction is Claude as a full life-sciences research partner with scientific connectors, Agent Skills, prompt support, and dedicated support. This motivates tool deployment, but does not by itself solve calibrated trust. |
| Local PDF extract: `tmp/pdfs/claude_for_life_sciences.txt` | target-verified | Same page was locally captured on 2025-12-16. Key local lines: connectors/tools at 52-80, Agent Skills at 84-109, use cases at 113-147, AI for Science support at 196-208. |
| Anthropic, `Paving the way for agents in biology`, published 2026-06-08, current official page checked 2026-06-25 | target-verified | Strong external validation for deterministic execution layers: biological agents fail on brittle retrieval, and a deterministic tool layer makes outputs accurate and reproducible. This is the closest public analogue for SFM tool deployment. |

## Local Project Anchors

| source | status | deployment relevance |
| --- | --- | --- |
| `<local-workspace>/LLM_SFM_interpretability/README.md` | target-verified | Direct parent project. It states that presentation alone is insufficient and motivates enforcement via tools, MCP constraints, or post-training. |
| `<local-workspace>/LLM_SFM_interpretability/HANDOFF.md` | target-verified | Current frontier says variant-effect cleared precondition gates but reliability interface still produced a null, so the next step is enforcement-layer routing. |
| `<local-workspace>/LLM_SFM_interpretability/NEXT_SUBSTRATE.md` | target-verified | Variant-effect / clinical VUS is the strongest next substrate for calibrated trust-routing; it has real trust-vs-verify stakes. |
| `<local-workspace>/LLM_SFM_interpretability/experiments/trust_cue_attribution/SCHEMA.md` | target-verified | Reusable action/reward interface: `trust_sfm`, `verify_assay`, `default_baseline`, `defer`, with hidden truth scored after the action. |
| `<local-workspace>/llm-sfm-safety/README.md` | target-verified | Safety-recognition sibling project. It emphasizes trigger surfaces, recognition boundaries, and tool-mode behavior rather than generic refusal rate. |
| `<local-workspace>/Bio_Grounding_Eval/README.md` | target-verified | Grounding layer. Rule: train the skill, retrieve changing knowledge, orchestrate heavy specialist compute. |
| `<local-workspace>/Bio_Grounding_Eval/results/calibration_routing.md` | target-verified | Continuous confidence thresholding is more stable than binary defer. Web-exposure is an a-priori routing prior. |
| `<local-workspace>/Bio_Grounding_Eval/docs/UQ_ROUTING_POC_DESIGN.md` | target-verified | Per-item override is hard; the orchestrator's value is often domain routing, faithful grounding, and safe deferral, not beating the specialist item-by-item. |
| `<local-workspace>/Negative_result_DB/src/nullatlas_mcp/README.md` | target-verified | NullAtlas is the model for inference-time evidence guardrails: service-prefixed MCP tools, hard flags only for measured negatives, advisory status for unsupported claims. |
| `<local-workspace>/Causal_Grounding_Eval/results/move1/MOVE1_SYNTHESIS.md` | target-verified | Causal deployment recipe: default to a cheap baseline where it dominates, call the specialist in its competitive regime, verify high-value uncertain cases. |
| `<local-workspace>/LLM_SFM_phase4_planning/PHASE4A_CORE_RESULT_2026-06-25.md` | target-verified | Most important latest anchor. Phase 4a C1-C4 is already run on an N=57 hard-complex substrate: deterministic gate beats free-form LLM at meaningful verification costs and is robust to inverted/misleading reliability cues by computing from raw calibrated risk. |
| `<local-workspace>/LLM_SFM_phase4_planning/POSITIONING_BRIEF_2026-06-25.md` | target-verified | Current novelty framing: calibrated-risk trust-routing threshold over a fallible non-LLM scientific specialist, with manipulation robustness as the safety objective. Immediate next verification gap is the antibody-antigen OOD boundary / C5. |
| `<local-private-companion>/HANDOFF.md` | target-verified | Private companion handoff confirms the H2 null -> enforcement-layer transition and the no-overclaiming culture. |
| `<local-workspace>/bio_sfm_designer/README.md` | target-verified | Implementation sibling: DBTL designer where Claude orchestrates ProteinMPNN/ESMFold/Boltz-2, but the external calibrated trust gate decides trust/verify/default/defer. |
| `<local-workspace>/bio_sfm_designer/HANDOFF.md` | target-verified | Latest implementation status: gate thesis demonstrated end-to-end on complex/binder regime; pAE_interaction is informative but miscalibrated, and conformal risk control certifies a held-out guarantee at alpha=0.3. |
| `<local-workspace>/bio-sfm-trust-core/README.md` | target-verified | Reusable pure-stdlib trust engine: action set, isotonic calibration, deterministic gates, RCPS conformal risk control, no GPU/network required. |
| `<local-workspace>/bio-sfm-trust-core/src/bio_sfm_trust/gate.py` | target-verified | Implements `confidence_to_risk`, raw and leave-one-out calibrated gates, shuffled/inverted controls, and the `verify iff risk > lambda` policy. |
| `<local-workspace>/bio-sfm-trust-core/src/bio_sfm_trust/conformal.py` | target-verified | Implements RCPS/Hoeffding thresholding so the trusted set can carry a controlled false-accept bound. |

## Next Lookup Boundary

No unresolved sibling repo remains from the first recovery scan. Next source
lookup should be narrower: C5 antibody-antigen target curation and
`bio_sfm_designer` manifest/records scripts.

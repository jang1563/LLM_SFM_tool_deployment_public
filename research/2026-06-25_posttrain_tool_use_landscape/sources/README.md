# Source Cards

This directory stores compact source cards for papers, official pages, and local
anchors used in the post-training/tool-use landscape research.

Do not paste full papers, PDFs, or long extracted text here. Each card should
capture only:

- source identity and URL/path,
- evidence type: `paper`, `official`, `local`, or `scan`,
- domain,
- verifier/reward signal,
- why it matters for the project,
- claim boundary.

## Files

| file | purpose |
| --- | --- |
| `math.md` | Formal mathematics, theorem proving, proof checking, proof-state RL. |
| `physics.md` | Physics/engineering simulations, PDE/FEA solvers, physical reasoning benchmarks. |
| `chemistry.md` | Chemistry agents, molecular/reaction tools, spectra, lab/API execution. |
| `biology.md` | Biology/biomedical benchmarks, protocols, database/evidence grounding, agentic biology. |
| `sfm_trust_calibration.md` | Specialist foundation-model confidence, calibration, OOD, and baseline-routing sources. |
| `trajectory_post_training.md` | SFT, RLHF, DPO, process supervision, and tool-use RL method sources. |
| `tool_use_rl.md` | General tool-use/post-training/RL/SFT/function-calling sources. |
| `research_agent_eval.md` | End-to-end research-agent, reproducibility, simulated discovery, and benchmark-auditing sources. |
| `official_product.md` | Company product and research pages relevant to scientific tool systems. |

## Tutor-Mode Bridges

| file | purpose |
| --- | --- |
| `../METHOD_LADDER_HANDOUT.md` | Maps SFT/RLHF/DPO/process supervision/RLVR/runtime enforcement to biology trajectory slices. |
| `../FIRST_EXPERIMENT_DESIGN.md` | Turns the literature map into Stage A retrieval/evidence dry run and Stage B C5 Ab-Ag OOD trust-gate design. |
| `../STAGE_A_C5_RESEARCH_BRIDGE_2026-07-04.md` | Active research bridge from source map to concrete Stage A mini-manifest and C5 trust-gate tickets. |
| `../STAGE_A_SCHEMA_AUDIT_2026-07-04.md` | Checks that the current evaluator schema is sufficient for the first Stage A dry run. |
| `../STAGE_A_IMPLEMENTATION_CHECKPOINT_2026-07-04.md` | Records the first code-backed Stage A manifest, no-API baselines, and validator result. |

## Update Rule

Add a source only when it clarifies at least one of:

1. what can be automatically scored,
2. what needs expert preference/judgment,
3. what should be a runtime policy or hard gate,
4. how the domain differs from math/physics/chemistry/biology neighbors.

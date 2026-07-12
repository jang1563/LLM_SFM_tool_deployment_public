# Synthesis

## Working Claim

For scientific LLM x SFM deployment, the important unit of post-training is not
only final-answer quality. It is the trajectory:

```text
intent -> tool/database choice -> valid call -> evidence packet -> trust/verify/default/defer action -> scored outcome
```

## Current Interpretation

The likely strongest project framing is:

> build trainable and enforceable biology tool-use trajectories, then test which
> parts should be learned by SFT/preference/RLVR and which should be made binding
> by deterministic gates, validators, and calibrated risk policies.

## Cross-Domain Difference

Do not treat "science" as one domain. The verifier changes by field:

| domain | verifier center | post-training implication |
| --- | --- | --- |
| Math | Formal proof checker and exact symbolic states. | RLVR/RL can be unusually clean because proof states can be checked, but autoformalization and search remain bottlenecks. |
| Physics | Simulators, units, conservation laws, numerical solvers, predictive holdouts. | Rewards are decomposable but assumption-dependent; agents need to state boundary conditions and run/check simulations. |
| Chemistry | Structure/reaction validators, spectra, property models, lab logs, robotic APIs. | Tool trajectories matter because agents must translate among molecules, reactions, spectra, papers, and executable protocols. |
| Biology | Source existence, metadata, QC, hidden labels, assays, SFM calibration, evidence status. | RLVR is narrow; calibrated trust-routing and runtime policy are central because context shift and negative evidence are hard. |

See `DOMAIN_COMPARISON.md` for the current detailed map and
`BENCHMARK_VERIFIER_MAP.md` for the source-by-verifier evidence map.

## 2026-06-25 Landscape Read

The field has converged on tool use as an agent substrate, but the decisive
distinction for this project is **learned trajectory behavior vs enforced runtime
policy**.

### What The External Landscape Supports

- Anthropic's life-sciences direction is connector/skill/MCP heavy, not just
  model-knowledge heavy.
- Anthropic's agents-in-biology note directly supports deterministic biological
  retrieval as reliability infrastructure.
- Anthropic's BioMysteryBench supports verifiable, messy, real-world
  bioinformatics tasks with containerized tool use, but it mostly grades final
  answers; this project should add trajectory/action scoring.
- Anthropic's long-running scientific-computing note supports the practical
  need for test oracles, persistent memory, and progress files.
- Anthropic's chemistry work strengthens the representation-translation axis:
  scientific agents must move among structures, spectra, database queries,
  papers, and instrument outputs.
- OpenAI's deep research and agent tooling show that browser/Python/tool use is
  now a post-training and product-infrastructure frontier.
- Google/FutureHouse show science-agent systems are becoming multi-agent and
  tool-rich, but they are less specifically about calibrated trust in fallible
  specialist SFMs.
- The paper literature has moved from prompting and SFT datasets toward RL/GRPO,
  RLVR, and long-horizon agent-environment training.
- ToolUniverse and TxAgent show a concrete path for scientific tool ecosystems:
  standardize tool specs, search/call tools, generate multi-step traces, and
  train domain agents on function calls and evidence-grounded reasoning.

### What The Local Program Adds

The local projects add a sharper deployment thesis than generic "agents use
tools":

1. A biological specialist can be fallible, miscalibrated, or regime-limited.
2. The LLM should not be the final authority over whether to trust it.
3. Tool outputs must become evidence packets with calibration and failure-mode
   metadata.
4. Runtime policy should bind actions like `verify`, `baseline`, and `defer`.
5. Post-training should optimize trajectories around those actions, not only
   final prose.

### Caution On RLVR

RLVR is a strong fit only for the verifiable slice of the environment. For
biology tool use, that means:

- exact database retrieval,
- valid schema/tool arguments,
- citation/source existence,
- hidden-label correctness in curated benchmarks,
- cost-aware routing reward,
- deterministic validator/gate compliance.

It is not automatically the right answer for scientific interpretation, causal
claim quality, or SFM confidence calibration. Those need either expert
preference labels, calibrated statistical tools, or hard deployment gates.

## Recommended First Experiment

Start with a small open-model tool-use environment whose rewards are mostly
verifiable, then add a calibrated-trust action:

```text
task -> choose evidence tool/database/SFM -> call with valid schema
     -> receive evidence packet -> choose trust/verify/baseline/defer
     -> score hidden correctness + cost + policy compliance
```

Best first substrate candidates:

| substrate | why |
| --- | --- |
| NullAtlas-style claim verification | Deterministic evidence-status labels, source existence, request-more-evidence action. |
| Structure-confidence gate compliance | Direct bridge to Phase 4a/C5, known calibrated-risk action set. |
| Database retrieval/completeness | Easiest RLVR rewards; closest to Anthropic agents-in-biology deterministic retrieval lesson. |
| Perturbation/SFM routing | Most program-central, but riskier because rewards and calibration are more subtle. |

Recommendation: begin with **database/NullAtlas-style verifiable retrieval plus
the same trust/verify/defer action schema**, then port the harness to C5
structure-confidence routing after the evaluator is stable.

The benchmark/verifier map strengthens this recommendation: the right first
biology environment should use math-like verifier slices where possible
(`valid tool call`, `source exists`, `hidden label`, `gate compliance`) while
making biological uncertainty explicit through `verify`, `baseline`, and
`defer` actions.

Add a verifier-audit pass before training: recent benchmark-auditing work shows
execution-based agent benchmarks can fail through cross-artifact mismatches
between instructions, reference solutions, evaluation scripts, and environments.
For this project, the evaluator itself must be treated as a scientific object:
audited, versioned, and stress-tested before it becomes an RLVR reward source.

The newest domain-verifier pass adds one more design constraint: record the
agent's provenance graph. Scientific reward should not only ask whether the
final answer is correct, but whether each claim is linked to a source, tool
output, benchmark artifact, simulation assumption, database check, or explicit
policy gate. This is especially important in biology, where GenoTEX-style
expert-alignment, GeneAgent-style self-verification, and single-cell workflow
metrics often substitute for a clean ground-truth oracle.

The C5/SFM calibration pass makes the first experiment more concrete. Treat each
foundation model as a specialist witness that emits typed evidence: metric
scope, interaction regime, confidence, affinity, baseline comparison, and
calibration status. A trust action is only valid if the metric is calibrated for
that regime; otherwise the correct deployment action is `verify`, `baseline`, or
`defer`. This applies both to structure models, where antibody-antigen interfaces
may break a general confidence gate, and transcriptomics perturbation models,
where simple baselines can outperform more elaborate foundation-model routes.

The trajectory post-training pass clarifies the method ladder. Start with SFT on
expert/reference scientific trajectories; use DPO or RLHF/RLAIF for paired
better/worse traces where expert judgment is needed; use process supervision for
intermediate actions such as tool choice, query construction, evidence handling,
and gate compliance; reserve RLVR/tool-use RL for slices where the environment
can score outcomes or intermediate actions. Agentic RL frameworks such as
Agent Lightning and VerlTool are infrastructure, not evidence that the biology
reward is correct.

The method ladder handout turns that into an implementation rule: the biology
trajectory is split into layers, and each layer gets the weakest sufficient
training/enforcement method. Schema validity and metric extraction can be RLVR;
evidence synthesis and biological interpretation usually need process or
preference supervision; uncalibrated SFM trust must remain a runtime gate.

## Tutor-Mode Prep

When enough sources are gathered, explain this to JK as a layered stack:

1. Product signal: Claude LS, skills, connectors, MCP, ToolUniverse.
2. Research signal: ReAct/Toolformer/API-Bank/ToolLLM to ToolRL/RLVR/Agent
   Lightning.
3. Biology signal: BioMysteryBench, agents in biology, TxAgent,
   BioDiscoveryAgent.
4. Domain distinction: math is proof-checking, physics is simulation/assumption
   checking, chemistry is representation/execution, biology is
   context/evidence/calibration.
5. Local contribution: calibrated SFM trust-routing plus deterministic
   verification/default/defer gates.
6. Concrete next experiment: verifiable retrieval/tool-use environment first,
   then C5 antibody-antigen trust-gate OOD.

Use tutor mode with examples and checks for understanding, not a slide-deck
style monologue.

## Open Questions

- Which early substrate should be used for a small open-model training loop:
  NullAtlas claim verification, database retrieval, structure-confidence gate
  compliance, or variant/perturbation evidence routing?
- Which rewards are truly verifiable, and which need expert preference labels?
- Where does RLVR help, and where do local negative results imply a hard external
  tool/gate is the better intervention?

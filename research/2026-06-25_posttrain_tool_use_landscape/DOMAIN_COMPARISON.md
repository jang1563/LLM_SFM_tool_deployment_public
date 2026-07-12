# Domain Comparison: Math / Physics / Chemistry / Biology

Purpose: separate scientific-agent design by domain instead of treating "science"
as one homogeneous tool-use problem.

## One-Line Thesis

The domains differ mainly by what counts as a reliable verifier:

```text
math: proof checker
physics: simulator + units + predictive/experimental fit
chemistry: molecular/reaction validators + spectra + lab execution
biology: database evidence + noisy assays + context/calibration gates
```

That verifier difference determines what can be learned with RLVR, what needs
SFT/preference learning, and what should be hard runtime policy.

## Comparison Table

| domain | object of reasoning | tool substrate | strongest verifiers | where reward gets messy | best post-training target | hard deployment gate |
| --- | --- | --- | --- | --- | --- | --- |
| Math | Theorems, proofs, symbolic objects. | Lean/Coq/Isabelle, CAS, symbolic search, proof libraries. | Formal proof checker; exact symbolic equivalence; generated problem variants. | Natural-language to formal-statement translation; search cost; whether the formalized theorem captures the intended math. | RL/search over proof states, synthetic curricula, test-time adaptation, proof-trajectory imitation. | Reject unverified proof claims; expose formal statement and checker trace. |
| Physics | Laws, fields, particles, dynamics, physical systems. | Numerical simulators, PDE solvers, FEA, units/dimensional-analysis tools, symbolic regression, data fitting. | Conservation laws, dimensional consistency, PDE residuals, solver convergence, held-out simulation/experiment prediction. | Boundary conditions, model mismatch, numerical instability, extrapolation beyond simulated worlds. | Tool-use RL for experiment/simulation design; SFT on research-code workflows; rewards decomposed into units, solver validity, and predictive holdout. | Require explicit assumptions, units, boundary conditions, and simulator/solver checks. |
| Chemistry | Molecules, reactions, materials, spectra, lab procedures. | RDKit/SMILES/InChI, retrosynthesis tools, docking/property predictors, quantum chemistry, spectra parsers, ELNs, lab robotics APIs. | Molecular validity, reaction rule checks, spectral match, instrument/log execution, assay result when available. | Predicted property versus wet-lab reality; synthesis feasibility; safety and procedural ambiguity. | SFT on chemistry tool trajectories; validator-guided tool calls; closed-loop optimization when assay/log rewards exist. | Require structure/reaction validation, safety filters, source-linked protocols, and lab-execution constraints. |
| Biology | Genes, cells, perturbations, organisms, phenotypes, clinical or ecological contexts. | NCBI/Ensembl/PubMed/ClinicalTrials/Open Targets/ChEMBL, omics pipelines, SFMs, perturbation models, statistical QC. | Accession/source existence, schema validity, benchmark hidden labels, QC metrics, known metadata, deterministic retrieval. | Context dependence, confounding, negative evidence, dataset leakage, SFM calibration, regime shift, causal interpretation. | SFT/preference learning for evidence workflows; RLVR only for retrieval/schema/hidden-label slices; calibrated trust-routing around SFMs. | Bind `verify`, `baseline`, `defer`, evidence-status, and calibrated-risk policies outside the model. |

## Why Math Is The Cleanest RLVR Case

Math has a rare property: a proof can often be checked mechanically. AlphaProof
and AlphaGeometry show why this matters. The agent can search in a formal
environment, receive a binary or near-binary signal from a proof checker, and
train on large synthetic curricula. The key bottleneck shifts from "is the answer
true?" to "did we formalize the right problem, and can search find the proof?"

Implication for this project: math is the upper bound for clean verifier-driven
post-training. Biology should borrow the trajectory discipline, but not pretend
it has the same oracle.

## Why Physics Is Between Math And Empirical Science

Physics has strong mathematical structure, but the verifier is usually a
simulator, solver, or experiment rather than a proof checker. That makes rewards
less binary. A physics agent may be right under one boundary condition and wrong
under another; it may fit a simulated world without discovering the latent law.

Good physics-agent rewards are decomposed:

- unit/dimensional consistency,
- explicit assumptions and boundary conditions,
- solver convergence and numerical stability,
- conservation-law checks,
- predictive holdout under new initial conditions,
- explanation against a hidden simulator or expert description.

Implication for this project: physics is a good analogy for tool trajectory and
experiment design, but still cleaner than biology because many tasks can be
placed inside controlled simulators.

## Why Chemistry Is Representation And Execution Heavy

Chemistry agents are forced to translate among names, structures, SMILES/InChI,
spectra, reactions, properties, papers, patents, and lab protocols. This is why
ChemCrow, Coscientist, and Anthropic's chemistry work all emphasize tools and
representation translation. Chemistry has many local validators, but final truth
often still depends on feasibility, safety, and assay or instrument output.

Implication for this project: chemistry is the closest neighboring domain for
"tool wrapper + validator + lab/API execution" design. It is a better analogy
than pure math for deployment, but the proxy-to-reality gap remains smaller and
more instrumentable than many biology tasks.

## Why Biology Needs Calibrated Trust Routing

Biology is the hardest case for naive RLVR because much of the difficulty is not
syntax, retrieval, or single-step prediction. It is context:

- the same perturbation can behave differently by cell type, assay, dose, time,
  batch, and disease state;
- absence of evidence and true negative evidence are easy to confuse;
- SFMs can be useful but miscalibrated outside their training regime;
- causal interpretation often requires external design/statistics, not just
  final-answer correctness.

So biology rewards should be split into two layers:

1. Verifiable layer: valid database/tool call, source existence, schema validity,
   hidden benchmark label, QC threshold, citation/accession retrieval.
2. Judgment/policy layer: trust the SFM, verify with another assay/model,
   default to baseline, defer, or request more evidence.

The second layer should not be left to model vibes. It needs calibrated gates,
deterministic validators, and explicit evidence-status policy.

## Training Implications

| method | math | physics | chemistry | biology |
| --- | --- | --- | --- | --- |
| SFT | Proof-style trajectories, formalization examples. | Research-code, simulation, derivation workflows. | Tool-use traces over structures/reactions/spectra. | Evidence workflows, omics pipelines, SFM routing cases. |
| Preference / DPO | Cleaner vs messier proof strategies. | Better assumptions, experiment choices, interpretation. | Safer, more feasible, better-sourced chemistry plans. | Better evidence use, uncertainty handling, conservative claims. |
| RLVR | Strong: checker and formal state rewards. | Medium: simulator/solver/holdout rewards. | Medium: structure/reaction/log/assay rewards. | Narrow: retrieval, schema, QC, hidden labels, gate compliance. |
| Runtime enforcement | Formal proof required. | Units, boundary conditions, solver checks. | Structure, safety, and lab protocol constraints. | Evidence-status, calibrated trust, verify/default/defer gates. |

## Tutor-Mode Framing

Use this ladder:

1. Math asks: "Can you prove it?"
2. Physics asks: "Does the law/simulation predict the system under stated
   assumptions?"
3. Chemistry asks: "Is the molecule/reaction/procedure valid and executable?"
4. Biology asks: "Under this biological context, should we trust, verify,
   baseline, or defer?"

That is why the local project belongs in biology deployment: it studies the
hardest part of scientific tool use, where the model must route among fallible
specialists and evidence sources under context shift.

## Sources Added In This Pass

- Google DeepMind AlphaProof / AlphaGeometry: formal math reasoning and proof
  verification.
- Nature AlphaProof: RL in Lean formal environments, test-time RL, verifier
  grounding.
- SciCode: cross-domain scientific coding benchmark across math, physics,
  chemistry, biology, and materials science.
- DiscoverPhysics: interactive physics-law discovery with black-box N-body
  simulators.
- PDEBench: PDE simulation benchmark against numerical methods and ML baselines.
- LLM-SRBench: scientific equation discovery benchmark designed to reduce
  memorization artifacts.
- MCP-SIM: language-to-physics-simulation multi-agent framework.
- ChemCrow and Coscientist: chemistry agents with expert tools, documentation,
  code execution, and lab APIs.

## Sources Added In Follow-Up Collection

- LeanDojo, miniF2F, ProofNet, and FunSearch sharpen the math side: formal proof
  environments, benchmark portability, autoformalization, and executable
  evaluators.
- FEABench, Mind's Eye, and PHYBench add physics-specific grounding:
  FEA/COMSOL interaction, MuJoCo simulation input, original physics problems,
  and expression-level scoring.
- ChemBench, ChemToolAgent, and ChemAgent add chemistry-specific calibration:
  chemistry Q&A against experts, the task-dependent value of tools, and
  fine-tuning for tool selection and parameter filling.
- LAB-Bench, LABBench2, BioPlanner, BioKGBench, BioProBench, and ABC-Bench add
  biology-specific realism: literature/database access, protocols, KG checking,
  procedural logic, sequence/cloning tasks, lab automation, and governance-aware
  agentic capability evaluation.

## Sources Added In Goal-Mode Domain-Verifier Pass

- PutnamBench and FrontierMath sharpen the internal split inside math: formal
  theorem proving gives machine-checkable rewards, while advanced hidden-problem
  benchmarks can use automated final-answer verification without exposing a
  proof-checker trajectory.
- PhysGym and PDE-Grounded Intent Verification sharpen physics: an agent must
  infer laws under controlled priors, and runnable simulation code is not enough
  if it violates the intended physics specification.
- ChemSafetyBench and LabSafety Bench add the deployment layer for chemistry and
  lab-facing science: safety, legality, hazard recognition, and consequence
  prediction are verifier dimensions.
- GeneAgent, GenoTEX, CellAgent, and the single-cell omics agent benchmark add
  computational-biology specificity: database self-verification, expert-standard
  analysis artifacts, single-cell workflow planning, and multidimensional process
  metrics.
- SciCode, SciAgentGym, and the evidence-provenance survey provide the generic
  tool-use bridge: scientific code/tasks, typed tools, stateful interaction,
  structured traces, and evidence/action lineage.

See `BENCHMARK_VERIFIER_MAP.md` for a source-by-verifier table.

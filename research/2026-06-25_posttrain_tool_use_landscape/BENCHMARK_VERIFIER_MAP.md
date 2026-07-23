# Benchmark / Verifier Map

Purpose: collect research sources by what they can verify. This keeps the
post-training discussion grounded: reward design follows the verifier, not the
domain label.

## Verifier Ladder

| level | verifier type | clean reward? | examples |
| --- | --- | --- | --- |
| L0 | syntax/schema validity | yes, but shallow | valid function call, valid SMILES, valid FASTA, valid JSON. |
| L1 | deterministic tool result | mostly yes | proof checker pass, database accession exists, simulator runs, RDKit parses. |
| L2 | hidden label / benchmark oracle | yes within benchmark | exact answer, held-out simulation, curated metadata, known protocol step. |
| L3 | expert/scientific judgment | partly | better assumptions, better evidence use, better chemistry route, safer protocol. |
| L4 | real-world experiment | slow/noisy | wet-lab assay, clinical outcome, physical experiment, instrument output. |
| L5 | deployment policy | yes if encoded | must verify, must defer, must cite source, must not trust SFM outside regime. |

Working rule: RLVR is strongest at L0-L2 and L5. L3 needs preference/expert
labels. L4 needs closed-loop experimental infrastructure and should be treated
as expensive evidence, not a cheap reward.

## Training Method Sources By Feedback Signal

| method family | source | feedback / reward signal | why it matters here |
| --- | --- | --- | --- |
| SFT + RLHF | InstructGPT | Demonstrations, ranked outputs, reward model, RLHF. | Canonical post-training ladder: imitate first, then optimize preference. |
| RLAIF / policy training | Constitutional AI | Principle-guided critique/revision plus AI preference feedback. | Useful for scalable `verify/defer/unsupported` policy examples, but not a scientific verifier. |
| Preference optimization | DPO | Paired preference examples with direct policy optimization. | Natural fit for better-vs-worse scientific trajectories where exact reward is unavailable. |
| Process supervision | Let's Verify Step by Step | Step-level feedback over intermediate reasoning. | Supports scoring tool choice/evidence integration before final answer. |
| Interactive agent training | AgentGym / WebRL | Environment interaction, trajectory datasets, online curricula, outcome rewards. | Helps design scientific curricula where failed attempts become new tasks. |
| Tool-use RL | ReTool / Search-R1 / ToolRL | Code/search/tool rollouts, decomposed rewards, outcome rewards. | Direct evidence that models can learn when/how to invoke tools under RL, given rewardable tasks. |
| Agentic RL infrastructure | Agent Lightning / VerlTool | Decoupled agent execution, MDP transitions, standardized tool APIs. | Systems path for turning existing biology agents into trainable trajectories. |
| Trajectory verifier design | The Art of Building Verifiers for Computer Use Agents | Separate process/outcome criteria, controllable/uncontrollable failure attribution, and low false-positive evaluation. | Direct design guidance for evidence packets, tool-sequence checks, and distinguishing agent failure from unavailable or failed tools. |
| Trajectory reward-model audit | Plan-RewardBench | Pairwise preference over valid and confusable tool trajectories. | Shows LLM judges and reward models degrade on difficult long-horizon traces; deterministic Stage A gates should remain authoritative where available. |

## Domain Sources By Verifier

| domain | source | verifier signal | why it matters here |
| --- | --- | --- | --- |
| Math | AlphaProof / AlphaGeometry | Formal proof checker, formalized problem statements, theorem-proving search. | Clean upper-bound case for verifier-driven RL; biology cannot assume this oracle. |
| Math | LeanDojo | Programmatic Lean interaction, proof states, tactics, premises, benchmark splits. | Shows how a tool environment exposes trajectory states for training/evaluation. |
| Math | miniF2F | Cross-system formal Olympiad statements in Lean/Metamath/Isabelle/HOL Light. | Good example of benchmark portability across formal systems. |
| Math | ProofNet | Paired natural-language theorem, Lean statement, and natural-language proof. | Autoformalization is the messy bridge from prose to a clean verifier. |
| Math | PutnamBench | Hand-built formalizations across Lean 4, Isabelle, and Coq. | Separates formal theorem-proving evaluation from ordinary answer-only math benchmarks. |
| Math | FrontierMath | Expert-authored unpublished advanced math problems with automated verification where possible. | Useful reminder that even math is not always proof-trajectory RLVR. |
| Math | FunSearch | Executable program evaluator guarding against LLM confabulation. | Adjacent to science discovery: frozen LLM proposes, evaluator selects. |
| Physics | FEABench | COMSOL/FEA execution, executable API calls, numerical solver outputs. | Strong physics analogue for tool-use agents operating real scientific software. |
| Physics | Mind's Eye | MuJoCo simulation result as grounding input for physical reasoning. | Early direct evidence that physics simulators can act as external reasoning substrate. |
| Physics | PDEBench | Classical numerical simulations and ML baselines over PDE tasks. | Useful for solver/holdout/residual-style rewards rather than proof rewards. |
| Physics | DiscoverPhysics | Black-box N-body simulator, predictive accuracy, explanation score. | Tests experiment design and latent-law discovery, closer to science process. |
| Physics | PhysGym | Interactive simulation, controlled priors, hypothesis fidelity and data fit. | Makes context/prior masking explicit, a useful bridge to biology context shift. |
| Physics | PDE-Grounded Intent Verification | Runnable simulation code checked against intended physics specification. | Shows "simulation runs" is too weak unless intent fidelity is verified. |
| Physics | PHYBench | Original physics problems plus expression edit distance scoring. | Shows equation/process scoring can be richer than binary answer matching. |
| Physics | CritPt | Unpublished research-level physics problems and reasoning-trace trust questions. | Separates frontier research reasoning from textbook/exam physics. |
| Physics | Physics-IQ Verified | Controlled real-world physical videos plus audited prompt/metric quality. | Shows verifier design can itself distort capability rankings. |
| Physics | QuantiPhy | Numerical ground truth for kinematic properties in videos. | Clear example where verbal plausibility diverges from quantitative correctness. |
| Chemistry | ChemBench | Curated chemistry Q&A against chemist expertise, overconfidence analysis. | Knowledge/reasoning benchmark; useful warning that confident chemistry answers need calibration. |
| Chemistry | ChemSafetyBench | Chemical property, legality, and synthesis-method safety/appropriateness checks. | Adds policy/safety as a chemistry verifier dimension beyond molecular validity. |
| Chemistry | LabSafety Bench | OSHA-aligned lab safety questions and scenario hazard/consequence checks. | Cross-domain safety gate for lab-facing scientific agents. |
| Chemistry | ChemCrow | 18 expert-designed chemistry tools for synthesis, drug discovery, materials. | Canonical chemistry tool-agent source; representation and tool choice matter. |
| Chemistry | ChemToolAgent | 29 tools; tool augmentation helps specialized tasks but not all general questions. | Important caution: tools add cognitive load and can hurt if used indiscriminately. |
| Chemistry | ChemAgent / ChemToolBench | 137 tools, tool selection, parameter filling, step-level fine-tuning. | Strong source for chemistry SFT/process-supervision over tool traces. |
| Chemistry | Coscientist | Web/documentation search, code execution, robotic lab APIs, lab task execution. | Chemistry bridge from text/tooling to physical execution and lab logs. |
| Chemistry | RxnBench | Reaction-scheme QA and full-document QA over chemistry PDFs. | Multimodal chemistry benchmark where structure/mechanism reasoning is harder than text extraction. |
| Biology | LAB-Bench | Literature, figures/tables, database navigation, DNA/protein sequence tasks. | Practical biology-research benchmark beyond textbook QA. |
| Biology | LABBench2 | Open-response retrieval, patents, clinical trials, source quality, sequence/database tasks. | Moves biology eval closer to realistic research workflows and source-quality judgment. |
| Biology | BioPlanner | Natural-language protocols mapped to pseudocode functions; lab validation. | Biology protocol planning needs structured intermediate representation. |
| Biology | BioKGBench | Claim verification plus KGQA/RAG to identify knowledge-graph factual errors. | Very close to NullAtlas/evidence-status framing. |
| Biology | GeneAgent | Generated gene-set claims self-verified against biological databases. | Direct generation-then-verification pattern for biological narratives. |
| Biology | GenoTEX | Expert-curated gene expression analysis pipelines with code/intermediate/final artifacts. | When ground truth is absent, expert-standard process alignment becomes the target. |
| Biology | CellAgent | Planner/executor/evaluator roles for single-cell analysis workflows. | Shows role decomposition and self-iteration in realistic scRNA-seq analysis. |
| Biology | BioProBench | 27k protocols, 550k structured instances, protocol QA/ordering/error correction/reasoning. | Focuses biology on procedural logic, quantitative steps, and safety-critical reasoning. |
| Biology | ABC-Bench | Bioinformatics/lab automation tasks with algorithmic criteria and wet-lab validation. | Strong agentic-biology benchmark; also reminds us governance constraints matter. |
| Biology | BixBench | Realistic bioinformatics scenarios, Jupyter/code execution, open-answer expert evaluation. | Close to biology-agent workflow evaluation; final answers remain less clean than exact verifiers. |
| Biology | BiomniBench | Full analytical trajectory scored against expert task rubrics. | Directly supports process-level evaluation rather than final-answer-only scoring. |
| Biology | BioAgent Bench | End-to-end bioinformatics tasks with artifact checks and perturbation stress tests. | Adds robustness against corrupted inputs, decoys, and prompt bloat as verifier targets. |
| Biology | Single-cell omics agent benchmark | 50 real-world single-cell omics tasks with multidimensional process metrics. | Directly relevant to computational biology agents and SFM-adjacent workflows. |
| Biology | PromptBio-Bench | Structured output-file comparison against expert reference files. | Adds artifact/provenance-style evaluation for bioinformatics agents. |
| Biology | MedAgentGym | Executable biomedical sandboxes, verifiable ground truth, interactive feedback, trajectory generation. | Strong example of training environment rather than static benchmark. |
| Biology | PertEval-scFM / Tx foundation model perturbation benchmarks | Distribution-shift perturbation prediction and comparison to simple baselines. | Direct support for baseline/default actions in SFM trust routing. |
| SFM trust | AlphaFold confidence metrics | pLDDT, PAE, pTM, ipTM, and interface/global metric scope. | Specialist confidence must be typed by metric and regime before a trust action. |
| SFM trust | AlphaFold antibody-antigen evaluations | CAPRI/DockQ-style Ab-Ag accuracy and interface confidence evaluation. | Direct source for C5: Ab-Ag is a regime-specific calibration boundary. |
| SFM trust | Boltz-2 | Structure plus affinity prediction with model-specific confidence/affinity outputs. | Practical specialist model source for DBTL and structure-design evidence packets. |
| SFM trust | Boltz-2 reliability evaluation | Compressed confidence scores and target-dependent reliability behavior. | Supports fail-closed behavior when internal confidence lacks discrimination. |
| SFM trust | Perturbation FM baseline studies / scPerturBench | Simple baselines, distribution-shift splits, unseen contexts and perturbations. | Supports `baseline` as an explicit action rather than an afterthought. |
| Calibration | RCPS / conformal risk control | Finite-sample black-box risk control and distribution-shift extensions. | Method backbone for certifying `trust_specialist_output` thresholds. |
| General science | ScienceWorld | Interactive text environment for scientific experiments. | Early grounding source: agents must act and observe, not just answer. |
| General science | DiscoveryWorld | Virtual discovery cycles with task/action/explanatory-knowledge metrics. | Useful for hypothesis-experiment-analysis loop design. |
| General science | ScienceAgentBench | Scientific Python program generation from peer-reviewed workflow tasks. | Good task-level scientific coding/evaluation source before end-to-end claims. |
| General science | SciCode | Scientist-curated research coding tasks with tests across multiple science domains. | Cross-domain bridge from knowledge QA to executable research scripts. |
| General science | SciAgentGym / SciAgentBench | Typed scientific tools, stateful interaction, structured traces, long-horizon tasks. | Closest generic analogue to training scientific tool-use trajectories. |
| General science | PaperBench | Hierarchical rubrics for replicating AI research papers. | Strong source for decomposed rubric grading of research-agent work. |
| General science | CORE-Bench | Reproduction of paper results from code/data. | Reproducibility is a necessary deployment gate before discovery claims. |
| General science | ResearchClawBench | Hidden target papers, raw data, executable environments, expert rubrics. | Shows end-to-end science agents miss protocols/evidence chains despite complete reports. |
| Benchmark reliability | BenchGuard | Cross-artifact audit of instruction, environment, reference solution, and evaluation logic. | Verifier quality must itself be audited before rewards/evals are trusted. |
| Agent provenance | Evidence tracing / execution provenance survey | Links evidence, tools, memory, observations, claims, actions, and final answers. | Provides vocabulary for process-level accountability and trace schemas. |

## Cross-Domain Pattern

| question | math | physics | chemistry | biology |
| --- | --- | --- | --- | --- |
| What does the agent call? | theorem prover, proof library, CAS. | simulator, PDE/FEA solver, units tool. | molecule/reaction/spectrum/property/lab tools. | databases, omics pipelines, SFMs, protocol/statistics tools. |
| What is a cheap reward? | proof accepted. | simulation runs, units pass, held-out prediction. | valid molecule/reaction, property/tool result, lab API simulation. | valid accession/schema/QC/hidden label, gate compliance. |
| What is hard to reward? | intended theorem formalization. | assumptions and model mismatch. | feasibility, safety, wet-lab reality. | causal interpretation, negative evidence, context shift, SFM calibration. |
| What should be enforced? | no unverified proof claim. | explicit assumptions/units/boundary conditions. | structure/safety/protocol constraints. | evidence status, calibrated trust, verify/default/defer. |

## First-Experiment Implication

The best first experiment should not start at "biology discovery." It should
start where biology has math-like verifier slices:

1. database retrieval with exact source/accession checks,
2. schema-valid scientific tool calls,
3. hidden-label evidence-status benchmark,
4. enforced `trust | verify | baseline | defer` action policy,
5. cost/reward accounting for unnecessary tool calls.

This gives enough clean reward for SFT/RLVR experiments while preserving the
biology-specific contribution: calibrated trust routing under evidence limits.

## Open Collection Queue

Next sources worth adding if the map needs more density:

- math: current Lean 4 theorem-prover leaderboards, autoformalization failure
  analyses, proof-state RL papers.
- physics: real scientific discovery benchmarks with simulator access and
  physics benchmark audit/failure-mode papers.
- chemistry: lab automation logs, molecular representation consistency papers,
  and reaction/procedure safety evaluators.
- biology: deeper BiomniBench read, clinical-trial/patent retrieval tasks,
  additional SFM calibration/OOD benchmarks, and local Phase 4/C5 alignment.

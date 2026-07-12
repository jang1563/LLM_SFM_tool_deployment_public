# Biology Source Cards

## Anthropic Agents In Biology / gget Virus

- Type: `official`
- URL: https://www.anthropic.com/research/agents-in-biology
- Domain: biology
- Verifier/reward signal: deterministic retrieval layer over biological data.
- Project relevance: direct external support for tool/database infrastructure as
  reliability layer in biology.
- Claim boundary: retrieval success is a verifier slice, not full biological
  reasoning.

## BioMysteryBench

- Type: `official`
- URL: https://www.anthropic.com/research/Evaluating-Claude-For-Bioinformatics-With-BioMysteryBench
- Domain: bioinformatics
- Verifier/reward signal: expert-written bioinformatics questions with objective
  answers from controlled data properties or validated metadata.
- Project relevance: good model for messy real-world bioinformatics tasks with
  containerized tools.
- Claim boundary: mostly grades final answer; this project should add
  trajectory/action scoring.

## LAB-Bench

- Type: `paper`
- URL: https://arxiv.org/abs/2407.10362
- Domain: biology
- Verifier/reward signal: practical biology research tasks: literature, figures,
  databases, DNA/protein sequence manipulation.
- Project relevance: broad biology-research capability map beyond textbook QA.
- Claim boundary: many tasks are multiple-choice and do not fully capture
  agent trajectories.

## LABBench2

- Type: `paper`
- URL: https://arxiv.org/html/2604.09554v2
- Domain: biology
- Verifier/reward signal: open-response retrieval, source quality, patents,
  clinical trials, databases, protocols, sequences, cloning.
- Project relevance: newer, more realistic benchmark direction; close to the
  evidence/source-quality axis of this project.
- Claim boundary: still benchmarked tasks, not deployment policy by itself.

## BioPlanner

- Type: `paper`
- URL: https://arxiv.org/abs/2310.10632
- Domain: biology / protocols
- Verifier/reward signal: protocol-to-pseudocode intermediate representation,
  admissible functions, lab validation of generated protocol.
- Project relevance: biology analogue of autoformalization: natural-language
  protocols become structured executable steps.
- Claim boundary: pseudocode reconstruction is useful but not full wet-lab
  correctness.

## BioKGBench

- Type: `paper`
- URL: https://arxiv.org/abs/2407.00466
- Domain: biomedical evidence / KG grounding
- Verifier/reward signal: claim verification, KGQA, RAG-based factual-error
  detection in biomedical knowledge graphs.
- Project relevance: very close to NullAtlas/evidence-status framing.
- Claim boundary: KG factual error detection is narrower than full causal
  biological interpretation.

## GeneAgent

- Type: `paper` / `published article`
- URLs:
  - https://arxiv.org/abs/2405.16205
  - https://www.nature.com/articles/s41592-025-02748-6
- Domain: gene-set analysis / biological database verification
- Verifier/reward signal: self-verification against expert-curated biological
  databases for generated gene-set function claims.
- Project relevance: direct biology example of separating generation from
  evidence checking. The model proposes, then database-grounded verification
  supports, revises, or rejects claims.
- Claim boundary: gene-set functional description is narrower than full
  perturbation-response prediction or causal biology.

## GenoTEX

- Type: `paper`
- URL: https://arxiv.org/html/2406.15341v1
- Domain: gene expression analysis / bioinformatics agents
- Verifier/reward signal: benchmark pipeline for dataset selection,
  preprocessing, and statistical analysis, with annotated code, intermediate
  results, and final results curated by bioinformaticians.
- Project relevance: important biology-specific evaluation pattern: when
  interventional ground truth is absent, score alignment with expert-standard
  analysis procedures and artifacts.
- Claim boundary: expert-alignment benchmark; not direct biological truth.

## CellAgent

- Type: `paper`
- URL: https://arxiv.org/abs/2407.09811
- Domain: single-cell data analysis
- Verifier/reward signal: planner/executor/evaluator roles, hierarchical
  decision-making, self-iterative optimization, and benchmarked single-cell task
  outputs across tissues/cell types.
- Project relevance: practical example of biological expert-role decomposition
  for scRNA-seq analysis workflows.
- Claim boundary: reported automation quality should be treated as task-suite
  specific until independently reproduced.

## BioProBench

- Type: `paper`
- URL: https://arxiv.org/html/2505.07889v3
- Domain: biology / protocols
- Verifier/reward signal: 27k protocols, 550k structured instances; protocol
  QA, step ordering, error correction, generation, reasoning.
- Project relevance: supports procedural-logic and safety-critical reasoning as
  distinct biology agent capabilities.
- Claim boundary: largely textual/procedural; multimodal and actual execution
  remain future work.

## ABC-Bench

- Type: `paper`
- URL: https://arxiv.org/html/2606.11150v1
- Domain: agentic biology / biosecurity
- Verifier/reward signal: algorithmic checks and wet-lab validation for
  bioinformatics and lab-automation tasks.
- Project relevance: strong evidence that agentic biology tasks need practical
  tool execution and governance-aware evaluation.
- Claim boundary: dual-use/biosecurity material should be used only for
  high-level benchmark/governance framing here.

## ToolUniverse / TxAgent / BioDiscoveryAgent

- Type: `paper`
- URLs:
  - https://arxiv.org/html/2509.23426v1
  - https://arxiv.org/html/2503.10970
  - https://arxiv.org/html/2405.17631
- Domain: biomedical/scientific tool ecosystems
- Verifier/reward signal: standardized scientific tools, therapeutic function
  calls, evidence-grounded traces, closed-loop perturbation design.
- Project relevance: practical substrate for biology tool-use trajectories and
  SFT over function-call/evidence traces.
- Claim boundary: these systems do not directly solve calibrated trust in
  fallible SFMs; that remains the local contribution.

## BixBench

- Type: `paper`
- URL: https://arxiv.org/html/2503.00096v2
- Domain: bioinformatics / computational biology agents
- Verifier/reward signal: 53 realistic analytical scenarios and 296 open-answer
  questions; agents work in Python/R/bash/Jupyter over real analysis tasks.
- Project relevance: close to biology-agent workflow evaluation; measures data
  exploration, multi-step analysis, and nuanced interpretation.
- Claim boundary: open-answer expert evaluation remains less clean than exact
  verifiers; reported frontier performance is weak in the original study.

## BiomniBench

- Type: `paper` / `official blog`
- URL: https://www.biorxiv.org/content/10.64898/2026.05.12.724604v1.full-text
- Domain: biology agent process evaluation
- Verifier/reward signal: process-level evaluation of full analytical
  trajectories against expert-designed task-specific rubrics.
- Project relevance: aligns with this project's trajectory-scoring thesis:
  final answer alone hides data handling, method selection, statistics, and
  biological interpretation failures.
- Claim boundary: use detailed numerical claims cautiously; the accessible Phylo
  blog emphasizes that current results are early and expected to evolve.

## CORE-Bench

- Type: `paper`
- URL: https://arxiv.org/html/2409.11363v2
- Domain: computational reproducibility across science
- Verifier/reward signal: 270 tasks from 90 papers across computer science,
  social science, and medicine; agents reproduce results from code/data and
  answer task questions.
- Project relevance: reproducibility is a useful deployment gate for research
  agents before novel discovery claims.
- Claim boundary: not biology-specific, and reproduction is easier to verify
  than novel biological interpretation.

## BioAgent Bench

- Type: `paper`
- URL: https://arxiv.org/html/2601.21800v2
- Domain: practical bioinformatics agent workflows
- Verifier/reward signal: end-to-end tasks such as RNA-seq, variant calling, and
  metagenomics with concrete output artifacts; perturbation stress tests.
- Project relevance: good source for automated assessment plus robustness under
  corrupted inputs, decoy files, and prompt bloat.
- Claim boundary: LLM-grader plus artifact checking; still needs careful audit
  for hidden failure modes.

## Single-Cell Omics Agent Benchmark

- Type: `paper`
- URL: https://arxiv.org/abs/2508.13201
- Domain: single-cell omics / computational biology agents
- Verifier/reward signal: 50 real-world single-cell omics tasks with metrics for
  program synthesis, collaboration, execution efficiency, bioinformatics
  knowledge integration, and task completion quality.
- Project relevance: close to JK's domain. Supports evaluating code generation,
  planning, RAG, self-reflection, and collaboration as distinct agent
  capabilities rather than one final answer.
- Claim boundary: single-cell omics workflows; not a general SFM trust-routing
  benchmark by itself.

## PromptBio-Bench

- Type: `paper` / `official project page`
- URLs:
  - https://www.biorxiv.org/content/10.64898/2026.05.05.723092v2.full-text
  - https://www.promptbio.ai/publications.html
- Domain: bioinformatics agents / end-to-end data analysis
- Verifier/reward signal: 244 expert-curated tasks spanning bioinformatics and
  data science; structured file comparison and scoring against expert reference
  answer files.
- Project relevance: practical artifact-level evaluation: can the agent produce
  the correct output file with the right assumptions and provenance?
- Claim boundary: bioRxiv full-text fetch was flaky in this session; current
  card relies on the project publication page plus search snippets.

## MedAgentGym

- Type: `paper` / `project page`
- URLs:
  - https://openreview.net/forum?id=jHDZEUgS4r
  - https://wshi83.github.io/MedAgentGym-Page/
- Domain: biomedical data science / code-centric medical reasoning
- Verifier/reward signal: 72,413 task instances across 129 categories and 12
  biomedical scenarios; executable sandbox environments, verifiable ground
  truth, interactive feedback, trajectory generation.
- Project relevance: strong example of an agentic training environment, not just
  a benchmark. Directly relevant to SFT/RL over biomedical code trajectories.
- Claim boundary: biomedical data science/code reasoning; not directly SFM
  trust calibration.

## PertEval-scFM / Transcriptomics Foundation Model Benchmarks

- Type: `paper`
- URLs:
  - https://openreview.net/forum?id=t04D9bkKUq&noteId=cVQ96mwheg
  - https://arxiv.org/html/2410.13956v2
- Domain: single-cell foundation model / perturbation prediction
- Verifier/reward signal: perturbation-effect prediction under zero-shot and
  distribution-shift settings; comparison to simple baselines such as PCA/scVI.
- Project relevance: direct SFM calibration anchor. Specialist models may fail
  to beat simple baselines out of distribution, so trust routing must include
  baseline/default actions.
- Claim boundary: benchmark-specific results; still need alignment to local
  Phase 4/C5 setup before making project-level claims.

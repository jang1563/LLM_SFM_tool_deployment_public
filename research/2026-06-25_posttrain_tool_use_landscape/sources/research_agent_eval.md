# Research-Agent Evaluation Source Cards

## ScienceWorld

- Type: `paper`
- URL: https://arxiv.org/abs/2203.07540
- Domain: interactive science environment
- Verifier/reward signal: grounded text environment for elementary-school
  science experiments; tasks require experimental actions, not static recall.
- Project relevance: early evidence that agents need interactive grounding to
  reuse science concepts in new contexts.
- Claim boundary: elementary science level; useful for environment design, not
  frontier biomedical reasoning.

## DiscoveryWorld

- Type: `paper`
- URL: https://arxiv.org/abs/2406.06769
- Domain: simulated scientific discovery
- Verifier/reward signal: simulated text/visual environment with task
  completion, task-relevant actions, and discovered explanatory knowledge.
- Project relevance: good analogue for complete discovery cycles: form
  hypotheses, design/run experiments, analyze results, act on conclusions.
- Claim boundary: fictional/simulated environment; may not transfer to messy
  real-data biology.

## ScienceAgentBench

- Type: `paper`
- URL: https://arxiv.org/abs/2410.05080
- Domain: data-driven scientific discovery
- Verifier/reward signal: 102 tasks from 44 peer-reviewed publications across
  four disciplines, each targeting a self-contained Python program; metrics
  include generated program, execution result, and cost.
- Project relevance: strong source for task-level assessment before claiming
  end-to-end autonomous science.
- Claim boundary: evaluates code generation for scientific sub-tasks, not full
  scientific project completion.

## SciCode

- Type: `paper` / `official project page`
- URLs:
  - https://arxiv.org/abs/2407.13168
  - https://scicode-bench.github.io/
- Domain: cross-domain scientific coding
- Verifier/reward signal: scientist-curated research coding tasks across
  natural-science domains, decomposed into subproblems with gold-standard
  solutions and test cases.
- Project relevance: useful bridge from static QA to research-code workflows;
  spans math, physics, chemistry, biology, and materials science in one frame.
- Claim boundary: code/test-case success is still narrower than real discovery
  or wet-lab validation.

## SciAgentGym / SciAgentBench

- Type: `paper`
- URL: https://arxiv.org/html/2602.12984v1
- Domain: multi-step scientific tool-use
- Verifier/reward signal: interactive environment with 1,780 typed scientific
  tools, structured traces, execution feedback, domain-separated toolkits, and
  long-horizon tasks.
- Project relevance: highly relevant substrate analogue for this repo: typed
  scientific tools, reproducible traces, logic-aware trajectory synthesis, and
  cross-domain transfer.
- Claim boundary: new preprint; treat training/evaluation numbers as tentative
  until independently validated.

## PaperBench

- Type: `paper`
- URL: https://arxiv.org/abs/2504.01848
- Domain: AI research replication
- Verifier/reward signal: 20 ICML 2024 papers; hierarchical rubrics with 8,316
  gradable tasks; LLM judge benchmarked separately.
- Project relevance: strong reproducibility/replication gate before autonomous
  discovery claims.
- Claim boundary: focused on AI/ML research replication, not biology or wet lab.

## CORE-Bench

- Type: `paper`
- URL: https://arxiv.org/html/2409.11363v2
- Domain: computational reproducibility
- Verifier/reward signal: 270 tasks from 90 papers across CS, social science,
  and medicine; agents reproduce results from code/data.
- Project relevance: reproducibility as a deployment gate for research agents.
- Claim boundary: reproduction is not novel discovery and is easier to verify
  than open biological interpretation.

## ResearchClawBench

- Type: `paper`
- URL: https://arxiv.org/html/2606.07591v1
- Domain: end-to-end autonomous scientific research
- Verifier/reward signal: 40 tasks from 10 domains, hidden target papers, raw
  data, executable environments, expert-curated weighted rubrics.
- Project relevance: broad evidence that complete-looking research reports can
  miss protocol/evidence chains.
- Claim boundary: current scoring is report-heavy and dry-lab only; fine-grained
  trajectory scoring is listed as a limitation.

## BenchGuard

- Type: `paper`
- URL: https://arxiv.org/html/2604.24955v1
- Domain: benchmark auditing / evaluation reliability
- Verifier/reward signal: cross-artifact audit of instructions, environment,
  reference solution, and evaluation logic; surfaces structured defects for
  expert review.
- Project relevance: crucial meta-source. In execution-based scientific
  benchmarks, the verifier itself can be wrong; benchmark tasks need audits
  before becoming RLVR reward or evaluation ground truth.
- Claim boundary: audit assistant, not final autonomous judge; high-confidence
  findings still require expert adjudication.

## Evidence Tracing / Execution Provenance Survey

- Type: `paper`
- URL: https://arxiv.org/html/2606.04990v1
- Domain: agent provenance / trustworthy tool-use
- Verifier/reward signal: taxonomy for connecting retrieved evidence, tool
  outputs, memory items, environment observations, intermediate claims, actions,
  and final answers across an execution trajectory.
- Project relevance: strong conceptual support for recording source/tool lineage
  as a first-class layer in scientific-agent training and evaluation.
- Claim boundary: survey/framework source, not a benchmark result.

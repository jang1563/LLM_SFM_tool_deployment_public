# Chemistry Source Cards

## ChemCrow

- Type: `paper`
- URL: https://www.nature.com/articles/s42256-024-00832-8
- Domain: chemistry
- Verifier/reward signal: expert-designed chemistry tools for synthesis, drug
  discovery, and materials design.
- Project relevance: canonical chemistry tool-agent source; supports the
  representation/tool-routing thesis.
- Claim boundary: tool use improves selected tasks but does not remove the need
  for expert/lab validation.

## Coscientist

- Type: `paper`
- URL: https://www.nature.com/articles/s41586-023-06792-0
- Domain: chemistry / lab automation
- Verifier/reward signal: web/documentation search, code execution, robotic lab
  APIs, instrument/lab execution traces.
- Project relevance: bridge from text/tool planning to executable scientific
  experiments.
- Claim boundary: chemistry lab execution has safety, feasibility, and hardware
  constraints; not a generic LLM-only result.

## ChemBench

- Type: `paper`
- URL: https://www.nature.com/articles/s41557-025-01815-x
- Domain: chemistry
- Verifier/reward signal: curated chemistry Q&A against chemist expertise,
  overconfidence analysis.
- Project relevance: supports calibration framing: strong models can still be
  overconfident and weak on basic tasks.
- Claim boundary: Q&A benchmark is not the same as tool/lab execution.

## ChemSafetyBench

- Type: `paper`
- URL: https://arxiv.org/abs/2411.16736
- Domain: chemistry safety
- Verifier/reward signal: automated evaluation over chemical-property queries,
  legality-of-use assessment, and synthesis-method safety/appropriateness.
- Project relevance: adds the policy/safety dimension to chemistry: valid
  structure or reaction reasoning is not enough for deployment.
- Claim boundary: keep as evaluation/governance source; do not use operational
  synthesis content here.

## LabSafety Bench

- Type: `paper` / `official project page`
- URLs:
  - https://yujunzhou.github.io/LabSafetyBench.github.io/
  - https://openreview.net/forum?id=aRqyX0DsmW
- Domain: lab safety across chemistry, biology, physics, and general lab work
- Verifier/reward signal: OSHA-aligned multiple-choice questions, realistic lab
  scenarios, hazard identification, and consequence prediction.
- Project relevance: cross-domain reminder that scientific agents need hard
  safety gates and scenario-level risk assessment, not just task success.
- Claim boundary: mostly evaluates safety awareness; it is not a full lab
  execution environment.

## ChemToolAgent

- Type: `paper`
- URL: https://arxiv.org/html/2411.07228v2
- Domain: chemistry
- Verifier/reward signal: 29-tool chemistry agent evaluated on specialized
  molecule/reaction tasks and general chemistry questions.
- Project relevance: key caution source. Tool augmentation helps specialized
  chemistry tasks but can hurt general chemistry reasoning.
- Claim boundary: "more tools" is not automatically better; tool policies and
  cognitive load matter.

## ChemAgent / ChemToolBench

- Type: `paper`
- URL: https://arxiv.org/html/2506.07551v1
- Domain: chemistry / materials
- Verifier/reward signal: 137 chemistry tools, tool selection, parameter
  filling, step-level fine-tuning, PRM/ORM training.
- Project relevance: strong source for process-supervised chemistry tool-use
  training.
- Claim boundary: chemistry tool selection is still domain-specific and cannot
  be directly mapped to biology SFM trust without calibration tests.

## Anthropic Making Claude A Chemist

- Type: `official`
- URL: https://www.anthropic.com/research/making-claude-a-chemist
- Domain: chemistry
- Verifier/reward signal: representation translation among structures, spectra,
  database queries, patents/publications, and instrument outputs.
- Project relevance: reinforces that chemistry agents need representation-aware
  routing, not generic prose.
- Claim boundary: official product/research framing; use as company signal, not
  as independent benchmark result.

## RxnBench

- Type: `paper`
- URL: https://arxiv.org/abs/2512.23565
- Domain: chemistry / multimodal reaction understanding
- Verifier/reward signal: single-figure QA over 1,525 questions from 305
  reaction schemes and full-document QA across 108 articles.
- Project relevance: strong chemistry evidence that agents must integrate
  reaction diagrams, text, schemes, and tables; explicit text extraction is much
  easier than mechanistic/structural understanding.
- Claim boundary: multimodal benchmark; not a lab-execution benchmark.

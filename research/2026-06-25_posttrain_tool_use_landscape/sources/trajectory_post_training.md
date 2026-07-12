# Trajectory Post-Training Source Cards

Purpose: separate post-training methods by what feedback they can actually use.
This prevents the map from collapsing SFT, RLHF, DPO, process supervision, and
RLVR into one vague "train it better" bucket.

## InstructGPT / RLHF

- Type: `paper`
- URL: https://arxiv.org/abs/2203.02155
- Axis: SFT plus RLHF
- Verifier/reward signal: labeler demonstrations for supervised fine-tuning,
  ranked outputs for reward modeling, then RL from human feedback.
- Project relevance: establishes the canonical post-training stack: imitate
  desired behavior first, then optimize against preference/reward signals.
- Claim boundary: output preference alignment is not the same as scientific
  tool-call correctness or calibrated SFM trust.

## Constitutional AI / RLAIF

- Type: `paper`
- URL: https://arxiv.org/abs/2212.08073
- Axis: self-critique, revision, AI-feedback preference training
- Verifier/reward signal: principle-guided critiques/revisions for supervised
  learning, then model-generated preferences for RL.
- Project relevance: useful for runtime-policy phrasing and scalable preference
  data, especially around `defer`, `verify`, and unsupported-claim behavior.
- Claim boundary: principles can shape behavior, but do not replace
  deterministic scientific validators or hidden biological labels.

## Direct Preference Optimization

- Type: `paper`
- URL: https://arxiv.org/abs/2305.18290
- Axis: preference optimization
- Verifier/reward signal: paired preference data optimized with a classification
  loss, avoiding explicit reward-model fitting and RL rollout during fine-tuning.
- Project relevance: natural method for paired trajectories:
  complete-query-vs-partial-query, calibrated-defer-vs-confident-unsupported,
  gate-compliant-vs-cue-following.
- Claim boundary: preference data must be biologically meaningful; DPO does not
  create a verifier where none exists.

## Process Supervision / PRM800K

- Type: `paper`
- URL: https://arxiv.org/abs/2305.20050
- Axis: process reward modeling
- Verifier/reward signal: step-level human feedback labels over intermediate
  reasoning steps, contrasting process supervision with outcome supervision.
- Project relevance: close analogue for scientific trajectories: score tool
  choice, query construction, evidence integration, and trust action before
  final prose.
- Claim boundary: demonstrated on math reasoning; scientific tool trajectories
  need domain-specific step labels and validator-backed observations.

## AgentGym / AgentEvol

- Type: `paper` / `project`
- URLs:
  - https://arxiv.org/abs/2406.04151
  - https://agentgym.github.io/
- Axis: multi-environment agent training and self-evolution
- Verifier/reward signal: diverse interactive environments, high-quality
  trajectory set, benchmark suite, and scalable agent-evolution method.
- Project relevance: supports a two-part recipe for this repo: bootstrap with
  expert/reference trajectories, then let agents explore inside controlled
  scientific environments.
- Claim boundary: broad agent environments; not specific to biology or SFM
  calibration.

## ReTool

- Type: `paper` / `project`
- URLs:
  - https://arxiv.org/abs/2504.11536
  - https://retool-rl.github.io/
- Axis: tool-integrated RL after cold-start SFT
- Verifier/reward signal: synthetic code-augmented reasoning traces for
  fine-tuning, followed by outcome-reward RL with real-time code execution in
  rollouts.
- Project relevance: clear method pattern for this project: generate
  cold-start scientific tool trajectories, then optimize tool-use strategy only
  where outcome rewards are reliable.
- Claim boundary: math/code-interpreter setting; biology needs audited tools,
  provenance, and calibrated specialist outputs.

## Search-R1

- Type: `paper` / `code`
- URLs:
  - https://arxiv.org/abs/2503.09516
  - https://github.com/PeterGriffinJin/Search-R1
- Axis: RL for retrieval/tool-augmented reasoning
- Verifier/reward signal: multi-turn search queries inside reasoning rollouts,
  retrieved-token masking for stable RL, and outcome-based rewards.
- Project relevance: strong analogue for biomedical database querying and
  evidence retrieval: train the model to ask the tool at the right moment with
  useful queries, not merely append RAG context.
- Claim boundary: QA/search setting; biological claim verification still needs
  evidence-status labels and source-specific validators.

## WebRL

- Type: `paper`
- URL: https://arxiv.org/abs/2411.02337
- Axis: web-agent RL with online curriculum
- Verifier/reward signal: self-evolving curriculum from failed attempts,
  outcome-supervised reward model, adaptive online RL to handle sparse feedback
  and policy drift.
- Project relevance: useful for long-horizon scientific tools where failures can
  seed new training tasks and sparse rewards need curriculum design.
- Claim boundary: web task success is not scientific correctness; use as agent
  training infrastructure analogy.

## ToolRL

- Type: `paper`
- URL: https://arxiv.org/abs/2504.13958
- Axis: reward design for tool selection/application
- Verifier/reward signal: reward strategies varied by type, scale, granularity,
  and temporal dynamics; trained with GRPO on tool-use tasks.
- Project relevance: direct support for decomposed rewards in this repo:
  format/schema, tool selection, argument completeness, observation use, final
  action, and cost.
- Claim boundary: tool-use reward design helps only if scientific validators are
  correct and sufficiently dense.

## Agent Lightning

- Type: `paper` / `framework`
- URL: https://arxiv.org/abs/2508.03680
- Axis: RL infrastructure for existing agents
- Verifier/reward signal: decouples agent execution from training, formulates
  execution as an MDP, and decomposes trajectories into training transitions via
  hierarchical credit assignment.
- Project relevance: useful systems analogue for turning existing biology
  agents/tools into training data without rewriting the whole runtime.
- Claim boundary: infrastructure source; project still needs domain rewards,
  validators, and safe action policies.

## VerlTool

- Type: `paper` / `framework`
- URL: https://arxiv.org/abs/2509.01055
- Axis: agentic RL with standardized tool APIs
- Verifier/reward signal: multi-turn trajectories with text/image/video
  observations, unified tool management, asynchronous rollouts, and evaluation
  across tool-use domains including code, search, SQL, vision, web, and software.
- Project relevance: supports treating scientific tool-use RL as multi-turn,
  multi-observation agentic RL rather than single-turn RLVR.
- Claim boundary: generic infrastructure; biology deployment still requires
  evidence-status and SFM calibration gates outside the learner.

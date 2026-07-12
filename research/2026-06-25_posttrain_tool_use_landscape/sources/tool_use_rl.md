# General Tool-Use / Post-Training Source Cards

## ReAct

- Type: `paper`
- URL: https://arxiv.org/abs/2210.03629
- Axis: prompting / trajectory format
- Verifier/reward signal: interleaved reasoning/action trajectory format.
- Project relevance: canonical `reason -> act -> observe` shape.
- Claim boundary: prompting paradigm, not post-training by itself.

## Toolformer

- Type: `paper`
- URL: https://arxiv.org/abs/2302.04761
- Axis: self-supervised tool-use training
- Verifier/reward signal: self-supervised API-call insertion/filtering.
- Project relevance: early model for learning when/how to call tools.
- Claim boundary: not enough for high-stakes biology verification.

## API-Bank / ToolLLM / ToolBench / BFCL

- Type: `paper` / `benchmark`
- URLs:
  - https://arxiv.org/abs/2304.08244
  - https://arxiv.org/abs/2307.16789
  - https://proceedings.mlr.press/v267/patil25a.html
  - https://gorilla.cs.berkeley.edu/leaderboard.html
- Axis: API/function-calling datasets and evaluation
- Verifier/reward signal: tool selection, argument construction, multi-step
  API calls, function-call accuracy.
- Project relevance: baseline substrate for scientific tool-call syntax and
  planning.
- Claim boundary: function-call correctness is not evidence correctness.

## tau-bench

- Type: `paper`
- URL: https://arxiv.org/abs/2406.12045
- Axis: domain-policy constrained agents
- Verifier/reward signal: dynamic user-agent-tool interaction under policy.
- Project relevance: strong analogy for policy-constrained scientific workflows.
- Claim boundary: not biology-specific.

## ToolRL / Tool Zero / Agent Lightning / VERL-Tool / Tool-R0

- Type: `paper`
- URLs:
  - https://arxiv.org/html/2504.13958v1
  - https://aclanthology.org/2025.findings-emnlp.485.pdf
  - https://arxiv.org/abs/2508.03680
  - https://arxiv.org/html/2509.01055v1
  - https://arxiv.org/html/2602.21320v1
- Axis: RL/RLVR for tool-use trajectories
- Verifier/reward signal: decomposed format/tool/correctness rewards,
  trajectory credit assignment, holistic agentic RL.
- Project relevance: method family for training tool-use trajectories after an
  environment exists.
- Claim boundary: reward design must match verifier quality; final-answer RL is
  too sparse for scientific trust routing.

## ReTool / Search-R1 / WebRL / AgentGym

- Type: `paper`
- URLs:
  - https://arxiv.org/abs/2504.11536
  - https://arxiv.org/abs/2503.09516
  - https://arxiv.org/abs/2411.02337
  - https://arxiv.org/abs/2406.04151
- Axis: interactive trajectory post-training
- Verifier/reward signal: code execution, search engine calls, web actions,
  online curricula, and environment trajectories.
- Project relevance: supports training agents to decide when and how to call
  external systems during reasoning, not merely how to format one API call.
- Claim boundary: these are mostly math/search/web/general-agent settings; map
  to biology only after source/evaluator audits.

## OpenAI Agents / Deep Research

- Type: `official`
- URLs:
  - https://openai.com/index/new-tools-for-building-agents/
  - https://openai.com/index/introducing-deep-research/
  - https://developers.openai.com/cookbook/topic/agents
- Axis: product/platform tool-use infrastructure
- Verifier/reward signal: tool calls, tracing, guardrails, browser/Python/file
  use, real-world task training.
- Project relevance: product landscape signal that tool-use trajectories are now
  first-class agent infrastructure.
- Claim boundary: official product framing; use for ecosystem direction, not
  independent experimental evidence.

## ResearchClawBench

- Type: `paper`
- URL: https://arxiv.org/html/2606.07591v1
- Axis: end-to-end autonomous scientific research evaluation
- Verifier/reward signal: 40 tasks from 10 scientific domains, hidden target
  papers, raw data, executable environments, expert-curated weighted rubrics.
- Project relevance: broad evidence that autonomous science agents still miss
  protocol/evidence chains even when producing complete-looking reports.
- Claim boundary: current scoring mainly targets final reports and dry-lab
  settings; fine-grained trajectory scoring is listed as a limitation.

See also `research_agent_eval.md` for broader research-agent and benchmark
auditing sources that are not specifically post-training methods. See
`trajectory_post_training.md` for the method ladder from SFT/RLHF/DPO to
process supervision and tool-use RL.

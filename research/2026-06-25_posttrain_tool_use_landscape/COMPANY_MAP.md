# Company / Product Map

Purpose: capture current public direction for agentic tool deployment, especially
life sciences and scientific workflow products.

## Findings

| company / product | public direction | relevance to this project |
| --- | --- | --- |
| Anthropic Claude for Life Sciences | Connectors, Agent Skills, dedicated life-sciences support, Benchling/PubMed/10x/Synapse/Wiley/BioRender and newer ChEMBL/Owkin-style integrations. | Closest product target. Project should speak the language of life-science tool use, skills, connectors, MCP, and reproducible workflows. |
| Anthropic healthcare/life-sciences expansion | Adds Medidata, ClinicalTrials.gov, ToolUniverse, bioRxiv/medRxiv, Open Targets, ChEMBL, Owkin, plus skills for problem selection, Allotrope conversion, scVI-tools, Nextflow, and clinical protocols. | Very strong signal that life-science agents are becoming connector + skill + tool-universe systems across preclinical, clinical, and regulatory workflows. |
| Anthropic agents in biology / gget virus | Deterministic retrieval layer raised accuracy near 100% in a biology-data retrieval setting. | External validation for deterministic tool layers as reliability infrastructure. |
| Anthropic BioMysteryBench | Objective bioinformatics benchmark where Claude can install tools and access NCBI/Ensembl-style resources in a container. | Useful evaluation model, but this project can improve on it by scoring trajectories and trust/verify/defer decisions, not only final answers. |
| Anthropic long-running scientific computing | Multi-day scientific coding workflows depend on test oracles, persistent memory, and progress files. | Directly supports durable agent lab-notes, test harnesses, and long-horizon trajectory evaluation. |
| Anthropic chemistry/NMR work | Focuses on representation translation across structures, instrument readouts, databases, patents, and publications. | Reinforces that scientific agents need representation-aware tool routing, not generic prose generation. |
| Anthropic MCP | Standard protocol for connecting models to tools, data sources, prompts, and resources. | Deployment layer for binding tool availability and auditability. |
| Claude Agent Skills | Portable instruction/script/resource bundles, callable across Claude apps, Claude Code, and API. | Candidate packaging format for scientific protocols, tool wrappers, validators, and training/eval harnesses. |
| Benchling + Anthropic | Source-linked answers over R&D records; agents search structured assay/molecule data and lab-notebook context. | Shows enterprise life-science agents need provenance and data-system integration, not just chat. |
| Manifold + Anthropic | Agents translate scientist-language questions into execution over specialized datasets/tools. | Very close to `semantic intent -> technical tool execution` framing. |
| OpenAI agents platform | Responses API, web/file/computer tools, Agents SDK, tracing, evals, guardrails. | Comparable platform-layer agent infrastructure; useful for arguing this is a deployment/post-training systems problem. |
| OpenAI deep research | RL-trained browser/Python tool use on real-world research tasks. | Evidence that post-training on tool trajectories is a mainstream frontier. |
| Google DeepMind Co-Scientist | Multi-agent hypothesis generation, reflection, ranking, evolution, and lab validation case studies. | Shows science-agent products are moving toward structured multi-agent reasoning; project distinction is calibrated tool trust and enforcement. |
| Google DeepMind AlphaEvolve | LLM coding agent plus automated evaluators and evolutionary search. | Strong analogy for verifiable/evaluator-driven post-training or search loops. |
| FutureHouse | Scientific agents with papers, specialized scientific tools, databases, and source-quality evaluation. | Product analogue for scientific agents using tools; less centered on SFM trust-routing. |
| ToolUniverse / Zitnik Lab ecosystem | Open scientific tool ecosystem with 600+ tools, unified tool specs, MCP support, and specialized agent skills. | Most concrete external substrate for building a biology tool-use training/eval harness. |

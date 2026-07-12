# Physics Source Cards

## FEABench

- Type: `official` / `paper`
- URL: https://research.google/pubs/feabench-evaluating-language-models-on-real-world-physics-reasoning-ability/
- Domain: physics / engineering
- Verifier/reward signal: COMSOL Multiphysics API calls, FEA solver execution,
  numerical outputs, iterative tool use.
- Project relevance: strong physics analogue for agents operating real
  scientific software, not just answering questions.
- Claim boundary: execution validity is not the same as fully correct physics;
  published report says tested systems did not completely solve any problem.

## Mind's Eye

- Type: `paper`
- URL: https://arxiv.org/abs/2210.05359
- Domain: physics
- Verifier/reward signal: MuJoCo simulation outputs used to ground physical
  reasoning.
- Project relevance: early evidence that an external simulator can improve
  language-model physical reasoning.
- Claim boundary: simulator grounding helps when the simulator captures the
  relevant world; it does not solve model mismatch or domain transfer.

## PDEBench

- Type: `paper`
- URL: https://arxiv.org/abs/2210.07182
- Domain: physics / scientific ML
- Verifier/reward signal: time-dependent PDE simulation datasets, classical
  numerical baselines, ML baselines, extensible data generation.
- Project relevance: anchors physics rewards in numerical simulation, held-out
  initial/boundary conditions, and solver-style validation.
- Claim boundary: benchmark success may still be simulator-distribution success,
  not real-world physical discovery.

## DiscoverPhysics

- Type: `paper`
- URL: https://arxiv.org/html/2605.26087v1
- Domain: physics
- Verifier/reward signal: black-box N-body simulator, predictive accuracy,
  explanation scoring against hidden world rules.
- Project relevance: strong analogy for iterative experiment design and latent
  law discovery.
- Claim boundary: simulator worlds are controlled; transfer to real scientific
  discovery remains an open question.

## PhysGym

- Type: `paper`
- URL: https://arxiv.org/html/2507.15550v2
- Domain: interactive physics discovery
- Verifier/reward signal: controlled interactive simulations where agents vary
  inputs, observe outputs, and infer equations under different levels of prior
  knowledge.
- Project relevance: sharp physics-specific lesson: benchmark difficulty changes
  when context, variable names, and priors are masked. This is a clean analogue
  for biological context shift.
- Claim boundary: simulated physics environments are still cleaner than
  biological assays and literature-derived evidence.

## PDE-Grounded Intent Verification

- Type: `paper`
- URL: https://arxiv.org/html/2605.09360v1
- Domain: multiphysics simulation / benchmark auditing
- Verifier/reward signal: checks whether runnable PDE simulation code actually
  matches the intended boundary conditions, source terms, diffusivity fields,
  transient formulation, and initial conditions.
- Project relevance: strong caution source: "simulation runs" is only L1.
  Scientific rewards must verify intent fidelity, not just executability.
- Claim boundary: focuses on generated multiphysics simulation code; do not
  generalize directly to every physics task.

## PHYBench

- Type: `paper`
- URL: https://arxiv.org/abs/2504.16074
- Domain: physics
- Verifier/reward signal: original physics problems, expression edit distance
  for equation-level scoring.
- Project relevance: supports non-binary evaluation of process/equations, not
  only final answer matching.
- Claim boundary: mostly reasoning benchmark, not a full tool-use environment.

## MCP-SIM

- Type: `paper`
- URL: https://www.nature.com/articles/s44387-025-00057-z
- Domain: physics / engineering simulation
- Verifier/reward signal: simulation construction, code generation, error
  diagnosis, validated reports.
- Project relevance: shows value of plan-act-reflect-revise cycles for physical
  simulation workflows.
- Claim boundary: reported robustness is limited to tested benchmark suite.

## CritPt

- Type: `paper`
- URL: https://arxiv.org/html/2509.26574v3
- Domain: frontier physics reasoning
- Verifier/reward signal: unpublished research-level physics problems across
  condensed matter, quantum, AMO, astrophysics, high-energy, mathematical
  physics, statistical physics, nuclear physics, nonlinear dynamics, fluid
  dynamics, and biophysics.
- Project relevance: pushes physics evaluation beyond textbook/exam questions
  toward research-like reasoning and trust in reasoning traces.
- Claim boundary: difficult open-ended reasoning benchmark; not primarily a
  tool-execution environment.

## Physics-IQ Verified

- Type: `paper`
- URL: https://arxiv.org/html/2606.18943v1
- Domain: physical understanding in video generative models
- Verifier/reward signal: controlled real-world physical experiment videos,
  sample-level scoring, prompt/ground-truth audit, metric aggregation fixes.
- Project relevance: good reminder that benchmark design itself can confound
  the intended capability; verifier quality must be audited.
- Claim boundary: video-generation physical realism, not LLM scientific
  reasoning directly.

## QuantiPhy

- Type: `paper`
- URL: https://arxiv.org/html/2512.19526v1
- Domain: vision-language physical reasoning
- Verifier/reward signal: numerical ground truth for size, velocity, and
  acceleration over 3.3k+ video-text instances.
- Project relevance: supports the "plausible prose is not quantitative
  correctness" lesson; useful analogy for SFM calibration.
- Claim boundary: kinematic inference benchmark; not broad physics discovery.

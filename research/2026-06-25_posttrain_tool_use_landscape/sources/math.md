# Math Source Cards

## AlphaProof / AlphaGeometry

- Type: `official` + `paper`
- URLs:
  - https://deepmind.google/blog/ai-solves-imo-problems-at-silver-medal-level/
  - https://deepmind.google/blog/alphageometry-an-olympiad-level-ai-system-for-geometry/
  - https://www.nature.com/articles/s41586-025-09833-y
- Domain: math
- Verifier/reward signal: formal proof environment, proof checker, formalized
  theorem, test-time RL over variants.
- Project relevance: cleanest contrast case for verifier-driven RL. Useful as
  the upper bound for "RLVR works when the verifier is strong."
- Claim boundary: do not generalize proof-checker style rewards to biology
  without identifying equivalent verifier slices.

## LeanDojo

- Type: `paper`
- URL: https://arxiv.org/abs/2306.15626
- Domain: math
- Verifier/reward signal: Lean proof states, tactics, premises, accessible
  premise annotations, benchmark splits.
- Project relevance: shows how a tool environment can expose trajectory-level
  state for training and evaluation.
- Claim boundary: theorem proving has programmatic feedback that most biology
  interpretation tasks do not.

## miniF2F

- Type: `paper`
- URL: https://arxiv.org/abs/2109.00110
- Domain: math
- Verifier/reward signal: formal Olympiad-level problem statements across Lean,
  Metamath, Isabelle, and HOL Light.
- Project relevance: benchmark portability across formal systems; good analogy
  for schema portability across scientific tools.
- Claim boundary: statement correctness and intended meaning still depend on
  formalization quality.

## ProofNet

- Type: `paper`
- URL: https://arxiv.org/abs/2302.12433
- Domain: math
- Verifier/reward signal: paired natural-language theorem statements, natural
  proofs, and Lean statements.
- Project relevance: captures the messy prose-to-verifier bridge. Useful when
  explaining why autoformalization is analogous to mapping biological questions
  into tool/evidence schemas.
- Claim boundary: autoformalization errors can make a clean proof checker check
  the wrong problem.

## PutnamBench

- Type: `paper` / `official project page`
- URLs:
  - https://trishullab.github.io/PutnamBench/
  - https://arxiv.org/abs/2407.11214
- Domain: formal mathematics
- Verifier/reward signal: hand-constructed formalizations of Putnam competition
  problems across Lean 4, Isabelle, and Coq.
- Project relevance: strengthens the math contrast: verifier portability can be
  studied explicitly when the target object is a formal theorem.
- Claim boundary: theorem-prover benchmark success still depends on whether the
  formalized statement matches the intended informal problem.

## FrontierMath

- Type: `official` / `paper`
- URLs:
  - https://epoch.ai/frontiermath
  - https://arxiv.org/abs/2411.04872
- Domain: advanced mathematical reasoning
- Verifier/reward signal: expert-authored, unpublished advanced math problems;
  evaluation uses automated verification where possible, but is not the same as
  formal proof-checker trajectory verification.
- Project relevance: useful counterpoint inside math itself: not all math evals
  are clean formal RLVR; some still rely on expert-vetted problem secrecy and
  grader design.
- Claim boundary: use as capability/evaluation source, not as proof that
  open-ended math reasoning has a cheap formal reward.

## FunSearch

- Type: `paper`
- URL: https://www.nature.com/articles/s41586-023-06924-6
- Domain: math / executable discovery
- Verifier/reward signal: executable program evaluator selects and improves LLM
  proposals.
- Project relevance: useful bridge from math to scientific discovery: LLM
  generates candidates, evaluator blocks confabulations.
- Claim boundary: works best when candidate quality can be evaluated
  automatically and cheaply.

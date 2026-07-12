# Post-Train Deployment Framing

## Core Distinction

This project is not about pretraining a biology foundation model.

It is about **post-training a general reasoning model for deployed scientific
tool use**:

```text
scientific intent -> database/tool selection -> tool call trajectory
-> evidence validation -> trust/verify/defer action -> auditable result
```

The target capability is not "knows more biology from weights." The target is:

- chooses the right biological database or specialist model,
- calls tools with valid schemas and complete filters,
- handles database quirks and missing evidence,
- grounds conclusions in returned records,
- knows when to verify, defer, or request more evidence,
- resists misleading reliability framing,
- produces reproducible trajectories.

## Why Post-Training

Current scientific-agent systems increasingly depend on this layer:
post-training, evaluation design, RL environments, agentic workflows, and
biology tool infrastructure. The project builds the substrate that makes those
behaviors trainable and auditable.

Claude itself is not available for direct training, so the experimental route is:

1. Use Claude/frontier models as reference agents and evaluators where useful.
2. Use strong open-source models as trainable agents.
3. Generate or curate tool-use trajectories from controlled biology tasks.
4. Train with SFT / preference optimization / RL from verifiable rewards.
5. Evaluate whether post-training improves tool choice, database use,
   verification allocation, and trajectory faithfulness.

## Training Targets

### SFT

Train on high-quality trajectories:

- correct database/tool choice,
- valid query construction,
- complete filter application,
- source-backed evidence synthesis,
- correct use of `trust_sfm`, `verify_assay`, `default_baseline`, and `defer`.

### DPO / Preference Optimization

Use paired trajectories:

- complete query vs partial query,
- deterministic tool call vs hallucinated citation,
- calibrated defer vs confident unsupported answer,
- raw-risk gate compliance vs prompt-framing cue-following,
- source-backed synthesis vs plausible prose without evidence.

### RLHF

Use human/domain-expert ratings where the reward cannot be made fully
deterministic:

- biological workflow realism,
- usefulness to a scientist,
- interpretation quality,
- whether caveats are scoped correctly.

### RLVR

Use verifiable rewards where possible:

- exact database retrieval/count tasks,
- schema validity,
- citation existence,
- value/range validity,
- hidden-label correctness,
- tool-call completeness,
- cost-aware routing reward.

## Relation To Existing Projects

Keep the existing projects distinct:

- `LLM_SFM_interpretability`: measures what LLMs trust when reading specialist
  model outputs.
- `LLM_SFM_phase4_planning`: shows enforcement-based trust routing can beat
  free-form LLM decisions under calibrated risk.
- `bio-sfm-trust-core`: reusable gate/calibration/conformal engine.
- `bio_sfm_designer`: DBTL application with Claude as orchestrator and an
  external trust gate.
- `NullAtlas`: deterministic negative-evidence and claim-guardrail layer.
- `Bio_Grounding_Eval`: content-grounding and routing map.

This folder's identity should be:

> Build the post-training substrate for biology tool-use deployment: datasets,
> reward functions, trajectory schemas, and open-model training/evaluation loops
> that improve scientific tool use.

## First Concrete Direction

Start with a small, trainable open-model environment rather than a new SFM:

1. Define trajectory schema shared with the trust-gate action set.
2. Pick one tool domain with deterministic rewards:
   - database retrieval/completeness,
   - NullAtlas claim verification,
   - structure-confidence gate compliance,
   - variant-effect evidence routing.
3. Generate expert/reference trajectories.
4. Train an open model with SFT first.
5. Add preference pairs or RLVR only after the deterministic evaluator is stable.

The strongest first experiment is not "does the model answer biology questions?"
It is:

> Does post-training improve valid, complete, verifiable biology tool trajectories
> compared with prompting alone?

# Project Brief: Enforcement-Layer SFM Tool Deployment

## One-Sentence Aim

Turn the existing LLM x SFM trust-routing audits into a post-training and
deployment harness where tool calls, database use, trust, verification, cheap
baselines, and deferral are explicit, logged, trainable, and policy-constrained
actions.

## Why This Project Exists

The parent trust-audit work already tested the presentation layer. The consistent
finding is negative but useful: a calibrated reliability interface is not a free
win. The next project layer should therefore test whether deployment enforcement
and post-training can make the right tool trajectory happen.

Anthropic's life-sciences direction points toward Claude connected to scientific
platforms and skills. Anthropic's agents-in-biology work makes the complementary
point that biology needs deterministic execution/retrieval layers because agent
answers can look plausible while being wrong. This project sits exactly between
those: Claude-like orchestration over specialist biology tools, but with measured
trust/verify/defer behavior.

The distinguishing bet is **post-train / deployment**, not pretraining. Claude
cannot be directly trained here, but open-source models can be trained on the
trajectory substrate: SFT for expert tool traces, preference optimization for
better/worse trajectories, RLHF-style domain ratings where needed, and RLVR
where database/tool rewards are deterministic.

## Design Principles

1. **Separate routing from reasoning.** The model can explain, but routing is a
   scored action with constraints.
2. **Use priors before calls.** Web-exposure, representation type, and cheap
   baseline dominance can decide whether self-answering is even allowed.
3. **Make specialists tools, not authorities.** Specialist outputs enter as
   evidence packets with calibration status, domain, and known failure modes.
4. **Prefer deterministic guardrails where available.** NullAtlas-style MCP
   checks and value validators should be hard gates for claim types they cover.
5. **Keep the cheap baseline alive.** In perturbation settings, defaulting to a
   free additive baseline can dominate trusting an FM.
6. **Score trajectories, not prose.** The unit of evaluation is the sequence of
   tool calls and actions under cost, not just final answer quality.
7. **Make trajectories trainable.** The same schema should support prompt-only
   baselines, expert trajectory generation, SFT, preference pairs, and
   verifiable-reward training.

## Current Local Phase 4 Status

This is not starting from zero. A sibling planning folder already records a
Phase 4a internal result:

```text
<local-workspace>/LLM_SFM_phase4_planning/
```

The key result there is C1-C4 on an N=57 leakage-controlled hard-complex
substrate:

- free-form LLM routing loses to a deterministic calibrated-risk gate at
  meaningful verification costs,
- inverted/misleading cue framing degrades the free-form LLM,
- the gate is robust because it computes from raw calibrated risk rather than
  reading prompt-supplied reliability framing,
- the next boundary test is C5, antibody-antigen OOD, where structure-predictor
  confidence calibration is known to degrade.

A second sibling repo already implements the application pattern:

```text
<local-workspace>/bio_sfm_designer/
```

There, Claude is the orchestrator over ProteinMPNN / ESMFold / Boltz-2, while an
external calibrated trust gate decides `trust_sfm | verify_assay |
default_baseline | defer`.

## Minimal Harness Contract

### Action Enum

```text
answer_self
call_specialist_tool
trust_specialist_output
verify_with_assay_or_database
use_cheap_baseline
defer_or_request_more_evidence
reject_or_flag_unsupported_claim
```

### Evidence Packet

```text
input_id
representation_type
web_exposure_tag
specialist_name
specialist_output
specialist_confidence
calibration_status
cheap_baseline_output
baseline_dominance_flag
negative_evidence_status
claim_guard_status
allowed_actions
hidden_truth_pointer
```

### Policy Hooks

- `web_zero_requires_tool_or_defer`: web-zero representation-to-property mapping
  cannot be self-answered unless a deterministic computation tool covers it.
- `uncalibrated_specialist_requires_verify_or_baseline`: uncalibrated SFM
  confidence cannot be treated as a trust signal.
- `baseline_dominates_default_to_baseline`: when a cheap baseline dominates the
  specialist in the known regime, the default action is baseline, not trust.
- `recorded_negative_hard_flag`: measured negative evidence can block or flag a
  claim; unsupported evidence remains advisory.
- `continuous_confidence_threshold`: confidence is thresholded continuously;
  binary self-reported defer is not taken literally.

## First Substrate Recommendation After Recovery

Do not restart with a generic perturbation or variant spec. The latest local
Phase 4 note already names the next scientific boundary:

**C5: antibody-antigen OOD.**

Reason: it tests the important boundary condition. A general calibrated gate can
beat free-form LLM routing in-distribution, but antibody-antigen complexes are a
known regime where structure-predictor confidence can be miscalibrated. This is
the right place to ask whether a deterministic gate needs regime-specific
calibration or whether contextual LLM reasoning recovers something useful.

Recommendation: next inspect `bio-sfm-trust-core` and the C5 planning/compute
requirements before writing new code in this folder.

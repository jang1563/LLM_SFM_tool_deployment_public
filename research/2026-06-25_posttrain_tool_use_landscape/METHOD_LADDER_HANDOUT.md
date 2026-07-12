# Method Ladder Handout: Post-Training Scientific Tool Trajectories

Purpose: tutor-mode bridge from the research map to a concrete training plan.

## Core Rule

Choose the post-training method from the feedback signal, not from hype:

```text
method = f(available trajectories, preference labels, verifiable rewards, runtime policy)
```

If the feedback signal is weak, the method should stay weak too. Do not use RLVR
language for tasks where the reward is actually expert judgment or a fragile
benchmark artifact.

## Ladder

| method | needs | can teach | biology trajectory slice | main failure mode |
| --- | --- | --- | --- | --- |
| Prompt baseline | Task schema and allowed actions. | Measures untrained routing and tool-use behavior. | `self_answer` vs `call_tool`; raw tendency to trust specialist cards. | Looks competent because prose is fluent; no stable behavior change. |
| SFT | Reference trajectories. | Valid tool choice, complete arguments, evidence packet format, action enum use. | Database query construction; extracting pLDDT/PAE/ipTM; filling `trust/verify/baseline/defer`. | Imitates traces but may fail on unfamiliar tools or OOD regimes. |
| Preference / DPO | Paired better/worse trajectories. | Prefer complete, calibrated, source-backed behavior over plausible shortcuts. | Complete PubMed/query filters vs partial ones; calibrated defer vs unsupported confidence; gate compliance vs cue-following. | Preference labels can encode taste or presentation instead of scientific validity. |
| RLHF / RLAIF | Human or AI preference labels plus a reward model/policy principles. | Helpful policy behavior and conservative claim style where exact reward is unavailable. | When to explain uncertainty, ask for more evidence, refuse unsupported causal claims, or escalate to verification. | Reward model can reward agreeable prose rather than correct evidence handling. |
| Process supervision | Step labels or validators for intermediate actions. | Correct tool choice, query construction, evidence integration, and gate decisions before final answer. | Score `call_tool`, `observe`, `evidence_status`, `specialist_metric_scope`, `action_choice`. | Step labels are expensive and can overfit to annotation style. |
| RLVR / tool-use RL | Audited environment with verifiable rewards. | Optimize policy over tool calls and actions under cost. | Schema validity, accession/source exactness, hidden evidence labels, metric extraction, gate compliance. | Sparse or wrong reward teaches shortcuts; benchmark defects become training signal. |
| Runtime enforcement | Deterministic validators, calibrated thresholds, and policy gates. | Blocks unsafe or unsupported actions regardless of model preference. | `uncalibrated_specialist_requires_verify_or_baseline`; `baseline_dominates_default_to_baseline`; `recorded_negative_hard_flag`. | Overly rigid gates can block useful work if calibration scope is wrong or stale. |

## Biology Mapping

The biology trajectory should be split into layers:

```text
intent
-> tool/database choice
-> valid call and complete filters
-> evidence packet
-> specialist confidence/baseline comparison
-> calibrated action
-> sourced final answer
```

Method assignment:

| trajectory layer | best first method | why |
| --- | --- | --- |
| Tool/database choice | SFT, then DPO | Expert traces and paired bad/good choices are easy to curate. |
| Schema and filter validity | SFT, RLVR | Validity can be checked automatically. |
| Evidence packet completeness | SFT, process supervision | Intermediate artifacts are visible and labelable. |
| Evidence-status label | RLVR if hidden label exists; otherwise expert/process labels | Reward depends on whether a gold label exists. |
| Specialist confidence extraction | RLVR | Metric extraction and scope typing are deterministic. |
| Trust/verify/baseline/defer | Runtime enforcement plus DPO/process supervision | Some parts are policy; some parts are expert judgment. |
| Biological interpretation | RLHF/RLAIF/DPO with expert review | Mostly L3 judgment, not clean RLVR. |

## First Experiment Recipe

Use this order:

1. Build a prompt-only baseline with the action enum fixed.
2. Create 50-200 compact reference trajectories for Stage A retrieval/evidence
   tasks.
3. SFT on the reference trajectories.
4. Create paired trajectories for common failure modes:
   partial query, hallucinated citation, unsupported trust, unnecessary tool
   call, failure to defer.
5. Run DPO/preference optimization on those pairs.
6. Add process rewards for intermediate artifacts that can be checked.
7. Add RLVR only for audited deterministic slices.
8. Keep runtime gates outside the learner for high-stakes actions.

## C5 Transfer

For antibody-antigen OOD:

- SFT teaches the model to extract and record metric fields.
- DPO teaches it to prefer calibrated/fail-closed trajectories over persuasive
  reliability-card trust.
- Process supervision scores correct metric scope and regime identification.
- RLVR can score schema, extraction, hidden interface label, and gate compliance.
- Runtime enforcement decides whether `trust_sfm` is even allowed under the
  current calibration regime.

## Tutor Check Questions

1. If a task has only expert judgment and no hidden label, is it RLVR?
2. If a model extracts ipTM correctly but the Ab-Ag regime is uncalibrated, can
   it trust the SFM?
3. If a simple baseline beats a foundation model under distribution shift, which
   action should be available?
4. What is the difference between teaching the model to defer and enforcing a
   deterministic defer gate?

## Short Answer Key

1. No. It is preference/process supervision unless a verifiable reward is added.
2. No. Correct extraction is not calibrated trust.
3. `baseline`, and often `verify` or `defer` depending on evidence status.
4. Teaching changes model tendencies; enforcement constrains allowed actions.

# Project Status

Last updated: 2026-07-23

## Current Thesis

Biology agents should be evaluated on executable
`tool -> evidence packet -> terminal action` trajectories. Learned policies
remain useful where evidence-conditioned choices are genuinely uncertain, but
fixed schema transformations and safety invariants belong in fail-closed
runtime code. DPO/RLVR and Hugging Face publication remain gated by broader,
independent evidence.

## Current Result

The prospective Stage A development checkpoint is complete.

- Built 25 public development tasks with case-specific, model-visible integer
  drug and condition IDs. They are disjoint from the original public Stage A
  manifest and use a source file declared as a zero-overlap exclusion in the
  completed sealed-set commitment.
- Built 180 synthetic routing states across clean, missing-attribution, stale,
  contradiction, invalid-value, partial-query, wrong-tool, and unavailable-tool
  conditions.
- Frozen Qwen2.5-0.5B candidate routing predicts `verify/insufficient` on all
  180 rows: 35/180 exact, below the 80/180 best static pair.
- The deterministic routing gate is 180/180 by construction on its policy-test
  mutations. The runtime hybrid is 115/180, has zero unsafe grounding, and has
  zero decisive coverage because every decisive model/gate disagreement is
  sent to verification.
- On 25 real-query tool-call prompts, the base model and the pre-prospective
  placeholder-SFT state both score 0/25 exact. The frozen SFT also reduces
  parseable JSON from 20/25 to 14/25.
- An adaptive explicit output contract fixes top-level target keys to 25/25 but
  still scores 0/25 for strict call shape and exact output.
- Because Stage A currently uses a fixed four-tool sequence and copies two
  already-visible typed IDs, this step is now implemented as a runtime
  compiler, not a learned policy. It matches 25/25 clean targets and rejects
  150/150 malformed contract mutations for the intended reason.

The completed private sealed set was not read, regenerated, rescored, or used
for prompt selection. Its commitment remains
`post_training/stage_a_sealed_extension_commitment_2026-07-10.json` and its
previous one-time routing result remains 5/25. Do not tune on or rescore these
25 sealed rows.

## Source Changes

[JSONSchemaBench](https://arxiv.org/abs/2501.10868) separates constrained-output
coverage, efficiency, and output quality. That distinction changes the local
design: schema validity is measured independently from evidence/action quality,
and deterministic copy-only query construction is enforced at runtime instead
of treated as scientific model competence.

## Next Decision

Proceed with `stage_b_c5_manifest_prototype_after_stage_a_runtime_split`.

1. Keep the Stage A tool-query compiler in the runtime layer and add no
   corrective SFT for its current deterministic contract.
2. Preserve routing as the learned decision surface, but do not start DPO/RLVR:
   the frozen policy does not beat the static prior and the hybrid has no
   decisive coverage.
3. Build the first C5 antibody-antigen OOD manifest with typed calibration,
   metric-scope, regime-match, baseline, hidden-interface-label, and expected
   action fields.
4. Compare `trust_all`, general gate, regime-specific gate, and fail-closed
   behavior before any C5 model training.

Raw generations, candidate scores, trainable states, scheduler logs, private
manifests, and completed sealed rows remain uncommitted.

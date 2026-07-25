# Project Status

Last updated: 2026-07-25

## Current Thesis

Biology agents should be evaluated on executable
`tool -> evidence packet -> terminal action` trajectories. Learned policies
remain useful where evidence-conditioned choices are genuinely uncertain, but
fixed schema transformations and safety invariants belong in fail-closed
runtime code. Specialist confidence earns trust only through regime-matched
calibration. DPO/RLVR and Hugging Face publication remain gated by broader,
independent evidence.

## Current Result

The prospective Stage B C5 method, source panel, and AlphaFold 3 input set are
now frozen before prediction or label reveal.

- The protocol locks SAbDab2 v0.1.0, its official antibody-and-antigen
  sequence-aware split, AlphaFold 3 v3.0.3, DockQ v2.1.3, a 50-threshold
  ranking-score grid, primary `alpha = 0.30`, `delta = 0.10`, overlap
  exclusions, sample selection, and stopping rules. Its canonical SHA-256 is
  `9c3fd6784fecef3b8971daedb8bfbfc3a1ca725f0353e05f0b7420a30f06e17a`.
- Exact source validation passes for 15,641 rows and 48 columns. Metadata-only
  filtering retains 2,417 eligible rows: 1,997 train and 420 test.
- Deterministic selection produces 80 calibration, 20 calibration-reserve, 40
  evaluation, and 10 evaluation-reserve targets. The 150 rows have unique PDB
  and SAbDab IDs, zero overlap with all prior Fromm/Gray PDBs, and zero
  source-cluster overlap between calibration and evaluation.
- Cayuga structure QC passes 150/150 with no reserve promotion. The retained
  input freeze contains 80 calibration and 40 evaluation targets, with 120/120
  template-free AF3 JSON inputs and a set checksum of
  `3569fc8641613c5328a05c991942a576f9ba1f9ad24daf135ea2b62806a52b18`.
- A fail-closed AF3 environment preflight and Cayuga 120-task array path now
  enforce source commit/tag, container checksum, official parameter-set
  checksum, database inventory checksum, input count/checksum, clean output
  boundary, and immutable private attestation.
- The first real Cayuga preflight passes runtime, AF3 source, input-set, and
  output-boundary checks. It correctly blocks prediction because the pinned
  container, official model parameters, and database manifest are not yet
  installed.
- For zero observed failures, this design needs at least 35 trusted calibration
  targets at `alpha = 0.30`, 78 at 0.20, and 311 at 0.10. The 80-target
  calibration slice cannot certify 0.10 under the preregistered bound.
- No sequence, structure, DockQ value, interface label, local path, parameter,
  raw prediction, or scheduler log is published. No model training, DPO, RLVR,
  or external specialist trust is enabled.

This is prospective method/source/input evidence, not a new AF3 performance
result, a blinded hidden test, or a trust claim.

The previous Stage A result remains unchanged: frozen routing is 35/180 versus
an 80/180 static prior, the runtime hybrid is 115/180 with zero unsafe
grounding and zero decisive coverage, and the tool-query compiler is 25/25
clean with 150/150 malformed inputs rejected.

The completed private sealed set was not read, regenerated, rescored, or used
for prompt selection. Its commitment remains
`post_training/stage_a_sealed_extension_commitment_2026-07-10.json` and its
one-time routing result remains 5/25. Do not tune on or rescore these 25 sealed
rows.

## Source Changes

SAbDab2 Machine Learning Dataset v0.1.0 adds a checksum-locked
antibody-antigen source with an official sequence-aware split and a
`CC-BY-4.0` data boundary. AlphaFold 3 v3.0.3 fixes the prediction code at
commit `7b197fe859790fc3e04d03ea70dd0b9ba48881c9`; its official installation
guide makes model-parameter authorization, database installation, container
identity, and A100/H100-class compute explicit dependencies. DockQ v2.1.3
fixes the evaluator at commit
`d9cbb1940bb0f42db3257f7da3b0e96f162b94d9`.

These sources change source selection, leakage control, execution preflight,
and label reproducibility, so they are logged in
`research/2026-06-25_posttrain_tool_use_landscape/SOURCE_LOG.md`. The prior
Fromm, Hitawala-Gray, FoldBench, and RCPS conclusions remain unchanged.

## Next Decision

Proceed with `stage_b_c5_af3_environment_attestation_and_prediction`.

1. Obtain the official AF3 3.0.x parameters through the authorized access
   process; do not substitute unofficial or untracked weights.
2. Build the v3.0.3 container, install the official databases, create their
   private checksum inventory, and rerun the Cayuga preflight.
3. Submit the 120-target array only after the private attestation has zero
   violations and its SHA-256 is frozen.
4. Freeze five prediction outputs per retained target and the preregistered
   target-level selection before any DockQ calculation.
5. Reveal calibration labels first, freeze the selected threshold or
   `verify_all`, and only then reveal evaluation labels.
6. Keep model training, DPO, RLVR, evaluation-threshold tuning, and external
   specialist trust closed.

Raw structures, sequences, AF3 inputs, parameters, databases, predictions,
attestations, candidate scores, and scheduler logs remain uncommitted.

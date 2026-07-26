# Project Status

Last updated: 2026-07-26

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
- The pinned AF3 v3.0.3 Apptainer build now passes on a Cayuga CPU node. The
  3,699,896,320-byte SIF is frozen at SHA-256
  `128a62b4849f3606a61a12fbe754e3f928bdbe43fe1c0894a231380f419fe7b2`;
  embedded source/package tests, runner import, sidecar checksum replay, and a
  one-device JAX GPU smoke all pass. The path-free compact result is
  `c5_af3_container_readiness_2026-07-25.json`. This closes the container
  blocker only; no parameters, databases, predictions, or labels are included.
- The official AF3 database fetch has now completed and atomically promoted
  after validation. Its private inventory binds all 9 required entries and
  195,867 files (672,435,030,513 bytes) by per-file content SHA-256; sidecar
  replay passes and no partial staging remains. The path-free projection is
  `c5_af3_database_readiness_2026-07-26.json`. No filename, database content,
  private path, or scheduler identifier is published.
- Authorized-parameter intake is now fail-closed and ready before any weight is
  present. The Cayuga provisioning job requires an explicit assertion that the
  files were received directly from Google, rejects symlinks and multiple model
  families, copies only recognized fragments into private staging, content
  hashes the staged family, and promotes it atomically with a private manifest.
  The authorization assertion records provenance intent; it is not independent
  license verification.
- The private runtime attestation contract is now versioned as
  `c5_af3_environment_attestation_v2`. A dedicated Cayuga CPU job binds a clean
  benchmark commit, clean pinned AF3 source, container content, authorized
  model inventory plus its user-asserted provenance manifest, complete database
  inventory, frozen input set, and clean output boundary. It performs the
  expensive content scan once, immediately rebinds the mounted runtime in quick
  mode, and promotes the checksum sidecar before the attestation completion
  marker. Missing benchmark identity, manifest authorization, or any required
  component fails closed.
- Cayuga access testing exposed a host-runtime mismatch: the login Python is
  too old for this package, and an attestation checksum alone did not bind the
  paths mounted by each array task. The array now runs verification inside the
  pinned AF3 image using its `/app/alphafold` working directory and
  `uv run --no-sync python3`, requires the private database manifest, and rechecks
  container/model/database identities against the attestation before
  inference. Full mode rehashes container, parameter, and database content;
  quick per-task mode uses content-attested sizes and nanosecond mtimes, a
  model identity digest, and deterministic database sentinels.
- The post-run phase gates are implemented and synthetic-tested before any
  prediction exists. The intake requires the exact AF3 v3.0.3 five-sample
  directory/file contract, cross-checks full-precision
  `ranking_scores.csv` against summary confidence JSON, applies the frozen
  score/ipTM/lexical tie rule, and checksum-locks all 600 samples plus the 120
  selected models.
- A private native-structure lock reconstructs the 150 target-specific hashes
  behind the pre-label aggregate commitment. Calibration labels must match the
  selected model, native structure, evaluator commit/version, metric scope,
  and committed chain mapping.
- Calibration and evaluation reveal commands enforce exact 80/40 target sets.
  They freeze the finite-grid Hoeffding decision before evaluation, force
  `verify_all` when no certificate exists, and reject evaluation-time
  threshold changes. Synthetic tests exercise both certified and
  uncertified paths; these are contract tests, not scientific results.
- The official v3.0.3 split-execution contract is now implemented. A CPU array
  runs data-pipeline-only jobs into private staging and promotes only an exact,
  label-free `<job>_data.json`; a GPU array runs inference-only with
  `--force_output_dir=true`, validates the canonical five-sample target output,
  and replaces the processed target only after validation. The combined array
  remains a fallback, and the final prediction intake is unchanged.
- The initial Cayuga preflight historically blocked on the container,
  parameters, and databases. The container and official database inventory are
  now checksum-frozen; authorized official model parameters are the sole
  unresolved runtime dependency.
- For zero observed failures, this design needs at least 35 trusted calibration
  targets at `alpha = 0.30`, 78 at 0.20, and 311 at 0.10. The 80-target
  calibration slice cannot certify 0.10 under the preregistered bound.
- No sequence, structure, DockQ value, interface label, local path, parameter,
  raw prediction, or scheduler log is published. No model training, DPO, RLVR,
  or external specialist trust is enabled.

The tracked preregistration therefore remains
`panel_locked_prediction_pending`. This is prospective
method/source/input/phase-gate evidence, not a new AF3 performance result, a
blinded hidden test, a completed label reveal, or a trust claim.

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
identity, and A100/H100-class compute explicit dependencies. Its pinned
Dockerfile fixes `/app/alphafold` as the working directory and invokes
`uv run python3`; the output writer fixes the sanitized job directory, five
sample subdirectories, `ranking_scores.csv`, and summary-confidence schemas.
The performance guide explicitly supports CPU data-pipeline-only output,
inference-only reuse of processed JSON, and same-directory continuation via
`--force_output_dir`.
The official ranking score is not restricted to `[0,1]` because disorder and
clash terms are included. DockQ v2.1.3 fixes the evaluator at commit
`d9cbb1940bb0f42db3257f7da3b0e96f162b94d9`.

These sources change source selection, leakage control, execution preflight,
prediction intake, score validation, and label reproducibility, so they are
logged in
`research/2026-06-25_posttrain_tool_use_landscape/SOURCE_LOG.md`. The prior
Fromm, Hitawala-Gray, FoldBench, and RCPS conclusions remain unchanged.

## Next Decision

Proceed with `stage_b_c5_af3_environment_attestation_and_prediction`.

1. Obtain the official AF3 3.0.x parameters through the authorized access
   process linked from the pinned installation guide; do not substitute
   unofficial or untracked weights.
2. Run `attest_c5_af3_runtime_cayuga.sbatch` from a clean benchmark checkout
   against the checksum-locked SIF, authorized parameter manifest, and
   completed database inventory.
3. Submit the 120-target CPU data-pipeline array followed by the GPU inference
   array only after the v2 private attestation has zero violations and its
   SHA-256 sidecar is frozen. Keep the per-task quick runtime binding enabled;
   retain the combined array as fallback.
4. Run `prospective_predictions.py` to freeze five outputs per retained target
   and the preregistered target-level selection before any DockQ calculation.
5. Reconstruct the private native-structure lock against the existing
   150-structure aggregate commitment.
6. Run `prospective_reveal.py calibrate` on exactly 80 calibration labels,
   freeze the selected threshold or `verify_all`, and only then run
   `prospective_reveal.py evaluate` on the 40 evaluation labels once.
7. Keep model training, DPO, RLVR, evaluation-threshold tuning, and external
   specialist trust closed.

Raw structures, sequences, AF3 inputs, parameters, databases, predictions,
attestations, candidate scores, and scheduler logs remain uncommitted.

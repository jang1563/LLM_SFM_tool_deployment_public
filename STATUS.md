# Project Status

Last updated: 2026-07-25

## Current Thesis

Biology agents should be evaluated on executable
`tool -> evidence packet -> terminal action` trajectories. Learned policies
remain useful where evidence-conditioned choices are genuinely uncertain, but
fixed schema transformations and safety invariants belong in fail-closed
runtime code. DPO/RLVR and Hugging Face publication remain gated by broader,
independent evidence.

## Current Result

The first Stage B C5 source-backed public-score pilot is complete.

- A commit-pinned AlphaFold 3 table from Fromm et al. passed exact archive,
  checksum, schema, and shape validation: 22,000 samples, 110 Ab-Ag targets,
  and 200 samples per target.
- The adapter uses 9 of 56 columns. It excludes 47 columns and 132,000 absolute
  compute-path cells; raw paths, structures, sequences, and unhashed sample IDs
  are not present in the public artifacts.
- One sample per target is selected by maximum ranking confidence with a
  deterministic tie break. Sixty-four of 110 targets have a top-score tie.
- A frozen SHA-256 target split produces 55 calibration and 55 evaluation
  targets with zero target overlap. The selected sample succeeds on 55/110
  targets by `DockQ >= 0.23`.
- On frozen evaluation, `trust_all` trusts 55/55 with 28 failures. A fixed
  `ipTM >= 0.80` baseline trusts 20/55 with 3 failures. This fixed threshold is
  not a calibrated general-PPI transfer gate.
- A uniform Hoeffding/union-bound search over 50 thresholds certifies no trusted
  set at `alpha = 0.30`, `0.20`, or `0.10`. The correct primary action is
  therefore fail-closed verification: 0/55 trusted and 55/55 routed to verify.
- The 110 derived rows reuse `TaskSpec`, `Trajectory`, `EvidencePacket`, and
  `TrajectoryEvaluator`; hidden interface labels remain outside
  `model_visible_task`.
- This is a published-label replay, not an independent hidden test or a new
  structure-prediction result.

The previous prospective Stage A checkpoint remains unchanged: frozen routing
is 35/180 versus an 80/180 static prior, the runtime hybrid is 115/180 with
zero unsafe grounding and zero decisive coverage, and the tool-query compiler
is 25/25 clean with 150/150 malformed inputs rejected.

The completed private sealed set was not read, regenerated, rescored, or used
for prompt selection. Its commitment remains
`post_training/stage_a_sealed_extension_commitment_2026-07-10.json` and its
previous one-time routing result remains 5/25. Do not tune on or rescore these
25 sealed rows.

## Source Changes

The Fromm et al. source member was verified byte-identical inside Zenodo record
`17978681`, whose release is `CC-BY-4.0`. The derived-data attribution and
transformation boundary are recorded in
`c5_antibody_ood/SOURCE_BACKED_PILOT_PROVENANCE.md`.

FoldBench's paper and MIT Zenodo/GitHub release establish 279 PPI and 172 Ab-Ag
targets under a common DockQ success definition. Its public repository and
Zenodo v1.0 archive expose target tables, evaluator code, and examples. The
article Source Data workbook exposes target-level PPI/Ab-Ag DockQ but does not
pair those strata with per-sample ranking/confidence fields. A validated full
per-sample transfer table is therefore unavailable for the current adapter, so
general-PPI-to-Ab-Ag calibration transfer remains deferred.

## Next Decision

Proceed with `stage_b_c5_independent_calibration_evidence`.

1. Keep the synthetic 12-row fixture as a contract positive control and the
   110-row source-backed manifest as a published-label replay.
2. Do not tune threshold grids on the 55 frozen evaluation targets.
3. Seek a license-compatible per-sample PPI/Ab-Ag confidence table, author-
   supplied FoldBench score export, or an independently defined Ab-Ag panel.
4. Pre-register any new threshold family and certification correction before
   reading new labels; preserve target-level grouping.
5. Run Cayuga/Expanse structure prediction only if compatible saved public
   scores cannot supply the missing evidence. Model training, DPO, and RLVR
   remain closed.

Raw generations, candidate scores, trainable states, scheduler logs, private
manifests, and completed sealed rows remain uncommitted.

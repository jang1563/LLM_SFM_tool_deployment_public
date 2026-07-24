# Project Status

Last updated: 2026-07-24

## Current Thesis

Biology agents should be evaluated on executable
`tool -> evidence packet -> terminal action` trajectories. Learned policies
remain useful where evidence-conditioned choices are genuinely uncertain, but
fixed schema transformations and safety invariants belong in fail-closed
runtime code. DPO/RLVR and Hugging Face publication remain gated by broader,
independent evidence.

## Current Result

The Stage B C5 no-API policy prototype is complete.

- Added 12 synthetic public policy-test records, balanced at three expected
  `trust`, `baseline`, `verify`, and `defer` actions.
- Each row includes a visible complex ID, chain-role mapping, typed specialist
  metric, general and Ab-Ag calibration cards, baseline result, and verifier
  availability. Interface labels and expected actions remain evaluator-only.
- Reused `TaskSpec`, `Trajectory`, `EvidencePacket`, and
  `TrajectoryEvaluator`; no parallel schema family was introduced.
- Oracle trajectories pass 12/12. Missing metric scope, calibration dataset,
  RCPS threshold, or regime match cannot produce a fail-closed trust action.
- `trust_all` passes 3/12 with 9 unsafe trusts. The general gate passes 3/12
  with 8 unsafe trusts. The regime-specific certifier passes 6/12 with zero
  unsafe trust but does not choose operational fallbacks. The fail-closed
  router passes 12/12 with zero unsafe trust.
- These outcomes are defined by a synthetic policy fixture. They validate the
  contract and test harness, not Ab-Ag calibration transfer or model quality.

The previous prospective Stage A checkpoint remains unchanged: frozen routing
is 35/180 versus an 80/180 static prior, the runtime hybrid is 115/180 with
zero unsafe grounding and zero decisive coverage, and the tool-query compiler
is 25/25 clean with 150/150 malformed inputs rejected.

The completed private sealed set was not reused. Its commitment remains
`post_training/stage_a_sealed_extension_commitment_2026-07-10.json` and its
one-time routing result remains 5/25. Do not tune on or rescore these 25 sealed
rows.

The completed private sealed set was not read, regenerated, rescored, or used
for prompt selection. Its commitment remains
`post_training/stage_a_sealed_extension_commitment_2026-07-10.json` and its
previous one-time routing result remains 5/25. Do not tune on or rescore these
25 sealed rows.

## Source Changes

The 2026 Bioinformatics study
[Evaluating deep learning based structure prediction methods on antibody-antigen complexes](https://doi.org/10.1093/bioinformatics/btag136)
provides a 110-complex unseen Ab-Ag benchmark, public score/code artifacts, and
direct evidence that internal confidence often fails to select the best sampled
model. [FoldBench](https://www.nature.com/articles/s41467-025-67127-3) adds
low-homology general PPI and Ab-Ag strata under a common DockQ-style evaluation.
Together they change the next ticket from new heavy prediction runs to a
public-score intake and grouped calibration-transfer pilot.

## Next Decision

Proceed with `stage_b_c5_source_backed_public_score_pilot`.

1. Keep the 12 synthetic rows as contract tests only; do not report their
   policy scores as biological performance.
2. Audit the Fromm et al. and FoldBench public score schemas, licenses, model
   cutoffs, target IDs, chain mappings, and metric definitions before intake.
3. Split by `complex_id`, never by sampled model, so predictions from one
   target cannot cross calibration and evaluation partitions.
4. Keep DockQ/interface success hidden. Fit any threshold on calibration
   targets only, then compare general-PPI transfer, Ab-Ag-specific calibration,
   trust-all, and fail-closed routing on frozen evaluation targets.
5. Do not run model training or new structure prediction until public-score
   intake, leakage checks, and deterministic calibration baselines pass.

Raw generations, candidate scores, trainable states, scheduler logs, private
manifests, and completed sealed rows remain uncommitted.

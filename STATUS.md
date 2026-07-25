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

The first independent-source Stage B C5 calibration replay is complete.

- The pinned Hitawala-Gray AF3 table passes checksum, schema, and shape
  validation: 1,900 rows and 130 source targets. The adapter retains 1,565
  complete bound predictions across 108 targets and excludes 335 unbound rows
  without antibody-antigen DockQ labels.
- Target selection follows the source protocol: maximum AF3 ranking score,
  maximum heavy-antigen ipTM for score ties, then deterministic lexical sample
  ID. Raw filenames are salted and hashed before export.
- All 110 Fromm PDB IDs are blocked before calibration. Nine overlapping PDB
  IDs representing 11 Gray complex copies are removed, leaving 97
  independent-source targets: 44 antibodies and 53 nanobodies.
- At `ranking_score >= 0.80`, the Gray antibody slice trusts 17/44 with 5
  failures; the nanobody slice trusts 10/53 with 2 failures. Trust-all failure
  rates are 25/44 and 34/53, respectively.
- The pre-existing 0.50-0.99 finite grid with uniform
  Hoeffding/union-bound correction certifies no antibody or nanobody trusted
  set at `alpha = 0.30`, `0.20`, or `0.10`. The closest primary upper bounds
  are 0.644 for antibody and 0.638 for nanobody, both above 0.30.
- Because the independent antibody certificate fails, the locked 55-target
  Fromm evaluation remains fail closed: 0/55 trusted and 55/55 routed to
  verify. No new threshold was tuned on those frozen evaluation labels.
- The 97 derived rows reuse `TaskSpec`, `Trajectory`, `EvidencePacket`, and
  `TrajectoryEvaluator`; hidden interface labels remain outside
  `model_visible_task`.
- This is independent-source published-label evidence, not a blinded hidden
  test or a new structure-prediction result.

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

Hitawala and Gray's peer-reviewed article, Zenodo version 4 dataset
(`10.5281/zenodo.16426003`, `CC-BY-4.0`), and MIT repository add an independent
post-cutoff Ab-Ag/Nb-Ag panel with sample-level AF3 ranking score, ipTM-HA, and
DockQ. The repository input is commit pinned and SHA-256 locked. The archive
was inspected by byte range to verify confidence-score files and standardized
`H`, `L`, `A` chain roles. Attribution, overlap exclusion, transformation, and
privacy boundaries are recorded in
`c5_antibody_ood/INDEPENDENT_CALIBRATION_PROVENANCE.md`.

The prior Fromm and FoldBench source conclusions remain unchanged. FoldBench's
public target-level workbook still lacks a validated paired per-sample
confidence/DockQ table for general-PPI-to-Ab-Ag transfer.

## Next Decision

Proceed with `stage_b_c5_prospective_panel_preregistration`.

1. Keep the synthetic 12-row fixture, 110-row Fromm replay, and 97-row Gray
   calibration manifest as distinct evidence layers.
2. Do not retune the threshold grid or inspect the frozen 55-target Fromm
   evaluation for method selection.
3. Seek one larger non-overlapping antibody-only panel or a paired FoldBench
   per-sample confidence export before spending cluster compute.
4. If public saved scores remain insufficient, pre-register target inclusion,
   sample count, AF3 version, ranking metric, DockQ evaluator, overlap policy,
   risk correction, and stopping rule for a small Cayuga panel before running
   predictions.
5. Keep model training, DPO, RLVR, and external specialist trust closed until a
   regime-matched certificate passes and the locked replay remains within its
   risk bound.

Raw generations, candidate scores, trainable states, scheduler logs, private
manifests, and completed sealed rows remain uncommitted.

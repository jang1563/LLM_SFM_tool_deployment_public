# C5 Antibody-Antigen OOD Plan

## Purpose

Test the boundary condition for Phase 4 enforcement-based trust routing.

Phase 4a C1-C4 shows that a deterministic calibrated-risk gate can beat
free-form LLM routing on an in-distribution hard-complex substrate and resist
misleading reliability framing. C5 asks the harder deployment question:

> What happens when the specialist confidence signal enters a known OOD regime,
> specifically antibody-antigen complexes where structure-predictor confidence
> can be regime-dependently miscalibrated?

The correct outcome is not necessarily "gate still wins." A good deployment
gate should either certify a trusted set under regime-specific calibration or
fail closed into verify/defer.

## Current Anchors

- Phase 4a result:
  `<local-workspace>/LLM_SFM_phase4_planning/PHASE4A_CORE_RESULT_2026-06-25.md`
- Phase 4 positioning:
  `<local-workspace>/LLM_SFM_phase4_planning/POSITIONING_BRIEF_2026-06-25.md`
- Application repo:
  `<local-workspace>/bio_sfm_designer`
- Reusable trust engine:
  `<local-workspace>/bio-sfm-trust-core`

## 2026-07-24 Prototype Checkpoint

The first public no-API contract prototype is complete in this repository:

```bash
python -m c5_antibody_ood.manifest \
  --out c5_antibody_ood/c5_policy_test_manifest_v1.jsonl
python -m c5_antibody_ood.evaluate_baselines
```

- 12 synthetic policy-test rows are balanced across trust, baseline, verify,
  and defer.
- Hidden interface labels do not enter model-visible tasks or trajectories.
- `trust_all` passes 3/12 with 9 unsafe trusts.
- The generic threshold gate passes 3/12 with 8 unsafe trusts.
- The regime-specific certifier passes 6/12 with zero unsafe trust.
- The operational fail-closed router passes 12/12 with zero unsafe trust.

These are fixture-defined contract results, not Ab-Ag performance estimates.

## 2026-07-25 Source-Backed Checkpoint

The first public-score pilot is complete without API calls, model loading, or
new structure prediction:

```bash
python -m c5_antibody_ood.evaluate_source_pilot \
  --scores /external/path/scores_alphafold3.csv
```

- The canonical source is the exact 24,588,543-byte AF3 CSV in Fromm et al.
  Zenodo record `17978681` (`CC-BY-4.0`), verified by SHA-256.
- Intake validates 22,000 rows, 110 targets, 200 samples per target, required
  columns, unit-interval metrics, sample uniqueness, and the `alphafold3`
  preset.
- A nine-column allowlist excludes source paths, structures, sequences, raw
  features, and unhashed source sample IDs.
- Maximum ranking confidence selects one sample per target; lexical sample ID
  resolves ties. Sixty-four targets have tied top scores.
- SHA-256 target grouping freezes 55 calibration and 55 evaluation targets
  with zero overlap.
- On evaluation, trust-all has 28 failures among 55 trusted targets. A fixed
  `ipTM >= 0.80` baseline has 3 failures among 20 trusted targets.
- A 50-threshold uniform Hoeffding search certifies no trusted set at
  `alpha = 0.30`, `0.20`, or `0.10`; fail-closed verification is the correct
  current policy.

This is a published-label replay. The source-backed intake is valid, but
regime-specific trust is not certified.

## 2026-07-25 Independent Calibration Checkpoint

The Hitawala-Gray published-score replay adds a second source:

- 1,565 complete bound predictions across 108 source targets pass intake;
- all Fromm PDB IDs are blocked before calibration;
- 9 overlapping PDB IDs representing 11 complex copies are excluded;
- 44 antibody and 53 nanobody targets remain;
- neither cohort earns a uniformly corrected ranking-score certificate at
  `alpha <= 0.30`;
- the locked Fromm evaluation therefore remains 0/55 trusted and 55/55
  verified.

This is independent-source published-label evidence, not a hidden test.

## 2026-07-25 Prospective Freeze Checkpoint

The next C5 evidence layer is now preregistered before prediction:

- protocol SHA:
  `9c3fd6784fecef3b8971daedb8bfbfc3a1ca725f0353e05f0b7420a30f06e17a`;
- exact SAbDab2 v0.1.0 source and official sequence-aware split;
- 80 calibration, 20 calibration-reserve, 40 evaluation, and 10
  evaluation-reserve targets;
- 150 unique PDB/SAbDab IDs, zero prior C5 PDB overlap, and zero
  source-cluster overlap between calibration and evaluation;
- Cayuga structure QC 150/150, no promotions, and 120 frozen template-free AF3
  inputs;
- AF3 v3.0.3 and DockQ v2.1.3 commit pins;
- fixed threshold family, risk correction, label scope, output-selection rule,
  and stopping rule.

The Cayuga preflight passes the runtime, source, and 120-input checks but
blocks prediction until the container, authorized AF3 parameters, and database
inventory are checksum locked. See
`c5_antibody_ood/C5_PROSPECTIVE_PANEL_PREREGISTRATION_2026-07-25.md`.

## Hypotheses

### H4.4 Boundary

A general-fit calibrated gate should degrade or refuse to certify when applied
to antibody-antigen complexes if its calibration was learned on a different
complex regime.

### H4.5 Fail-Closed Value

If the OOD regime lacks a certifiable trust threshold, the enforcement layer
should make that visible and route to verify/defer instead of letting the LLM
trust a persuasive reliability card.

### H4.6 Regime-Specific Recovery

If enough antibody-antigen calibration records are available, a regime-specific
calibrator or RCPS threshold may recover a safe trusted subset. This is stronger
than the generic claim "gate wins"; it says where trust has earned calibration.

## Dataset Build

The panel is no longer ad hoc. `prospective_source.py` validates the exact
SAbDab2 split file, projects metadata through a strict allowlist, excludes all
prior Fromm/Gray PDB IDs, and deterministically selects the locked primary and
reserve roles. `prospective_inputs.py` performs private native-structure and
chain-sequence QC, promotes only same-split reserves, and emits hash
commitments rather than raw sequence or structure content.

Public panel validation fails closed on:

- source checksum/schema/count drift;
- PDB or SAbDab duplication;
- prior C5 PDB overlap;
- calibration/evaluation source-cluster overlap;
- missing chain-role or source-split fields;
- leaked sequences, DockQ labels, confidence values, or local paths.

## Compute Pattern

Use the pinned prospective path:

```text
Cayuga AF3 array -> immutable prediction freeze -> calibration label reveal
-> threshold or verify-all freeze -> evaluation label reveal
```

Before submission, run `c5_antibody_ood.af3_preflight` against the private
environment. It verifies:

- AF3 source commit and tag;
- Singularity image checksum;
- one official parameter-set checksum;
- all required databases plus a checksum-locked private inventory;
- the 120-file input-set checksum;
- a clean output boundary.

The 120-task Cayuga array verifies the passed private attestation SHA before
starting. Expanse remains fallback only. No local heavy model compute is
allowed.

## Evaluation

After the prediction and label-reveal gates pass, compare:

| condition | meaning |
| --- | --- |
| `C1_free_form_llm` | LLM chooses from a reliability card, no binding gate |
| `C2_general_gate` | deterministic gate using prior/general calibration |
| `C5_regime_gate` | deterministic gate recalibrated on antibody-antigen split |
| `C5_fail_closed` | no trusted set if RCPS cannot certify alpha |

Only run LLM/API arms after the no-API gate and QC checks are complete.

## Decision Criteria

- `signal_validity`: pAE/ipTM-derived risk must be assessed against interface
  success labels, not assumed.
- `gate_transfer`: general gate must beat trust-all and shuffled/inverted
  controls before being treated as deployable in Ab-Ag.
- `certification`: if RCPS returns no threshold, the correct deployment behavior
  is no trusted set in that regime.
- `LLM comparison`: if an LLM appears better, require the same leakage,
  cue-manipulation, and cost controls as Phase 4a before calling it recovery.

## Immediate Next Research Checks

The method, source, and input gates have passed. The next work is execution and
staged reveal, not model learning:

1. Obtain the authorized official AF3 3.0.x parameters.
2. Build and checksum the v3.0.3 container.
3. Install the official databases and freeze a private inventory checksum.
4. Rerun preflight; submit no GPU task unless every component passes.
5. Freeze all five outputs per retained target and select one output by the
   preregistered rule before DockQ calculation.
6. Reveal calibration labels, freeze the certificate or `verify_all`, then
   reveal evaluation labels once.

No local heavy model compute, evaluation tuning, DPO/RLVR, or model training is
planned. General-PPI transfer remains a separate evidence gap.

## Sanity Check Result

Completed 2026-06-25:

- `bio-sfm-trust-core`: `PYTHONPATH=src python3 -m unittest discover -s tests -v`
  -> `Ran 32 tests ... OK`.
- `bio_sfm_designer`: `PYTHONPATH=src:<local-workspace>/bio-sfm-trust-core/src python3 -m unittest discover -s tests -v`
  -> `Ran 133 tests ... OK`.
- Template manifest check:
  `targets=3 ready=3 ok=True`.

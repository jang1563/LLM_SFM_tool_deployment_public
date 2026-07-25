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

1. Curate a small antibody-antigen complex panel.
   - For each target: source PDB, antigen chain, antibody/binder chain, notes on
     missing residues and interface contacts.
   - Start with at least 3 targets for manifest validation; scale only after QC.
2. Prepare clean two-chain PDBs with:

```bash
python hpc/prep_hetdimer.py \
  --pdb /path/to/source.pdb \
  --target-chain A \
  --binder-chain H \
  --out hpc_outputs/targets/prepared_TARGET_AH.pdb \
  --report hpc_outputs/targets/prepared_TARGET_AH.report.json
```

3. Build a manifest based on:

```text
<local-workspace>/bio_sfm_designer/configs/template_complex_targets.json
```

4. Validate before compute:

```bash
python -m bio_sfm_designer.experiments.complex_target_manifest \
  --manifest configs/c5_antibody_targets.json \
  --require-files \
  --min-targets 3 \
  --out results/c5_antibody_manifest.json \
  --emit-plan results/c5_antibody_submit.sh
```

## Compute Pattern

Use the existing HPC-first pattern:

```text
Cayuga/Expanse GPU job -> JSONL records -> local Precomputed adapter -> local gate/eval
```

For each ready target, the generated submit plan should call:

```bash
sbatch hpc/run_generate_proteinmpnn_complex.sbatch
sbatch hpc/run_predict_boltz_complex.sbatch
```

For a pure OOD calibration check, also allow a no-redesign/native-complex panel
if the prediction script can emit the same record schema. Do not spend scale-up
compute before verifying the records pass QC and the label definition is stable.

## Evaluation

Run the existing posthoc bundle and alpha planner first:

```bash
python -m bio_sfm_designer.experiments.complex_posthoc_bundle \
  --records hpc_outputs/predict/records_boltz_complex_antibody.jsonl \
  --alphas 0.3,0.2,0.1 \
  --out-dir results/c5_antibody_posthoc

python -m bio_sfm_designer.experiments.complex_alpha_plan \
  --records hpc_outputs/predict/records_boltz_complex_antibody.jsonl \
  --alphas 0.3,0.2,0.1
```

Then compare four routing conditions:

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

The public-score intake gate has passed. The next missing evidence is
independent calibration, not model learning:

1. Do not inspect or tune on the frozen 55-target evaluation labels.
2. Request or locate a license-compatible FoldBench export containing
   per-sample target ID, interaction regime, confidence metric, ranking score,
   and DockQ for both PPI and Ab-Ag.
3. If compatible scores remain unavailable, pre-register an independent
   Ab-Ag panel and run only the missing specialist outputs on Cayuga first,
   Expanse second.
4. Freeze threshold candidates, risk definition, alpha, delta, and
   multiple-threshold correction before reading the new labels.
5. Compare general-PPI transfer, Ab-Ag-specific calibration, trust-all,
   shuffled/inverted controls, and fail-closed routing on target-grouped data.

No local heavy model compute is planned. DPO/RLVR is unrelated to the current
calibration-data gap and remains closed.

## Sanity Check Result

Completed 2026-06-25:

- `bio-sfm-trust-core`: `PYTHONPATH=src python3 -m unittest discover -s tests -v`
  -> `Ran 32 tests ... OK`.
- `bio_sfm_designer`: `PYTHONPATH=src:<local-workspace>/bio-sfm-trust-core/src python3 -m unittest discover -s tests -v`
  -> `Ran 133 tests ... OK`.
- Template manifest check:
  `targets=3 ready=3 ok=True`.

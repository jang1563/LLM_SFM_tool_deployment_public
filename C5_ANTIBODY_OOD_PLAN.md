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

## Immediate Next Local Checks

Run these without API or HPC spend:

```bash
cd <local-workspace>/bio-sfm-trust-core
PYTHONPATH=src python3 -m unittest discover -s tests -v

cd <local-workspace>/bio_sfm_designer
PYTHONPATH=src:<local-workspace>/bio-sfm-trust-core/src \
  python3 -m unittest discover -s tests -v

PYTHONPATH=src:<local-workspace>/bio-sfm-trust-core/src \
  python3 -m bio_sfm_designer.experiments.complex_target_manifest \
  --manifest configs/template_complex_targets.json \
  --min-targets 1 \
  --out /tmp/template_complex_targets_check.json
```

If these pass, the next substantive work is curating
`configs/c5_antibody_targets.json` in `bio_sfm_designer`, not writing a parallel
engine in this folder.

## Sanity Check Result

Completed 2026-06-25:

- `bio-sfm-trust-core`: `PYTHONPATH=src python3 -m unittest discover -s tests -v`
  -> `Ran 32 tests ... OK`.
- `bio_sfm_designer`: `PYTHONPATH=src:<local-workspace>/bio-sfm-trust-core/src python3 -m unittest discover -s tests -v`
  -> `Ran 133 tests ... OK`.
- Template manifest check:
  `targets=3 ready=3 ok=True`.


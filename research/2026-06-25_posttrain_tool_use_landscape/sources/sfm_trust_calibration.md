# SFM Trust / Calibration Source Cards

Purpose: collect sources where a specialist foundation model emits a confidence,
score, embedding, or prediction that an LLM-orchestrator might be tempted to
trust. These cards support the local `trust | verify | baseline | defer` action
schema.

## AlphaFold Confidence Metrics

- Type: `official`
- URLs:
  - https://www.ebi.ac.uk/training/online/courses/alphafold/inputs-and-outputs/evaluating-alphafolds-predicted-structures-using-confidence-scores/pae-a-measure-of-global-confidence-in-alphafold-predictions/
  - https://www.ebi.ac.uk/training/online/courses/alphafold/inputs-and-outputs/evaluating-alphafolds-predicted-structures-using-confidence-scores/confidence-scores-in-alphafold-multimer/
- Domain: protein structure prediction
- Verifier/reward signal: pLDDT, PAE, pTM, and ipTM; multimers require joint
  interpretation of local, pairwise, global, and interface confidence.
- Project relevance: direct schema source for SFM evidence packets:
  `specialist_confidence` must be typed by metric and interpretation regime,
  not treated as a generic trust scalar.
- Claim boundary: confidence metrics are useful model self-assessments, not
  experimental truth or deployment certification.

## AlphaFold 3 / Biomolecular Interaction Modeling

- Type: `paper`
- URL: https://www.nature.com/articles/s41586-024-07487-w
- Domain: general biomolecular interaction prediction
- Verifier/reward signal: confidence-ranked samples and benchmarked structural
  accuracy across proteins, nucleic acids, ligands, ions, modifications, and
  antibody-antigen interfaces.
- Project relevance: establishes why SFM routing is now product-relevant:
  general biomolecular models can produce high-value specialist outputs, but
  those outputs still need interaction-type-specific validation.
- Claim boundary: higher benchmark accuracy does not remove the need for
  regime-specific calibration and assay/database verification.

## AlphaFold Antibody-Antigen Modeling Evaluations

- Type: `paper`
- URLs:
  - https://pmc.ncbi.nlm.nih.gov/articles/PMC10751731/
  - https://onlinelibrary.wiley.com/doi/10.1002/pro.4865
- Domain: antibody-antigen structure prediction
- Verifier/reward signal: CAPRI/DockQ-style structural accuracy over hundreds of
  antibody-antigen complexes; comparison of global and interface confidence
  metrics.
- Project relevance: direct external support for local C5. Antibody-antigen is
  a specific OOD regime where a general structure-confidence gate should not be
  assumed valid without regime-specific calibration.
- Claim boundary: exact performance numbers should be rechecked from the paper
  before publication; use here as a regime-boundary source.

## 2026 Unseen Antibody-Antigen Benchmark

- Type: `paper` / `public code and data`
- URLs:
  - https://doi.org/10.1093/bioinformatics/btag136
  - https://github.com/samuelfromm/abag-benchmark-set
- Domain: antibody-antigen structure sampling, ranking, and confidence
- Verifier/reward signal: 110 complexes filtered against the training-era set,
  multiple sampled predictions per target, DockQ interface truth, and internal
  or derived confidence/ranking metrics.
- Project relevance: concrete source-backed C5 intake candidate. Its central
  ranking failure means the split must be target-grouped and the trust action
  must be calibrated against hidden interface success, not inferred from a
  high internal score.
- Claim boundary: the published benchmark evaluates model ranking and structure
  quality. It does not itself certify this project's RCPS deployment gate.
- Local checkpoint: exact `CC-BY-4.0` archive intake yields 22,000 AF3 samples
  over 110 targets. A frozen 55/55 target split finds 28 failures under
  trust-all, 3 failures among 20 targets selected by fixed `ipTM >= 0.80`, and
  no Hoeffding-certified trusted set at primary `alpha = 0.30`.

## FoldBench General-to-Ab-Ag Transfer Candidate

- Type: `paper` / `public benchmark`
- URL: https://www.nature.com/articles/s41467-025-67127-3
- Domain: low-homology all-atom structure prediction
- Verifier/reward signal: common DockQ-style interface evaluation across 279
  protein-protein and 172 antibody-antigen targets, with train-similarity
  controls.
- Project relevance: candidate substrate for calibrating a general PPI gate and
  testing transfer to Ab-Ag before fitting a regime-specific gate.
- Source audit: paper DOI `10.1038/s41467-025-67127-3`; MIT code/target archive
  DOI `10.5281/zenodo.17180806`; GitHub commit
  `4273f6877d82bd0b2fa476d1b2f34d121cbccc70`.
- Claim boundary: the public code/archive exposes target tables, evaluation
  code, and examples. The article Source Data workbook has target-level
  PPI/Ab-Ag DockQ sheets but no paired per-sample ranking/confidence fields for
  those strata; it also has a different redistribution boundary. General-PPI
  transfer remains deferred.

## Boltz-2

- Type: `official` / `paper`
- URLs:
  - https://jclinic.mit.edu/boltz-2-towards-accurate-and-efficient-binding-affinity-prediction/
  - https://github.com/jwohlwend/boltz
- Domain: biomolecular interaction and binding-affinity prediction
- Verifier/reward signal: structure prediction plus binding affinity estimates;
  open-source model and weights; affinity fields are explicitly distinct
  prediction outputs.
- Project relevance: practical specialist model for the local DBTL/structure
  design stack. It motivates evidence packets that separate predicted structure,
  predicted affinity, confidence, input constraints, and assay-verification
  status.
- Claim boundary: official/model source; use independent benchmarks and local
  calibration before turning confidence or affinity into a trust action.

## Boltz-2 Reliability Evaluation

- Type: `paper`
- URL: https://arxiv.org/html/2603.05532v1
- Domain: drug-discovery structure and affinity prediction
- Verifier/reward signal: comparison of Boltz-2 structural and affinity outputs
  against downstream reliability checks, including analysis of compressed
  confidence scores.
- Project relevance: important caution for `specialist_confidence`: a model can
  produce uniformly high confidence with limited discriminatory power, so the
  deployment layer needs calibrated thresholds or fail-closed behavior.
- Claim boundary: preprint and dataset-specific; use as cautionary evidence,
  not as a final general assessment of Boltz-2.

## Perturbation Foundation-Model Baseline Studies

- Type: `paper`
- URLs:
  - https://www.nature.com/articles/s41592-025-02772-6
  - https://openreview.net/forum?id=t04D9bkKUq&noteId=cVQ96mwheg
  - https://arxiv.org/html/2410.13956v2
- Domain: transcriptomics / single-cell perturbation prediction
- Verifier/reward signal: perturbation-effect labels, distribution-shift splits,
  and explicit comparison against simple baselines such as no-change, additive
  linear models, PCA, and scVI.
- Project relevance: strongest external support for the `baseline` action. In
  biology, the correct routing policy may be to use a cheap baseline instead of
  trusting a specialist foundation model.
- Claim boundary: transcriptomics perturbation setting; not directly equivalent
  to antibody-antigen structure prediction, but the trust-routing lesson
  transfers.

## scPerturBench

- Type: `paper` / `project page`
- URL: https://bm2-lab.github.io/scPerturBench-reproducibility/
- Domain: generalizable single-cell perturbation response prediction
- Verifier/reward signal: 27 perturbation-response methods, 29 datasets, unseen
  cellular-context and unseen-perturbation generalization, multiple evaluation
  metrics.
- Project relevance: reinforces that method choice is regime-dependent. A
  deployment agent should route by context, not by model popularity or generic
  "foundation model" status.
- Claim boundary: benchmark scope is single-cell perturbation response, not
  general biological reasoning.

## Risk-Controlling Prediction Sets / Conformal Risk Control

- Type: `paper`
- URLs:
  - https://arxiv.org/abs/2101.02703
  - https://arxiv.org/html/2208.02814v4
- Domain: uncertainty calibration / risk control
- Verifier/reward signal: finite-sample risk control around black-box model
  outputs, including protein-structure-prediction examples and extensions to
  monotone losses and distribution shift.
- Project relevance: methodological backbone for the local gate. The action
  `trust_specialist_output` should be allowed only when a calibrated risk
  threshold certifies the regime; otherwise route to `verify`, `baseline`, or
  `defer`.
- Claim boundary: guarantees depend on calibration assumptions and the
  exchangeability/shift model; OOD regimes must be handled explicitly.

## SAbDab2 Prospective C5 Substrate

- Type: `official data` / `official code releases`
- URLs:
  - https://doi.org/10.5281/zenodo.20083995
  - https://sabdab.opig.stats.ox.ac.uk/about
  - https://github.com/google-deepmind/alphafold3/releases/tag/v3.0.3
  - https://github.com/google-deepmind/alphafold3/blob/v3.0.3/docs/installation.md
  - https://github.com/google-deepmind/alphafold3/commit/dd1a7badb62cbb0d4571666002159842c8c578c5
  - https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md
  - https://github.com/wallnerlab/DockQ/releases/tag/v2.1.3
- Domain: prospective antibody-antigen confidence calibration.
- Verifier/reward signal: official sequence-aware source split, exact
  source/code/evaluator fingerprints, checksum-bound prediction inputs,
  pre-prediction environment attestation, and staged DockQ reveal.
- Project relevance: replaces post-hoc public-score reuse with a method-locked
  80-target calibration and 40-target evaluation panel on Cayuga. It also makes
  official terms-bound model parameters, full databases, and container identity
  explicit execution gates. The July 23 distribution update replaces the
  closed application form with a generation-pinned direct Google object.
- Claim boundary: native structures are public and this is not an independent
  hidden test. Before prediction and label reveal, the only supported claim is
  reproducible prospective method/source/input freezing.

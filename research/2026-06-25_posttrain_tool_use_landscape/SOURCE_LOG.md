# Source Log

Only sources that change a verifier, reward, policy, split, or claim boundary
are recorded here. Access date for this checkpoint: 2026-08-02.

| Source | Primary URL | Change to the benchmark |
| --- | --- | --- |
| SAbDab2 Machine Learning Dataset v0.1.0 | https://doi.org/10.5281/zenodo.20083995 | Replaces ad hoc target curation with a checksum-locked 15,641-row source and an official antibody-and-antigen sequence-aware split. The official `ab_ag_cluster` is now the prospective sampling unit after target-level selection exposed severe within-cluster concentration. |
| SAbDab2 license and distribution statement | https://sabdab.opig.stats.ox.ac.uk/about | Fixes the source attribution and `CC-BY-4.0` public-data boundary while keeping structures and sequences out of this repository. |
| AlphaFold 3 v3.0.3 | https://github.com/google-deepmind/alphafold3/releases/tag/v3.0.3 | Pins prospective prediction to commit `7b197fe859790fc3e04d03ea70dd0b9ba48881c9`; the Dockerfile fixes `/app/alphafold` plus `uv run python3`, while the writer fixes the sanitized job/five-sample output layout, ranking CSV and summary schemas, and a ranking score with disorder bonus/clash penalty rather than a strict `[0,1]` range. |
| AlphaFold 3 installation guide | https://github.com/google-deepmind/alphafold3/blob/v3.0.3/docs/installation.md | Makes official parameter authorization, direct receipt from Google, database presence, container identity, and high-memory GPU execution explicit preflight gates rather than implicit environment assumptions. |
| AlphaFold 3 parameter-distribution update, 2026-07-23 | https://github.com/google-deepmind/alphafold3/commit/dd1a7badb62cbb0d4571666002159842c8c578c5 | Retires the request form and replaces it with a direct official Google Storage object. The provisioning verifier now pins the documented URL, object generation, byte count, source-instruction commit, and terms acceptance before hashing any private copy. |
| AlphaFold 3 Model Parameters Terms of Use | https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md | Keeps parameter use non-commercial and terms-bound, forbids weight redistribution, and keeps AF3 output out of training for similar biomolecular structure-prediction technology. The benchmark publishes no weights and trains no AF3-like predictor. |
| AlphaFold 3 performance guide | https://github.com/google-deepmind/alphafold3/blob/v3.0.3/docs/performance.md | Fixes the split execution contract: CPU data-pipeline-only output is reusable by inference-only execution, while `--force_output_dir=true` permits validated continuation in the same target directory without changing final output intake. |
| DockQ v2.1.3 | https://github.com/wallnerlab/DockQ/releases/tag/v2.1.3 | Pins label calculation to commit `d9cbb1940bb0f42db3257f7da3b0e96f162b94d9` and fixes metric scope before prediction. |
| Risk-controlling prediction sets | https://arxiv.org/abs/2101.02703 | Keeps trust conditional on a finite-sample risk certificate and exposes the panel-size limit for secondary `alpha` targets. |
| Hoeffding, Probability Inequalities for Sums of Bounded Random Variables | https://doi.org/10.1080/01621459.1963.10500830 | Makes the independence assumption behind the prior target-level concentration bound explicit. The bound remains a sensitivity analysis, not the primary v2 certificate. |
| Distribution-Free Prediction Sets for Two-Layer Hierarchical Models | https://arxiv.org/abs/1809.07441 | Shows that repeated observations within groups do not satisfy ordinary exchangeability. This changes C5 selection from targets to one target per official sequence cluster. |
| Learn then Test: Calibrating Predictive Algorithms to Achieve Risk Control | https://arxiv.org/abs/2110.01052 | Reframes finite-grid calibration as multiple hypothesis testing. This motivates exact one-sided binomial tests with Bonferroni correction over the 50 preregistered thresholds. |

No full papers, source archives, model parameters, database files, raw
predictions, or long copied passages are stored here.

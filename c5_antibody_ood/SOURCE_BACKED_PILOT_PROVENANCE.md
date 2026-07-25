# C5 Source-Backed Pilot Provenance

## Source

- Paper: Fromm, Ludaic, and Elofsson, *Evaluating deep learning based
  structure prediction methods on antibody-antigen complexes*,
  `10.1093/bioinformatics/btag136`.
- Archive: Zenodo record `17978681`, DOI
  `10.5281/zenodo.17978681`, licensed `CC-BY-4.0`.
- Archive file: `abag-benchmark-set-main.zip`, MD5
  `2933fb1bb66903469d02c64a9d21f5d9`.
- Source member:
  `abag-benchmark-set-main/data/scores/alphafold3/scores_alphafold3.csv`.
- Source member SHA-256:
  `56259a84f1e8cc216e5ee91a96584f824ca46f062ef4f2c06aa4674472daf1c8`.
- Matching upstream repository commit:
  `06b21927bcacf6fb0612e56cdc110c206d9eebdc`.

On 2026-07-25, the source member was extracted by byte range from the
Zenodo ZIP and compared with the commit-pinned input. Its byte count
(`24,588,543`) and SHA-256 matched exactly. The raw CSV is an external input
and is not redistributed by this repository.

## Transformation

The adapter requires 22,000 rows, 110 targets, 200 samples per target, the
`alphafold3` preset, complete target/sample IDs, and unit-interval DockQ, ipTM,
and ranking-confidence values. A target contributes exactly one sample:
maximum ranking confidence, then lexical sample ID for deterministic ties.
The exported sample identifier is salted and hashed.

Targets are sorted by SHA-256 of
`c5-abag-af3-v1::<complex_id>`. The first 55 targets form calibration and the
remaining 55 form evaluation. A target can therefore appear in only one
partition. DockQ success is defined as `DockQ >= 0.23`.

The derived JSONL contains public PDB IDs, chain-role mappings, selected
confidence values, binary interface labels, split assignments, and source
fingerprints. These derived data retain the upstream `CC-BY-4.0` attribution.
Project-authored code and documentation remain under the repository license.

## Privacy And Release Boundary

The source has 56 columns. Intake uses an explicit nine-column allowlist and
drops 47 columns. The canonical run detected 132,000 absolute compute-path
cells in excluded columns; none were emitted. Raw structures, sequences,
feature files, raw prediction paths, and unhashed sample IDs are absent from
the derived manifest and compact report.

The result is a published-label replay. It does not claim an independent
hidden test, new structure prediction, general-PPI calibration transfer, or
model training evidence.

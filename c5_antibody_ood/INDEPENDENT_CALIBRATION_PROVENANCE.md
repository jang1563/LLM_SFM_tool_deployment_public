# C5 Independent Calibration Provenance

## Source

- Paper: Hitawala and Gray, *What does AlphaFold3 learn about antibody and
  nanobody docking, and what remains unsolved?*,
  `10.1080/19420862.2025.2545601`.
- Updated dataset: Zenodo record `16426003`, DOI
  `10.5281/zenodo.16426003`, version 4, licensed `CC-BY-4.0`.
- Archive: `revised_Compiled_Benchmark_Data.zip`, MD5
  `6d0c48b4c30b75a6e331c0d3aba4c010`.
- Code/data repository: `NooriFatima/AF3_AbNb_Benchmark`, licensed MIT.
- Pinned commit: `749933edc2b7b5f841f453a667bd2204d3e31e56`.
- Input member: `datafiles/final_af3_rmsds.csv`.
- Input SHA-256:
  `c012928f1bd36ac255a43b6a3abc33d4f59033b97f6655d9b7c300850e0c433b`.

The raw CSV is an external input and is not redistributed by this repository.
The associated Zenodo archive was inspected by HTTP byte range. It contains
7,603 members, including AF3 confidence JSON files and standardized prediction
structures. A targeted archive check confirmed the AF3 score fields and the
`H`, `L`, `A` heavy/light/antigen chain convention used by the derived rows.

## Transformation

The adapter requires the pinned 1,900-row table and its complete schema. It
retains only 1,565 bound rows across 108 targets with unit-interval `Rank`,
`ipTM_HA`, and `DockQ` values. Unbound rows are excluded because they do not
carry an antibody-antigen DockQ label.

One prediction is selected per complex using the source paper's ranking logic:
maximum AF3 ranking score, then maximum heavy-antigen ipTM for ranking-score
ties. A lexical sample-ID tie break makes the remaining tie deterministic. The
source sample filename is salted and hashed on export.

Every four-character PDB ID in the 110-row Fromm manifest is blocked before
calibration. This removes 9 overlapping PDB IDs representing 11 Gray complex
copies, leaving 97 independent-source targets: 44 antibodies and 53
nanobodies. Residual PDB overlap is zero.

Format counts are 44 antibodies and 53 nanobodies.

DockQ success is defined as `DockQ >= 0.23`. Antibody and nanobody formats are
calibrated separately over the pre-existing 50-threshold ranking-score family
from 0.50 through 0.99. The risk gate uses a uniform Hoeffding/union-bound
correction with `delta=0.10` and primary `alpha=0.30`. No threshold family was
changed after reading this source.

## Privacy And Claim Boundary

The exported manifest contains public complex IDs, chain roles, selected
confidence values, hidden binary interface labels, salted sample-ID hashes,
and source fingerprints. It contains no raw CSV, structure, sequence, local
path, source filename, or unhashed sample ID.

This is independent-source published-label calibration evidence followed by a
locked replay on the existing Fromm evaluation rows. It is not a blinded
hidden test, a new structure-prediction experiment, or general-PPI transfer.
Neither the antibody nor nanobody ranking-score gate is certified, so external
trust stays disabled and all frozen Fromm evaluation targets route to verify.

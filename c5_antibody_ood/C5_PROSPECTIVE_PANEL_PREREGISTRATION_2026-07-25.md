# C5 Prospective Panel Preregistration

Date: 2026-07-25

## Claim Boundary

This checkpoint freezes a prospective Stage B C5 method, source panel, and
AlphaFold 3 input set before prediction or DockQ label reveal. It is not a new
structure-prediction result, an independent hidden test, or permission to trust
specialist output.

The protocol commitment is:

```text
9c3fd6784fecef3b8971daedb8bfbfc3a1ca725f0353e05f0b7420a30f06e17a
```

## Locked Design

- Source: SAbDab2 Machine Learning Dataset v0.1.0, DOI
  `10.5281/zenodo.20083995`, `CC-BY-4.0`.
- Split: the official antibody-and-antigen sequence-aware `ab_ag_split`.
- Panel: one paired-chain antibody-antigen instance per PDB, released on or
  after 2021-10-01, X-ray or EM, resolution at most 4.0 Angstrom.
- Overlap boundary: every PDB used by the prior Fromm or Hitawala-Gray C5
  replay is excluded before deterministic selection.
- Prediction: AlphaFold 3 v3.0.3 at commit
  `7b197fe859790fc3e04d03ea70dd0b9ba48881c9`, one fixed seed, five diffusion
  samples, and templates disabled.
- Label: DockQ v2.1.3 at commit
  `d9cbb1940bb0f42db3257f7da3b0e96f162b94d9`, with interface success defined
  as `DockQ >= 0.23`.
- Risk gate: 50 ranking-score thresholds from 0.50 through 0.99, primary
  `alpha = 0.30`, `delta = 0.10`, and a uniform Hoeffding/union correction.
- Failure behavior: no certificate means verify all; score- or label-dependent
  seed extension is prohibited.

## Source And Input Freeze

The exact 15,641-row SAbDab2 split table passed byte-count, checksum, schema,
and split-count validation. Metadata-only filtering retained 2,417 eligible
rows. Selection produced:

| Role | Targets |
| --- | ---: |
| calibration | 80 |
| calibration reserve | 20 |
| evaluation | 40 |
| evaluation reserve | 10 |

The 150 selected rows have 150 unique PDB IDs, 150 unique SAbDab IDs, no
Fromm/Gray PDB overlap, and no source-cluster overlap between calibration and
evaluation splits.

Cayuga structure QC passed 150/150 committed targets with no reserve
promotions. The retained prediction set is 80 calibration plus 40 evaluation
targets. The public commitments are:

| Commitment | SHA-256 |
| --- | --- |
| candidate panel | `d1bb1352d58345ac3c45ad6bd229df4a19aa9d0789c0ab3fbec94fca4cdbf7ea` |
| retained panel | `aa9f751affb5249e17e39894c9f7a3a1cdca61b9f819e694df0e288cf92db1a2` |
| AF3 input set | `3569fc8641613c5328a05c991942a576f9ba1f9ad24daf135ea2b62806a52b18` |
| sequence set | `3d033f75c8d610c29f5012a050ed0d2ac42b8b2b38048677e28be384474b994d` |
| native structure set | `1c2570f996958ec3c2e12379b9864eded4596c34492ce75b53adc5a7b5e89e59` |

No sequence, structure, local path, DockQ value, interface label, AF3
parameter, prediction, or scheduler log is included in the public artifacts.

## Statistical Scope

For a zero-failure trusted set under the preregistered 50-threshold uniform
bound, the minimum calibration trusted counts are 35 at `alpha = 0.30`, 78 at
`alpha = 0.20`, and 311 at `alpha = 0.10`. The 80-target calibration panel can
therefore test the primary 0.30 target and, only in a near-zero-failure
high-coverage case, the 0.20 target. It cannot certify 0.10 under this design.
That limitation is fixed before prediction.

## Execution Gate

The first Cayuga environment check confirms:

- Singularity/Apptainer is available;
- the AF3 source commit and tag match;
- all 120 frozen inputs are present and checksum-matched;
- the output boundary is clean.

Prediction remains blocked because the checksum-locked AF3 container, official
model parameters, and database manifest are not present. The public readiness
summary records only booleans, counts, and checksums. The array submission
script verifies a passed private attestation before any GPU task starts.

## Next Decision

Obtain authorized official AF3 model parameters, build and checksum the pinned
container, install and inventory the official databases, then rerun the
fail-closed preflight. Submit the 120-target Cayuga array only after every
component passes. Model training, DPO, RLVR, threshold tuning, label reveal,
and external specialist trust remain closed.

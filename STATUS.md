# Project Status

Last updated: 2026-08-03

## Current Thesis

Biology agents should be evaluated on executable
`tool -> evidence packet -> terminal action` trajectories. Learned policies
remain useful where evidence-conditioned choices are genuinely uncertain, but
fixed transformations and safety invariants belong in fail-closed runtime
code. Specialist confidence permits trust only after regime-matched,
sampling-aware calibration. DPO/RLVR and public dataset publication remain
gated by independent scientific evidence.

## Current Result

Stage A remains a negative model-readiness result and a positive runtime-gate
result.

- Frozen routing scores 35/180 versus an 80/180 static prior.
- The runtime hybrid scores 115/180 with zero unsafe grounding but zero
  decisive learned-policy coverage.
- The deterministic tool-query compiler passes 25/25 valid cases and rejects
  150/150 malformed cases for the intended reason.
- These results do not justify broader SFT, DPO, RLVR, Hugging Face
  publication, or release tagging. The sealed extension commitment remains
  `post_training/stage_a_sealed_extension_commitment_2026-07-10.json` and its
  one-time result remains 5/25. Do not tune on or rescore these 25 sealed rows.

Stage B C5 has moved from the target-sampled v1 panel to a pre-prediction,
cluster-balanced v2 amendment.

- The v1 panel concentrated repeated observations within official SAbDab2
  antibody-antigen sequence clusters: calibration contained 34 clusters for
  80 targets, with 35 targets in the largest cluster; evaluation contained 17
  clusters for 40 targets, with 16 in the largest cluster. No v1 predictions
  or labels were generated after this diagnosis.
- The completed v1 Cayuga runtime attestation remains valid infrastructure
  evidence: all 25 required components passed with zero violations. It does
  not authorize v2 compute because its input and protocol commitments differ.
- The append-only v2 protocol samples one target per official
  `ab_ag_cluster`, then one target within each selected cluster using fixed
  public hashes. It retains 80 calibration and 40 evaluation clusters, with
  20 and 4 same-split reserve clusters.
- The primary calibration rule is now an exact one-sided binomial test with
  Bonferroni correction over the preregistered 50-threshold family. The prior
  uniform Hoeffding bound is retained as a conservative sensitivity analysis.
- The public candidate panel contains 144 targets from 144 unique source
  clusters, with zero prior-PDB overlap, zero split/cluster overlap, and no
  label access. Private structure QC passed 144/144, so no reserve promotion
  was needed.
- The retained panel contains 80 calibration and 40 evaluation targets from
  120 unique clusters. Cayuga persisted 120 template-free AF3 inputs. An
  independent host-side replay matched the file-set, sequence-set, and
  retained-manifest commitments and confirmed that no DockQ/interface label
  was read.
- At `alpha = 0.30`, `delta = 0.10`, and 50 candidate thresholds, a
  zero-failure calibration candidate needs at least 18 trusted clusters. A
  fixed evaluation policy with 10 trusted clusters and zero failures can pass
  the exact `alpha = 0.30` test. These are design properties, not performance
  results.
- Cluster deduplication removes the known repeated-sequence concentration but
  does not prove arbitrary future-distribution transfer. Any eventual risk
  statement remains conditional on the preregistered cluster-level sampling
  model and the frozen SAbDab2 regime.
- The v2 runtime attestation passed, and one CPU data-pipeline task completed.
  The pinned AF3 runtime accepts input JSON versions 1-4 but serializes the
  processed record at version 4, so the phase validator now checks that
  source-backed conversion while preserving name, seed, chain, sequence, and
  hidden-label isolation. The completed task is the panel maximum by total
  polymer length, not a representative median smoke.
- The remaining CPU array was stopped at its first runtime-contract failure.
  A label-blind audit found AF3-incompatible native chain IDs in 11/120 inputs
  and 22 chains. Nine canceled partial directories were checksum-preserved in
  private read-only quarantine; the single completed task remains unchanged.
- An append-only v3 refreeze changes only those 22 runtime chain IDs. It keeps
  all 120 targets, roles, native chain mappings, sequences, templates, seeds,
  dialect, and input version fixed; 109 input files and the completed task's
  input are byte-identical. The private native-to-runtime mapping is separate,
  while the tracked artifact exposes only aggregate counts and commitments.
- No model training, DPO, RLVR, prediction score, DockQ value, label, or
  external specialist trust is enabled.

The current workflow state is `v3_inputs_refrozen_runtime_reattestation_pending`.
Tracked v2/v3 artifacts contain public metadata, commitments, and aggregate
QC only. Raw structures, sequences, AF3 inputs, parameters, databases,
predictions, attestations, and scheduler logs remain private and uncommitted.

## Source Changes

The SAbDab2 v0.1.0 `ab_ag_cluster` field now defines the prospective sampling
unit, not only a split-overlap check. Its target-level concentration changed
the benchmark design before prediction.

Hoeffding's classical bound assumes independent bounded summands; the v1
cluster concentration therefore made a target-level interpretation
scientifically weak. Hierarchical prediction work separately shows why
repeated observations within groups break ordinary exchangeability. Learn
Then Test frames finite threshold selection as multiple testing, motivating
the preregistered exact binomial tests with Bonferroni correction. These
sources change the sampling verifier and risk certificate, so they are logged
in `research/2026-06-25_posttrain_tool_use_landscape/SOURCE_LOG.md` and the
verifier maps.

The pinned AlphaFold 3 v3.0.3 source adds two enforced verifier facts: accepted
input JSON versions are 1-4 while serialization uses version 4, and chain IDs
must satisfy the runtime's alphabetic/non-lowercase predicate with no
duplicates. This changes the phase and input verifiers, not the prediction
method. DockQ v2.1.3 remains unchanged.

## Next Decision

Proceed with `stage_b_c5_v3_reattestation_and_remaining_cpu_pipeline`.

1. Commit the public-safe v3 refreeze evidence and its source-backed verifier
   amendment. Keep the private chain mapping, inputs, partial quarantine, and
   runtime logs uncommitted.
2. Build a fresh Cayuga runtime attestation against the clean benchmark commit
   and v3 input commitment. The prior attestation is evidence, not current
   authorization, because both benchmark and input-set commitments changed.
3. Run an exact resume audit: require one completed CPU output, zero partials,
   byte-identical task-0 input, and successful amended phase/binding checks.
4. Regenerate the remaining-task launch audit, then run tasks 1-119. Require
   exact 120/120 processed-output coverage before GPU authorization.
5. Run one GPU inference task and validate the exact five-sample output before
   submitting the remaining GPU array.
6. Freeze all prediction outputs and the deterministic selected model before
   any native-label calculation.
7. Reveal exactly 80 calibration labels, freeze either the selected threshold
   or `verify_all`, and reveal the 40 evaluation labels once only after that
   policy is immutable.
8. Keep Stage A optimizer escalation and all C5 trust claims closed until the
   corresponding held-out evidence passes its preregistered gate.

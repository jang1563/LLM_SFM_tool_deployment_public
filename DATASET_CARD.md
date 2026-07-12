# Dataset Card: LLM-SFM Tool Deployment Artifacts

Draft Hugging Face dataset card for the Stage A trajectory benchmark, synthetic
trajectory demo, and planned A2 entity-resolution release.

## Dataset Summary

This dataset family supports research on biology tool-use trajectories and
free-text biomedical entity resolution.

Included public-safe subsets:

- `public_trajectory_demo`: synthetic trajectory records for evaluator demonstration.
- `stage_a_manifest`: 25 benchmark cases with model-visible task fields
  separated from hidden evaluator metadata.
- `stage_a_sft`: 25 oracle tool-use trajectories for supervised trajectory
  learning.
- `stage_a_preferences`: 150 chosen/rejected trajectory pairs across eight
  failure modes.
- `stage_a_process_supervision`: 25 process targets for tools, query fields,
  evidence status, terminal action, attribution, and source IDs.
- `stage_a_strict_contract`: 25 compact JSON SFT targets, 50 observed-collapse
  preference pairs, and 25 process targets for the `stage_a_v2_strict`
  saved-prediction contract.
- `stage_a_train_heldout`: deterministic 20/5 case-level split with no
  `source_manifest_case_id`, `split_group`, or `source_task_id` overlap.

Planned or local-only subsets:

- `a2_band_cases`: de-leaked ambiguous MONDO disease-resolution cases.
- `a2_band_train_eval`: concept-disjoint closed-set reranker train/eval JSONL.
- `negbiodb_ct_public_demo`: small public-compatible trajectory examples, if
  approved and stripped of private DB dependencies.

Status: draft card; project-specific Hugging Face dataset repository not yet
published.

## Stage A Trajectory Schema

The Stage A manifest uses two explicit layers:

- `model_visible_task`: `input_id`, natural-language `claim`, and allowed tools.
- `hidden_eval_metadata`: source task ID, evidence status, expected terminal
  action, required tools/query fields, attribution requirement, source IDs, and
  split group.

The post-training exports preserve that boundary:

- SFT rows contain prompt messages plus an oracle tool trajectory and final
  `submit_decision`.
- Strict-contract SFT rows contain the same model-visible prompt used by the
  saved-prediction producer plus one compact assistant JSON target with
  `action`, `evidence_status`, `tool_calls`, `cited_source_ids`, and
  `rationale`.
- Preference rows contain prompt messages, passing chosen trajectory messages,
  failing rejected trajectory messages, and validator scores for both sides.
- Strict-contract preference rows target the observed prompt-only failure shape:
  `verify`/`supported` collapse with missing or misrouted tool evidence.
- Process-supervision rows contain prompt messages plus target process fields.
- Split rows add only a `split` label while keeping all variants from the same
  case on the same train or held-out side.

## Stage A Failure Modes

The preference artifact covers:

- `self_answering_without_tools`
- `wrong_tool`
- `missing_tool`
- `partial_query`
- `missing_attribution`
- `invalid_value_missed`
- `unsupported_trust`
- `insufficient_as_negative`

## A2 Band Schema

Each reranker record contains:

- `query`: clinical-trial condition string or held-out ontology surface form
- `gold`: gold MONDO identifier
- `gold_name`: gold disease name
- `candidates`: list of candidate concepts with `mondo_id`, `name`, and
  de-leaked `syns`
- `gold_in_top8` or `gold_in`: whether the gold concept is among candidates
- optional `nbhd_clean`: neighborhood-clean evaluation flag

The target is the candidate index matching `gold`, or `0` when the gold concept
is absent.

## Leakage Control

An earlier prototype accidentally displayed the held-out query inside the gold
candidate synonym list. That inflated a no-model string-match baseline to 0.925.
The corrected dataset removes the query string from displayed synonyms before
evaluation and reports the de-leaked metrics.

## Intended Use

- Demonstrate the trajectory evaluator without private databases or API calls.
- Evaluate tool-use trajectories on hidden labels rather than prose quality.
- Run local SFT/preference smoke tests with checked train/held-out source
  separation.
- Generate saved prediction JSONL files first, then score those artifacts
  offline before interpreting API or local-checkpoint behavior.
- Evaluate closed-set biomedical entity reranking.
- Train lightweight rerankers above SapBERT-style retrieval.
- Study when SFT versus from-base RL helps verifiable biomedical decisions.
- Build demos for evidence-aware biology agents.

## Out-of-Scope Use

- Clinical diagnosis or treatment decisions.
- Claims about full real-world medical search behavior without additional
  external validation.
- Training a model to generate biomedical identifiers without retrieval.

## Known Limitations

- Stage A is a small benchmark substrate, not a broad clinical benchmark.
- The A2 set is anchored in MONDO/ChEMBL-style entity resolution and does not
  cover every biomedical ontology or all clinical text types.
- NegBioDB-CT trajectory artifacts may require a public-safe slice because the
  full SQLite substrate is not bundled here.
- Some post-training examples are oracle-derived scaffolds and should be treated
  as controlled evaluation/training artifacts.

## Public Demo

The synthetic demo lives at `demo/public_trajectory_cases.jsonl` and can be run
with:

```bash
python examples/run_public_demo.py
```

## Citation

See `CITATION.cff` in the repository.

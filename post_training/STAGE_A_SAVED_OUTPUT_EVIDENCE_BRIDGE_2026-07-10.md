# Stage A Saved-Output Evidence-Conditioned Bridge

Purpose: connect failed saved-output candidate policies to prompt-visible evidence-gate reasons before adding another optimizer.

## Bridge Summary

- Bridge rows: 18/18 joined to evidence-gate rows
- Unique failed cases: 4 (`["stage_a::000007", "stage_a::000012", "stage_a::000019", "stage_a::000021"]`)
- Runtime reason counts: `{"invalid_numeric_value_in_same_indication_record": 6, "mixed_endpoint_records_for_same_claim": 6, "no_same_indication_or_related_failure_record": 3, "related_evidence_without_same_indication_record": 3}`
- Target pair counts: `{"defer/insufficient": 3, "flag/invalid_value": 6, "reject/contradicted": 6, "verify/insufficient": 3}`
- Candidate prediction granularity: `policy_level_predicted_pair_counts_only`

## Runtime Reference

- Full-trajectory hybrid runtime-over-collapse: 25/25, mean score 1.0
- Routing evidence gate uses hidden labels: `False`

## Decision

- Selected next step: `build_evidence_conditioned_candidate_routing_slice`
- Do not run more standalone SFT yet: `True`

Candidate failures cover multiple evidence reasons and target pairs that the prompt-visible runtime evidence gate resolves. The next training substrate should condition candidate routing on those visible evidence features rather than repeat standalone candidate SFT on the same small slice.

## Next Data Contract

- Name: `stage_a_evidence_conditioned_candidate_routing_rows`
- Required fields: `case_id`, `case_family`, `target_pair`, `runtime_evidence_reason`, `visible_evidence_features`, `candidate_policy`, `candidate_policy_predicted_pair_counts`

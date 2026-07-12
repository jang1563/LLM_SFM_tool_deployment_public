# Stage A Saved-Output Post-Candidate-CE Next Decision

Purpose: decide the next research move after the candidate-CE checkpoint failed to meet the runtime arbitration baseline.

## Current Evidence

| Checkpoint | Raw exact | Calibrated exact | Best candidate exact | Runtime arbitration exact | Trusted incorrect max |
| --- | ---: | ---: | ---: | ---: | ---: |
| `balanced_nonflag_candidate_rank_readout` | 1/4 | 1/4 | 1 | 4 | 3 |
| `candidate_ce_action_status_pair_field_readout` | 1/4 | 1/4 | 1 | 4 | 3 |

## Runtime References

- Routing evidence gate held-out exact: 5/5
- Routing evidence gate all-row exact: 25/25
- Full-trajectory hybrid runtime-over-collapse: 25/25, mean score 1.0

## Decision

- Selected next step: `build_evidence_conditioned_saved_output_bridge`
- Passes meet-or-beat gate: `False`
- Rejected next steps: `more_standalone_candidate_sft`, `tool_query_component_training`, `DPO_or_preference_optimization`, `audited_RLVR`, `Hugging_Face_dataset_publication`, `v0.1_release_tagging`

Standalone saved-output candidate routing did not learn enough from small supervised checkpoints. The next scientifically useful step is to bridge those failed held-out candidate decisions to prompt-visible evidence reasons and the existing runtime evidence gate before adding more optimization objectives.

## Next Checkpoint Contract

- Name: `stage_a_saved_output_evidence_conditioned_bridge`
- Purpose: Map failed saved-output candidate choices to prompt-visible evidence reasons, runtime gate decisions, and full-trajectory violations before deciding whether another training objective is warranted.
- Acceptance gate: candidate exact must meet runtime baseline, trusted incorrect must be zero, hidden labels stay isolated, and raw outputs remain uncommitted.

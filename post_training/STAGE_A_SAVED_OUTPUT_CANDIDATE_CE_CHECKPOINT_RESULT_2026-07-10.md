# Stage A Saved-Output Candidate-CE Checkpoint Result

Purpose: compact public-safe result for the candidate-CE Cayuga checkpoint.

## Result

- Train candidate top-1: 4/16, top-pair counts `{"verify/insufficient": 16}`
- Raw held-out candidate top-1: 1/4, top-pair counts `{"verify/insufficient": 4}`
- Calibrated held-out candidate top-1: 1/4, top-pair counts `{"defer/insufficient": 1, "ground/supported": 1, "verify/insufficient": 2}`
- Field diagnostic: exact 1/4, action 1/4, status 2/4; patterns `{"action_field_failure": 1, "both_field_failure": 2, "pair_top1": 1}`
- Train-selected gate: 1/4 strict-final correct, trusted incorrect 0.

## Arbitration

| Policy | Exact | Rows | Trusted candidate | Trusted incorrect |
| --- | ---: | ---: | ---: | ---: |
| raw_candidate_top1 | 1 | 4 | 4 | 3 |
| calibrated_candidate_top1 | 1 | 4 | 4 | 3 |
| train_selected_score_gap_gate | 1 | 4 | 0 | 0 |
| evidence_gate_override | 4 | 4 | 0 | 0 |
| hybrid_evidence_then_train_gate | 4 | 4 | 0 | 0 |

## Decision

- Passes meet-or-beat gate: `False`
- Selected next step: `keep_runtime_evidence_arbitration_baseline`
- Interpretation: Candidate CE pair+field supervision shifted the candidate prior toward verify/insufficient but did not improve held-out exact top-1 or meet the runtime arbitration baseline.
- Next research move: Do not escalate to DPO/RLVR, tool_query, Hugging Face publication, or release tagging. Treat standalone candidate-routing SFT as insufficient and keep runtime evidence arbitration as the baseline while designing evidence-conditioned or larger-slice candidate routing checks.

Public-safety contract: raw saved predictions, candidate-score JSONL,
scheduler logs, model state, and ignored run folders were not copied
into this checkpoint.

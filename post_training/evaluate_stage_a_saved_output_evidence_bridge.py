#!/usr/bin/env python3
"""Bridge saved-output candidate failures to prompt-visible evidence reasons.

The bridge reads public compact checkpoints only. Candidate predictions are
kept at the granularity those checkpoints expose: policy-level pair counts plus
failed case IDs, not raw per-candidate score tables.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from post_training.run_stage_a_sft_smoke_eval import write_json  # noqa: E402


DATASET = "negbiodb_ct_stage_a_saved_output_evidence_bridge_v1"
DEFAULT_CANDIDATE_CE_RESULT = (
    "post_training/stage_a_saved_output_candidate_ce_checkpoint_result_2026-07-10.json"
)
DEFAULT_NONFLAG_RESULT = "post_training/stage_a_saved_output_nonflag_checkpoint_result_2026-07-10.json"
DEFAULT_POST_CE_DECISION = (
    "post_training/stage_a_saved_output_post_candidate_ce_next_decision_2026-07-10.json"
)
DEFAULT_ROUTING_EVIDENCE_GATE = "post_training/stage_a_routing_evidence_gate_2026-07-08.json"
DEFAULT_FULL_TRAJECTORY_ARBITRATION = (
    "post_training/stage_a_full_trajectory_arbitration_2026-07-09.json"
)

CANDIDATE_POLICIES = (
    "raw_candidate_top1",
    "calibrated_candidate_top1",
    "train_selected_score_gap_gate",
)
RUNTIME_POLICIES = ("evidence_gate_override", "hybrid_evidence_then_train_gate")


def load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def public_path(path: str | Path) -> str:
    path = Path(path)
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def input_artifact(path: str | Path, *, role: str) -> dict[str, str]:
    return {
        "path": public_path(path),
        "role": role,
        "sha256": sha256_file(path),
    }


def as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def policy_rows(checkpoint: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    runtime = as_mapping(as_mapping(checkpoint.get("result")).get("runtime_arbitration"))
    rows = runtime.get("policies", ())
    out: dict[str, Mapping[str, Any]] = {}
    if not isinstance(rows, list):
        return out
    for row in rows:
        if isinstance(row, Mapping):
            policy = row.get("policy")
            if isinstance(policy, str):
                out[policy] = row
    return out


def evidence_rows_by_case(gate: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = gate.get("rows", ())
    out: dict[str, Mapping[str, Any]] = {}
    if not isinstance(rows, list):
        return out
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        case_id = row.get("case_id")
        if isinstance(case_id, str):
            out[case_id] = row
    return out


def compact_features(row: Mapping[str, Any]) -> dict[str, Any]:
    features = as_mapping(row.get("features"))
    return {
        "completeness_signal": features.get("completeness_signal"),
        "records_considered": int(features.get("records_considered", 0)),
        "same_indication_record_count": int(features.get("same_indication_record_count", 0)),
        "related_negative_evidence_count": int(features.get("related_negative_evidence_count", 0)),
        "citation_candidate_count": len(features.get("citation_candidates", ()) or ()),
        "value_validity_findings_count": len(features.get("value_validity_findings", ()) or ()),
        "observed_tool_loop_present": bool(features.get("observed_tool_loop_present", False)),
    }


def bridge_rows_for_checkpoint(
    *,
    checkpoint_label: str,
    checkpoint: Mapping[str, Any],
    evidence_by_case: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    policies = policy_rows(checkpoint)
    runtime_exact = max(
        int(as_mapping(policies.get(policy)).get("exact", 0)) for policy in RUNTIME_POLICIES
    )
    for policy_name in CANDIDATE_POLICIES:
        policy = as_mapping(policies.get(policy_name))
        error_case_ids = [str(case_id) for case_id in policy.get("error_case_ids", ())]
        for case_id in error_case_ids:
            evidence = evidence_by_case.get(case_id)
            if evidence is None:
                rows.append(
                    {
                        "checkpoint": checkpoint_label,
                        "policy": policy_name,
                        "case_id": case_id,
                        "join_status": "missing_evidence_gate_row",
                    }
                )
                continue
            rows.append(
                {
                    "checkpoint": checkpoint_label,
                    "policy": policy_name,
                    "case_id": case_id,
                    "join_status": "joined",
                    "case_family": evidence.get("case_family"),
                    "target_pair": evidence.get("expected_pair"),
                    "runtime_evidence_pair": evidence.get("predicted_pair"),
                    "runtime_evidence_reason": evidence.get("reason"),
                    "runtime_evidence_exact": bool(evidence.get("exact", False)),
                    "candidate_policy_exact": int(policy.get("exact", 0)),
                    "candidate_policy_rows": int(policy.get("rows", 0)),
                    "candidate_policy_predicted_pair_counts": dict(
                        policy.get("by_predicted_pair", {})
                    ),
                    "candidate_policy_trusted_candidate": int(
                        policy.get("trusted_candidate", 0)
                    ),
                    "candidate_policy_trusted_incorrect": int(
                        policy.get("trusted_candidate_incorrect", 0)
                    ),
                    "runtime_arbitration_best_exact": runtime_exact,
                    "visible_evidence_features": compact_features(evidence),
                }
            )
    return rows


def summarize_full_trajectory(arbitration: Mapping[str, Any], path: str | Path) -> dict[str, Any]:
    all_summary = as_mapping(as_mapping(arbitration.get("summary")).get("all"))
    hybrid = as_mapping(all_summary.get("hybrid_runtime_over_collapse"))
    collapse = as_mapping(all_summary.get("ground_supported_collapse"))
    citationless = as_mapping(all_summary.get("citationless_runtime_action"))
    return {
        "path": public_path(path),
        "sha256": sha256_file(path),
        "raw_model_outputs_used": bool(arbitration.get("raw_model_outputs_used", True)),
        "hybrid_runtime_over_collapse": {
            "passed": int(hybrid.get("passed", 0)),
            "cases": int(hybrid.get("cases", 0)),
            "mean_score": hybrid.get("mean_score"),
            "violations": dict(hybrid.get("violations", {})),
        },
        "ground_supported_collapse": {
            "passed": int(collapse.get("passed", 0)),
            "cases": int(collapse.get("cases", 0)),
            "unsafe_ground_supported_overrides": int(
                collapse.get("unsafe_ground_supported_overrides", 0)
            ),
        },
        "citationless_runtime_action": {
            "passed": int(citationless.get("passed", 0)),
            "cases": int(citationless.get("cases", 0)),
            "violations": dict(citationless.get("violations", {})),
        },
    }


def summarize_bridge(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    joined = [row for row in rows if row.get("join_status") == "joined"]
    unique_cases = sorted({str(row["case_id"]) for row in joined})
    reason_by_case = {
        case_id: next(
            str(row["runtime_evidence_reason"])
            for row in joined
            if str(row["case_id"]) == case_id
        )
        for case_id in unique_cases
    }
    target_by_case = {
        case_id: next(str(row["target_pair"]) for row in joined if str(row["case_id"]) == case_id)
        for case_id in unique_cases
    }
    return {
        "bridge_rows": len(rows),
        "joined_rows": len(joined),
        "missing_evidence_rows": len(rows) - len(joined),
        "unique_failure_case_ids": unique_cases,
        "unique_failure_cases": len(unique_cases),
        "runtime_reason_counts": dict(
            sorted(Counter(str(row["runtime_evidence_reason"]) for row in joined).items())
        ),
        "target_pair_counts": dict(sorted(Counter(str(row["target_pair"]) for row in joined).items())),
        "policy_counts": dict(sorted(Counter(str(row["policy"]) for row in joined).items())),
        "checkpoint_counts": dict(
            sorted(Counter(str(row["checkpoint"]) for row in joined).items())
        ),
        "runtime_reasons_by_case": reason_by_case,
        "target_pairs_by_case": target_by_case,
        "all_joined_rows_runtime_exact": all(bool(row["runtime_evidence_exact"]) for row in joined),
        "candidate_prediction_granularity": "policy_level_predicted_pair_counts_only",
    }


def build_report(
    *,
    candidate_ce_result_path: str | Path = DEFAULT_CANDIDATE_CE_RESULT,
    nonflag_result_path: str | Path = DEFAULT_NONFLAG_RESULT,
    post_ce_decision_path: str | Path = DEFAULT_POST_CE_DECISION,
    routing_evidence_gate_path: str | Path = DEFAULT_ROUTING_EVIDENCE_GATE,
    full_trajectory_arbitration_path: str | Path = DEFAULT_FULL_TRAJECTORY_ARBITRATION,
) -> dict[str, Any]:
    candidate_ce = load_json(candidate_ce_result_path)
    nonflag = load_json(nonflag_result_path)
    decision = load_json(post_ce_decision_path)
    evidence_gate = load_json(routing_evidence_gate_path)
    full_trajectory = load_json(full_trajectory_arbitration_path)
    evidence_by_case = evidence_rows_by_case(evidence_gate)
    bridge_rows = (
        bridge_rows_for_checkpoint(
            checkpoint_label="balanced_nonflag_candidate_rank_readout",
            checkpoint=nonflag,
            evidence_by_case=evidence_by_case,
        )
        + bridge_rows_for_checkpoint(
            checkpoint_label="candidate_ce_action_status_pair_field_readout",
            checkpoint=candidate_ce,
            evidence_by_case=evidence_by_case,
        )
    )
    bridge_summary = summarize_bridge(bridge_rows)
    selected_next_step = as_mapping(decision.get("decision")).get("selected_next_step")
    return {
        "dataset": DATASET,
        "checkpoint_name": "stage_a_saved_output_evidence_conditioned_bridge",
        "input_artifacts": {
            "candidate_ce_result": input_artifact(
                candidate_ce_result_path, role="candidate-CE checkpoint result"
            ),
            "nonflag_result": input_artifact(
                nonflag_result_path, role="nonflag checkpoint result"
            ),
            "post_ce_next_decision": input_artifact(
                post_ce_decision_path, role="post-candidate-CE next decision"
            ),
            "routing_evidence_gate": input_artifact(
                routing_evidence_gate_path, role="routing evidence gate rows"
            ),
            "full_trajectory_arbitration": input_artifact(
                full_trajectory_arbitration_path,
                role="full-trajectory runtime arbitration reference",
            ),
        },
        "bridge_summary": bridge_summary,
        "bridge_rows": bridge_rows,
        "runtime_reference": {
            "routing_evidence_gate_model_visible_fields_only": bool(
                evidence_gate.get("model_visible_fields_only", False)
            ),
            "routing_evidence_gate_hidden_labels_used": bool(
                evidence_gate.get("hidden_labels_used_by_gate", True)
            ),
            "full_trajectory_arbitration": summarize_full_trajectory(
                full_trajectory, full_trajectory_arbitration_path
            ),
        },
        "decision": {
            "source_next_decision": selected_next_step,
            "selected_next_step": "build_evidence_conditioned_candidate_routing_slice",
            "do_not_run_more_standalone_sft_yet": True,
            "ready_for_tool_query": False,
            "ready_for_dpo_rlvr": False,
            "ready_for_hugging_face_publication": False,
            "ready_for_release_tagging": False,
            "interpretation": (
                "Candidate failures cover multiple evidence reasons and target pairs "
                "that the prompt-visible runtime evidence gate resolves. The next "
                "training substrate should condition candidate routing on those "
                "visible evidence features rather than repeat standalone candidate "
                "SFT on the same small slice."
            ),
        },
        "next_data_contract": {
            "name": "stage_a_evidence_conditioned_candidate_routing_rows",
            "must_include_fields": [
                "case_id",
                "case_family",
                "target_pair",
                "runtime_evidence_reason",
                "visible_evidence_features",
                "candidate_policy",
                "candidate_policy_predicted_pair_counts",
            ],
            "must_not_include": [
                "hidden_eval_metadata",
                "raw_candidate_score_table",
                "raw_model_text",
                "scheduler_log",
                "model_state_path",
            ],
            "acceptance_gate": {
                "no_missing_evidence_gate_rows": bridge_summary["missing_evidence_rows"] == 0,
                "runtime_evidence_rows_exact": bridge_summary["all_joined_rows_runtime_exact"],
                "candidate_granularity_disclosed": True,
                "raw_outputs_uncommitted": True,
            },
        },
        "public_safety_contract": {
            "raw_prediction_jsonl_read": False,
            "raw_candidate_score_jsonl_read": False,
            "scheduler_logs_read": False,
            "model_state_read": False,
            "ignored_run_folder_read": False,
            "raw_artifacts_committed": False,
            "hidden_labels_used_for_bridge": False,
        },
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    summary = report["bridge_summary"]
    decision = report["decision"]
    runtime = report["runtime_reference"]["full_trajectory_arbitration"]
    hybrid = runtime["hybrid_runtime_over_collapse"]
    lines = [
        "# Stage A Saved-Output Evidence-Conditioned Bridge",
        "",
        "Purpose: connect failed saved-output candidate policies to prompt-visible evidence-gate reasons before adding another optimizer.",
        "",
        "## Bridge Summary",
        "",
        f"- Bridge rows: {summary['joined_rows']}/{summary['bridge_rows']} joined to evidence-gate rows",
        f"- Unique failed cases: {summary['unique_failure_cases']} (`{json.dumps(summary['unique_failure_case_ids'])}`)",
        f"- Runtime reason counts: `{json.dumps(summary['runtime_reason_counts'], sort_keys=True)}`",
        f"- Target pair counts: `{json.dumps(summary['target_pair_counts'], sort_keys=True)}`",
        f"- Candidate prediction granularity: `{summary['candidate_prediction_granularity']}`",
        "",
        "## Runtime Reference",
        "",
        (
            f"- Full-trajectory hybrid runtime-over-collapse: "
            f"{hybrid['passed']}/{hybrid['cases']}, mean score {hybrid['mean_score']}"
        ),
        (
            "- Routing evidence gate uses hidden labels: "
            f"`{report['runtime_reference']['routing_evidence_gate_hidden_labels_used']}`"
        ),
        "",
        "## Decision",
        "",
        f"- Selected next step: `{decision['selected_next_step']}`",
        f"- Do not run more standalone SFT yet: `{decision['do_not_run_more_standalone_sft_yet']}`",
        "",
        decision["interpretation"],
        "",
        "## Next Data Contract",
        "",
        f"- Name: `{report['next_data_contract']['name']}`",
        "- Required fields: "
        + ", ".join(f"`{field}`" for field in report["next_data_contract"]["must_include_fields"]),
    ]
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-ce-result", default=DEFAULT_CANDIDATE_CE_RESULT)
    parser.add_argument("--nonflag-result", default=DEFAULT_NONFLAG_RESULT)
    parser.add_argument("--post-ce-decision", default=DEFAULT_POST_CE_DECISION)
    parser.add_argument("--routing-evidence-gate", default=DEFAULT_ROUTING_EVIDENCE_GATE)
    parser.add_argument(
        "--full-trajectory-arbitration", default=DEFAULT_FULL_TRAJECTORY_ARBITRATION
    )
    parser.add_argument(
        "--out-json",
        default="post_training/stage_a_saved_output_evidence_bridge_2026-07-10.json",
    )
    parser.add_argument(
        "--out-md",
        default="post_training/STAGE_A_SAVED_OUTPUT_EVIDENCE_BRIDGE_2026-07-10.md",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(
        candidate_ce_result_path=args.candidate_ce_result,
        nonflag_result_path=args.nonflag_result,
        post_ce_decision_path=args.post_ce_decision,
        routing_evidence_gate_path=args.routing_evidence_gate,
        full_trajectory_arbitration_path=args.full_trajectory_arbitration,
    )
    write_json(Path(args.out_json), report)
    Path(args.out_md).write_text(render_markdown(report))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Choose the next Stage A move after the candidate-CE checkpoint.

This checkpoint reads only public-safe compact artifacts. It does not inspect
raw saved predictions, candidate-score JSONL, scheduler logs, model states, or
ignored run directories.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from post_training.run_stage_a_sft_smoke_eval import write_json  # noqa: E402


DATASET = "negbiodb_ct_stage_a_saved_output_post_candidate_ce_next_decision_v1"
DEFAULT_CANDIDATE_CE_RESULT = (
    "post_training/stage_a_saved_output_candidate_ce_checkpoint_result_2026-07-10.json"
)
DEFAULT_NONFLAG_RESULT = "post_training/stage_a_saved_output_nonflag_checkpoint_result_2026-07-10.json"
DEFAULT_ROUTING_EVIDENCE_GATE = "post_training/stage_a_routing_evidence_gate_2026-07-08.json"
DEFAULT_ROUTING_MODEL_READINESS = "post_training/stage_a_routing_model_readiness_2026-07-09.json"
DEFAULT_FULL_TRAJECTORY_ARBITRATION = (
    "post_training/stage_a_full_trajectory_arbitration_2026-07-09.json"
)


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


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def policy_from_checkpoint(checkpoint: Mapping[str, Any], policy_name: str) -> dict[str, Any]:
    policies = _as_mapping(_as_mapping(checkpoint.get("result")).get("runtime_arbitration")).get(
        "policies", ()
    )
    if not isinstance(policies, list):
        return {
            "policy": policy_name,
            "exact": 0,
            "rows": 0,
            "trusted_candidate": 0,
            "trusted_candidate_incorrect": 0,
            "error_case_ids": [],
            "by_predicted_pair": {},
        }
    for policy in policies:
        if isinstance(policy, Mapping) and policy.get("policy") == policy_name:
            return {
                "policy": policy_name,
                "exact": int(policy.get("exact", 0)),
                "rows": int(policy.get("rows", 0)),
                "trusted_candidate": int(policy.get("trusted_candidate", 0)),
                "trusted_candidate_incorrect": int(policy.get("trusted_candidate_incorrect", 0)),
                "error_case_ids": list(policy.get("error_case_ids", ())),
                "by_predicted_pair": dict(policy.get("by_predicted_pair", {})),
            }
    return {
        "policy": policy_name,
        "exact": 0,
        "rows": 0,
        "trusted_candidate": 0,
        "trusted_candidate_incorrect": 0,
        "error_case_ids": [],
        "by_predicted_pair": {},
    }


def summarize_checkpoint(
    *, label: str, checkpoint: Mapping[str, Any], path: str | Path
) -> dict[str, Any]:
    result = _as_mapping(checkpoint.get("result"))
    raw = _as_mapping(result.get("raw_heldout_candidate_top1"))
    calibrated = _as_mapping(result.get("calibrated_heldout_candidate_top1"))
    field = _as_mapping(result.get("field_diagnostic"))
    train_gate = _as_mapping(result.get("train_selected_gate"))
    raw_policy = policy_from_checkpoint(checkpoint, "raw_candidate_top1")
    calibrated_policy = policy_from_checkpoint(checkpoint, "calibrated_candidate_top1")
    train_policy = policy_from_checkpoint(checkpoint, "train_selected_score_gap_gate")
    evidence_policy = policy_from_checkpoint(checkpoint, "evidence_gate_override")
    hybrid_policy = policy_from_checkpoint(checkpoint, "hybrid_evidence_then_train_gate")
    candidate_exact_values = [
        int(raw.get("exact", 0)),
        int(calibrated.get("exact", 0)),
        int(train_policy.get("exact", 0)),
    ]
    runtime_exact_values = [
        int(evidence_policy.get("exact", 0)),
        int(hybrid_policy.get("exact", 0)),
    ]
    return {
        "label": label,
        "path": public_path(path),
        "sha256": sha256_file(path),
        "dataset": checkpoint.get("dataset"),
        "run_id": checkpoint.get("run_id"),
        "checkpoint_name": checkpoint.get("checkpoint_name"),
        "raw_heldout_candidate_top1": {
            "exact": int(raw.get("exact", 0)),
            "rows": int(raw.get("rows", 0)),
            "top_pair_counts": dict(raw.get("top_pair_counts", {})),
        },
        "calibrated_heldout_candidate_top1": {
            "exact": int(calibrated.get("exact", 0)),
            "rows": int(calibrated.get("rows", 0)),
            "top_pair_counts": dict(calibrated.get("top_pair_counts", {})),
        },
        "field_diagnostic": {
            "exact_top1": int(field.get("exact_top1", 0)),
            "rows": int(field.get("rows", 0)),
            "action_top1": int(field.get("action_top1", 0)),
            "evidence_status_top1": int(field.get("evidence_status_top1", 0)),
            "field_rank_patterns": dict(field.get("field_rank_patterns", {})),
        },
        "train_selected_gate": {
            "strict_final_correct": int(train_gate.get("strict_final_correct", 0)),
            "trusted": int(train_gate.get("trusted", 0)),
            "trusted_incorrect": int(train_gate.get("trusted_incorrect", 0)),
        },
        "candidate_policy_readouts": [raw_policy, calibrated_policy, train_policy],
        "runtime_arbitration_readouts": [evidence_policy, hybrid_policy],
        "best_candidate_exact": max(candidate_exact_values),
        "best_runtime_arbitration_exact": max(runtime_exact_values),
        "candidate_trusted_incorrect_max": max(
            int(raw_policy.get("trusted_candidate_incorrect", 0)),
            int(calibrated_policy.get("trusted_candidate_incorrect", 0)),
            int(train_policy.get("trusted_candidate_incorrect", 0)),
        ),
        "runtime_trusted_incorrect_max": max(
            int(evidence_policy.get("trusted_candidate_incorrect", 0)),
            int(hybrid_policy.get("trusted_candidate_incorrect", 0)),
        ),
        "decision": dict(_as_mapping(checkpoint.get("decision"))),
    }


def summarize_routing_evidence_gate(gate: Mapping[str, Any], path: str | Path) -> dict[str, Any]:
    summary = _as_mapping(gate.get("summary"))
    all_rows = _as_mapping(summary.get("all"))
    heldout = _as_mapping(summary.get("heldout"))
    return {
        "path": public_path(path),
        "sha256": sha256_file(path),
        "dataset": gate.get("dataset"),
        "model_visible_fields_only": bool(gate.get("model_visible_fields_only", False)),
        "hidden_labels_used_by_gate": bool(gate.get("hidden_labels_used_by_gate", True)),
        "all": {
            "exact": int(all_rows.get("exact", 0)),
            "rows": int(all_rows.get("rows", 0)),
            "by_predicted_pair": dict(all_rows.get("by_predicted_pair", {})),
        },
        "heldout": {
            "exact": int(heldout.get("exact", 0)),
            "rows": int(heldout.get("rows", 0)),
            "by_predicted_pair": dict(heldout.get("by_predicted_pair", {})),
        },
    }


def summarize_routing_readiness(readiness: Mapping[str, Any], path: str | Path) -> dict[str, Any]:
    decision = _as_mapping(readiness.get("decision"))
    all_family = _as_mapping(readiness.get("all_family_model_results"))
    compact: dict[str, Any] = {}
    for name in ("freeform_routing_after_loop", "constrained_routing_observed_pair"):
        row = _as_mapping(all_family.get(name))
        heldout = _as_mapping(row.get("heldout_summary"))
        gate = _as_mapping(row.get("readiness"))
        compact[name] = {
            "heldout_passed": int(heldout.get("passed", 0)),
            "heldout_cases": int(heldout.get("cases", 0)),
            "mean_score": heldout.get("mean_score"),
            "ready_for_escalation": bool(gate.get("ready_for_escalation", False)),
        }
    return {
        "path": public_path(path),
        "sha256": sha256_file(path),
        "dataset": readiness.get("dataset"),
        "all_family_model_results": compact,
        "decision": {
            "ready_for_tool_query": bool(decision.get("ready_for_tool_query", False)),
            "ready_for_dpo_rlvr": bool(decision.get("ready_for_dpo_rlvr", False)),
            "ready_for_hugging_face_publication": bool(
                decision.get("ready_for_hugging_face_publication", False)
            ),
            "ready_for_release_tagging": bool(decision.get("ready_for_release_tagging", False)),
            "runtime_enforcement_required": bool(
                decision.get("runtime_enforcement_required", True)
            ),
            "blockers": list(decision.get("blockers", ())),
        },
    }


def summarize_full_trajectory_arbitration(
    arbitration: Mapping[str, Any], path: str | Path
) -> dict[str, Any]:
    all_summary = _as_mapping(_as_mapping(arbitration.get("summary")).get("all"))
    compact: dict[str, dict[str, Any]] = {}
    for policy in (
        "ground_supported_collapse",
        "citationless_runtime_action",
        "hybrid_runtime_over_collapse",
        "oracle_full",
    ):
        row = _as_mapping(all_summary.get(policy))
        compact[policy] = {
            "passed": int(row.get("passed", 0)),
            "cases": int(row.get("cases", 0)),
            "mean_score": row.get("mean_score"),
            "unsafe_ground_supported_overrides": int(
                row.get("unsafe_ground_supported_overrides", 0)
            ),
            "violations": dict(row.get("violations", {})),
        }
    return {
        "path": public_path(path),
        "sha256": sha256_file(path),
        "dataset": arbitration.get("dataset"),
        "raw_model_outputs_used": bool(arbitration.get("raw_model_outputs_used", True)),
        "policies": compact,
    }


def build_decision(
    *,
    checkpoint_summaries: list[Mapping[str, Any]],
    evidence_gate: Mapping[str, Any],
    routing_readiness: Mapping[str, Any],
    full_trajectory: Mapping[str, Any],
) -> dict[str, Any]:
    candidate_best = max(int(row.get("best_candidate_exact", 0)) for row in checkpoint_summaries)
    runtime_best = max(
        int(row.get("best_runtime_arbitration_exact", 0)) for row in checkpoint_summaries
    )
    candidate_rows = max(
        int(row["raw_heldout_candidate_top1"].get("rows", 0)) for row in checkpoint_summaries
    )
    evidence_heldout = _as_mapping(evidence_gate.get("heldout"))
    hybrid = _as_mapping(_as_mapping(full_trajectory.get("policies")).get("hybrid_runtime_over_collapse"))
    routing_decision = _as_mapping(routing_readiness.get("decision"))
    blockers = [
        "nonflag and candidate-CE standalone candidate checkpoints both remain below runtime arbitration",
        "raw/calibrated candidate trust still produces unsafe incorrect held-out decisions",
        "routing model readiness remains closed for tool_query and DPO/RLVR",
    ]
    if int(hybrid.get("passed", 0)) >= int(hybrid.get("cases", 0)) and int(hybrid.get("cases", 0)) > 0:
        blockers.append("runtime-plus-evidence hybrid already has a full-trajectory passing reference")
    return {
        "selected_next_step": "build_evidence_conditioned_saved_output_bridge",
        "passes_meet_or_beat_gate": False,
        "candidate_best_exact": candidate_best,
        "candidate_rows": candidate_rows,
        "runtime_arbitration_best_exact": runtime_best,
        "routing_evidence_gate_heldout_exact": int(evidence_heldout.get("exact", 0)),
        "routing_evidence_gate_heldout_rows": int(evidence_heldout.get("rows", 0)),
        "rejected_next_steps": [
            "more_standalone_candidate_sft",
            "tool_query_component_training",
            "DPO_or_preference_optimization",
            "audited_RLVR",
            "Hugging_Face_dataset_publication",
            "v0.1_release_tagging",
        ],
        "ready_flags": {
            "ready_for_tool_query": bool(routing_decision.get("ready_for_tool_query", False)),
            "ready_for_dpo_rlvr": bool(routing_decision.get("ready_for_dpo_rlvr", False)),
            "ready_for_hugging_face_publication": bool(
                routing_decision.get("ready_for_hugging_face_publication", False)
            ),
            "ready_for_release_tagging": bool(routing_decision.get("ready_for_release_tagging", False)),
        },
        "blockers": blockers,
        "interpretation": (
            "Standalone saved-output candidate routing did not learn enough from "
            "small supervised checkpoints. The next scientifically useful step is "
            "to bridge those failed held-out candidate decisions to prompt-visible "
            "evidence reasons and the existing runtime evidence gate before adding "
            "more optimization objectives."
        ),
    }


def build_report(
    *,
    candidate_ce_result_path: str | Path = DEFAULT_CANDIDATE_CE_RESULT,
    nonflag_result_path: str | Path = DEFAULT_NONFLAG_RESULT,
    routing_evidence_gate_path: str | Path = DEFAULT_ROUTING_EVIDENCE_GATE,
    routing_model_readiness_path: str | Path = DEFAULT_ROUTING_MODEL_READINESS,
    full_trajectory_arbitration_path: str | Path = DEFAULT_FULL_TRAJECTORY_ARBITRATION,
) -> dict[str, Any]:
    candidate_ce = load_json(candidate_ce_result_path)
    nonflag = load_json(nonflag_result_path)
    evidence_gate = load_json(routing_evidence_gate_path)
    routing_readiness = load_json(routing_model_readiness_path)
    full_trajectory = load_json(full_trajectory_arbitration_path)
    checkpoints = [
        summarize_checkpoint(
            label="balanced_nonflag_candidate_rank_readout",
            checkpoint=nonflag,
            path=nonflag_result_path,
        ),
        summarize_checkpoint(
            label="candidate_ce_action_status_pair_field_readout",
            checkpoint=candidate_ce,
            path=candidate_ce_result_path,
        ),
    ]
    evidence_summary = summarize_routing_evidence_gate(
        evidence_gate, routing_evidence_gate_path
    )
    readiness_summary = summarize_routing_readiness(
        routing_readiness, routing_model_readiness_path
    )
    full_summary = summarize_full_trajectory_arbitration(
        full_trajectory, full_trajectory_arbitration_path
    )
    decision = build_decision(
        checkpoint_summaries=checkpoints,
        evidence_gate=evidence_summary,
        routing_readiness=readiness_summary,
        full_trajectory=full_summary,
    )
    return {
        "dataset": DATASET,
        "checkpoint_name": "post_candidate_ce_next_decision",
        "input_artifacts": {
            "candidate_ce_result": input_artifact(
                candidate_ce_result_path, role="candidate-CE checkpoint result"
            ),
            "nonflag_result": input_artifact(
                nonflag_result_path, role="nonflag checkpoint result"
            ),
            "routing_evidence_gate": input_artifact(
                routing_evidence_gate_path, role="deterministic routing evidence gate"
            ),
            "routing_model_readiness": input_artifact(
                routing_model_readiness_path, role="routing model readiness checkpoint"
            ),
            "full_trajectory_arbitration": input_artifact(
                full_trajectory_arbitration_path,
                role="canonical full-trajectory arbitration checkpoint",
            ),
        },
        "current_evidence": {
            "standalone_candidate_checkpoints": checkpoints,
            "routing_evidence_gate": evidence_summary,
            "routing_model_readiness": readiness_summary,
            "full_trajectory_arbitration": full_summary,
        },
        "decision": decision,
        "next_checkpoint_contract": {
            "name": "stage_a_saved_output_evidence_conditioned_bridge",
            "purpose": (
                "Map failed saved-output candidate choices to prompt-visible evidence "
                "reasons, runtime gate decisions, and full-trajectory violations before "
                "deciding whether another training objective is warranted."
            ),
            "minimum_public_inputs": [
                "post_training/stage_a_saved_output_candidate_ce_checkpoint_result_2026-07-10.json",
                "post_training/stage_a_saved_output_nonflag_checkpoint_result_2026-07-10.json",
                "post_training/stage_a_routing_evidence_gate_2026-07-08.json",
                "post_training/stage_a_full_trajectory_arbitration_2026-07-09.json",
            ],
            "acceptance_gate": {
                "candidate_policy_exact_must_meet_runtime_baseline": True,
                "candidate_policy_trusted_incorrect_must_be_zero": True,
                "hidden_labels_must_remain_isolated": True,
                "raw_outputs_must_remain_uncommitted": True,
                "must_report_failure_case_ids_and_predicted_pair_counts": True,
            },
        },
        "public_safety_contract": {
            "raw_prediction_jsonl_read": False,
            "raw_candidate_score_jsonl_read": False,
            "scheduler_logs_read": False,
            "model_state_read": False,
            "ignored_run_folder_read": False,
            "raw_artifacts_committed": False,
            "hidden_labels_used_for_decision": False,
        },
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    evidence = report["current_evidence"]
    decision = report["decision"]
    checkpoints = evidence["standalone_candidate_checkpoints"]
    routing_gate = evidence["routing_evidence_gate"]
    full = evidence["full_trajectory_arbitration"]["policies"]
    hybrid = full["hybrid_runtime_over_collapse"]
    lines = [
        "# Stage A Saved-Output Post-Candidate-CE Next Decision",
        "",
        "Purpose: decide the next research move after the candidate-CE checkpoint failed to meet the runtime arbitration baseline.",
        "",
        "## Current Evidence",
        "",
        "| Checkpoint | Raw exact | Calibrated exact | Best candidate exact | Runtime arbitration exact | Trusted incorrect max |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in checkpoints:
        raw = row["raw_heldout_candidate_top1"]
        calibrated = row["calibrated_heldout_candidate_top1"]
        lines.append(
            "| `{label}` | {raw_exact}/{raw_rows} | {cal_exact}/{cal_rows} | {best} | {runtime} | {trusted_bad} |".format(
                label=row["label"],
                raw_exact=raw["exact"],
                raw_rows=raw["rows"],
                cal_exact=calibrated["exact"],
                cal_rows=calibrated["rows"],
                best=row["best_candidate_exact"],
                runtime=row["best_runtime_arbitration_exact"],
                trusted_bad=row["candidate_trusted_incorrect_max"],
            )
        )
    lines.extend(
        [
            "",
            "## Runtime References",
            "",
            (
                f"- Routing evidence gate held-out exact: "
                f"{routing_gate['heldout']['exact']}/{routing_gate['heldout']['rows']}"
            ),
            (
                f"- Routing evidence gate all-row exact: "
                f"{routing_gate['all']['exact']}/{routing_gate['all']['rows']}"
            ),
            (
                f"- Full-trajectory hybrid runtime-over-collapse: "
                f"{hybrid['passed']}/{hybrid['cases']}, mean score {hybrid['mean_score']}"
            ),
            "",
            "## Decision",
            "",
            f"- Selected next step: `{decision['selected_next_step']}`",
            f"- Passes meet-or-beat gate: `{decision['passes_meet_or_beat_gate']}`",
            (
                "- Rejected next steps: "
                + ", ".join(f"`{step}`" for step in decision["rejected_next_steps"])
            ),
            "",
            decision["interpretation"],
            "",
            "## Next Checkpoint Contract",
            "",
            f"- Name: `{report['next_checkpoint_contract']['name']}`",
            f"- Purpose: {report['next_checkpoint_contract']['purpose']}",
            "- Acceptance gate: candidate exact must meet runtime baseline, trusted incorrect must be zero, hidden labels stay isolated, and raw outputs remain uncommitted.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-ce-result", default=DEFAULT_CANDIDATE_CE_RESULT)
    parser.add_argument("--nonflag-result", default=DEFAULT_NONFLAG_RESULT)
    parser.add_argument("--routing-evidence-gate", default=DEFAULT_ROUTING_EVIDENCE_GATE)
    parser.add_argument("--routing-model-readiness", default=DEFAULT_ROUTING_MODEL_READINESS)
    parser.add_argument(
        "--full-trajectory-arbitration", default=DEFAULT_FULL_TRAJECTORY_ARBITRATION
    )
    parser.add_argument(
        "--out-json",
        default="post_training/stage_a_saved_output_post_candidate_ce_next_decision_2026-07-10.json",
    )
    parser.add_argument(
        "--out-md",
        default="post_training/STAGE_A_SAVED_OUTPUT_POST_CANDIDATE_CE_NEXT_DECISION_2026-07-10.md",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(
        candidate_ce_result_path=args.candidate_ce_result,
        nonflag_result_path=args.nonflag_result,
        routing_evidence_gate_path=args.routing_evidence_gate,
        routing_model_readiness_path=args.routing_model_readiness,
        full_trajectory_arbitration_path=args.full_trajectory_arbitration,
    )
    write_json(Path(args.out_json), report)
    Path(args.out_md).write_text(render_markdown(report))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

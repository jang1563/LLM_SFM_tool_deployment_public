#!/usr/bin/env python3
"""Export Stage A manifest rows into SFT, preference, and process data."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm_sfm_tool_deployment import Action, EvidenceStatus, EvaluationResult, Trajectory
from negbiodb_ct.stage_a_manifest import (
    failure_trajectories_for_stage_a_row,
    ideal_trajectory_from_stage_a_row,
    load_stage_a_manifest,
    score_stage_a_trajectory,
    validate_stage_a_manifest,
)


STAGE_A_SFT_DATASET = "negbiodb_ct_stage_a_sft_v1"
STAGE_A_PREFERENCE_DATASET = "negbiodb_ct_stage_a_preferences_v1"
STAGE_A_PROCESS_DATASET = "negbiodb_ct_stage_a_process_v1"
STAGE_A_EXPORT_STRATEGY = "oracle_vs_failure_variants_v1"
ALLOWED_TERMINAL_ACTIONS = (
    Action.GROUND_WITH_ATTRIBUTION.value,
    Action.VERIFY_WITH_ASSAY_OR_DATABASE.value,
    Action.DEFER_OR_REQUEST_MORE_EVIDENCE.value,
    Action.REJECT_OR_FLAG_UNSUPPORTED_CLAIM.value,
    Action.USE_CHEAP_BASELINE.value,
    Action.TRUST_SPECIALIST_OUTPUT.value,
)


def prompt_messages(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    visible = row["model_visible_task"]
    return [
        {
            "role": "system",
            "content": (
                "You are a Stage A evidence-status trajectory agent. Use the listed tools, "
                "return sourced evidence, and choose a final action from the allowed action set."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "input_id": visible["input_id"],
                    "claim": visible["claim"],
                    "allowed_tools": visible["allowed_tools"],
                    "allowed_terminal_actions": list(ALLOWED_TERMINAL_ACTIONS),
                },
                sort_keys=True,
            ),
        },
    ]


def trajectory_messages(trajectory: Trajectory) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for step in trajectory.steps:
        messages.append(
            {
                "role": "assistant",
                "tool_call": {
                    "name": step.name,
                    "arguments": dict(step.arguments),
                },
            }
        )
        messages.append(
            {
                "role": "tool",
                "name": step.name,
                "content": dict(step.observation),
            }
        )
    messages.append(final_message(trajectory))
    return messages


def final_decision_json(trajectory: Trajectory) -> dict[str, Any]:
    out: dict[str, Any] = {
        "action": action_value(trajectory.terminal_action),
        "evidence_status": evidence_status_value(
            trajectory.predicted_evidence_status
            or trajectory.evidence_packet.negative_evidence_status
        ),
    }
    if trajectory.cited_source_ids:
        out["cited_source_ids"] = list(trajectory.cited_source_ids)
    fail_closed_reason = trajectory.evidence_packet.fail_closed_reason
    if fail_closed_reason:
        out["fail_closed_reason"] = fail_closed_reason
    return out


def final_message(trajectory: Trajectory) -> dict[str, Any]:
    return {
        "role": "assistant",
        "tool_call": {
            "name": "submit_decision",
            "arguments": final_decision_json(trajectory),
        },
    }


def sft_row(row: Mapping[str, Any]) -> dict[str, Any]:
    trajectory = ideal_trajectory_from_stage_a_row(row)
    score = score_stage_a_trajectory(row, trajectory)
    hidden = row["hidden_eval_metadata"]
    return {
        "id": f"stage_a_sft::{row['case_id']}",
        "dataset": STAGE_A_SFT_DATASET,
        "task_id": row["case_id"],
        "source_manifest_case_id": row["case_id"],
        "source_task_id": hidden.get("source_task_id"),
        "tool_profile": hidden.get("tool_profile"),
        "case_family": hidden.get("case_family"),
        "gold_evidence_status": hidden.get("gold_evidence_status"),
        "expected_terminal_action": hidden.get("expected_terminal_action"),
        "split_group": hidden.get("split_group"),
        "oracle_target": True,
        "messages": prompt_messages(row) + trajectory_messages(trajectory),
        "target_trajectory": serialize_trajectory(trajectory),
        "score": score_payload(score),
    }


def preference_rows(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    chosen = ideal_trajectory_from_stage_a_row(row)
    chosen_score = score_stage_a_trajectory(row, chosen)
    if not chosen_score.passed:
        raise ValueError(f"Chosen Stage A trajectory does not pass for {row['case_id']}")

    out: list[dict[str, Any]] = []
    hidden = row["hidden_eval_metadata"]
    for failure_mode, rejected in sorted(failure_trajectories_for_stage_a_row(row).items()):
        rejected_score = score_stage_a_trajectory(row, rejected)
        if rejected_score.passed:
            raise ValueError(f"Rejected Stage A variant unexpectedly passes: {row['case_id']} {failure_mode}")
        out.append(
            {
                "id": f"stage_a_pref::{row['case_id']}::{failure_mode}",
                "dataset": STAGE_A_PREFERENCE_DATASET,
                "task_id": row["case_id"],
                "source_manifest_case_id": row["case_id"],
                "source_task_id": hidden.get("source_task_id"),
                "tool_profile": hidden.get("tool_profile"),
                "case_family": hidden.get("case_family"),
                "split_group": hidden.get("split_group"),
                "strategy": STAGE_A_EXPORT_STRATEGY,
                "failure_mode": failure_mode,
                "prompt_messages": prompt_messages(row),
                "chosen_messages": trajectory_messages(chosen),
                "rejected_messages": trajectory_messages(rejected),
                "chosen": json.dumps(final_decision_json(chosen), sort_keys=True),
                "rejected": json.dumps(final_decision_json(rejected), sort_keys=True),
                "chosen_trajectory": serialize_trajectory(chosen),
                "rejected_trajectory": serialize_trajectory(rejected),
                "chosen_score": score_payload(chosen_score),
                "rejected_score": score_payload(rejected_score),
                "metadata": {
                    "gold_evidence_status": hidden.get("gold_evidence_status"),
                    "expected_terminal_action": hidden.get("expected_terminal_action"),
                    "gold_source_ids": list(hidden.get("gold_source_ids", ())),
                },
            }
        )
    return out


def process_row(row: Mapping[str, Any]) -> dict[str, Any]:
    trajectory = ideal_trajectory_from_stage_a_row(row)
    score = score_stage_a_trajectory(row, trajectory)
    hidden = row["hidden_eval_metadata"]
    return {
        "id": f"stage_a_process::{row['case_id']}",
        "dataset": STAGE_A_PROCESS_DATASET,
        "task_id": row["case_id"],
        "source_manifest_case_id": row["case_id"],
        "source_task_id": hidden.get("source_task_id"),
        "tool_profile": hidden.get("tool_profile"),
        "case_family": hidden.get("case_family"),
        "split_group": hidden.get("split_group"),
        "prompt_messages": prompt_messages(row),
        "target_process": [
            {"name": "required_tools", "value": list(hidden.get("required_tools", ()))},
            {"name": "required_query_fields", "value": list(hidden.get("required_query_fields", ()))},
            {"name": "evidence_status", "value": hidden.get("gold_evidence_status")},
            {"name": "terminal_action", "value": hidden.get("expected_terminal_action")},
            {"name": "gold_source_ids", "value": list(hidden.get("gold_source_ids", ()))},
            {"name": "requires_attribution", "value": bool(hidden.get("requires_attribution"))},
        ],
        "target_trajectory": serialize_trajectory(trajectory),
        "score": score_payload(score),
    }


def build_stage_a_exports(rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    sft = [sft_row(row) for row in rows]
    preferences: list[dict[str, Any]] = []
    process = [process_row(row) for row in rows]
    for row in rows:
        preferences.extend(preference_rows(row))
    return sft, preferences, process


def manifest_for_exports(
    *,
    source_manifest: str | Path,
    sft_out: str | Path,
    preference_out: str | Path,
    process_out: str | Path,
    rows: Sequence[Mapping[str, Any]],
    sft_rows: Sequence[Mapping[str, Any]],
    preference_rows_out: Sequence[Mapping[str, Any]],
    process_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    split_groups = [str(row["hidden_eval_metadata"]["split_group"]) for row in rows]
    overlap_groups = sorted(group for group, count in Counter(split_groups).items() if count > 1)
    return {
        "dataset": "negbiodb_ct_stage_a_export_v1",
        "strategy": STAGE_A_EXPORT_STRATEGY,
        "source_manifest": display_path(source_manifest),
        "sft": display_path(sft_out),
        "preferences": display_path(preference_out),
        "process": display_path(process_out),
        "sft_dataset": STAGE_A_SFT_DATASET,
        "preference_dataset": STAGE_A_PREFERENCE_DATASET,
        "process_dataset": STAGE_A_PROCESS_DATASET,
        "source_cases": len(rows),
        "sft_examples": len(sft_rows),
        "preference_pairs": len(preference_rows_out),
        "process_examples": len(process_rows),
        "by_case_family": counts(rows, lambda row: row["hidden_eval_metadata"]["case_family"]),
        "by_evidence_status": counts(rows, lambda row: row["hidden_eval_metadata"]["gold_evidence_status"]),
        "preference_failure_modes": counts(preference_rows_out, lambda row: row["failure_mode"]),
        "chosen_passed": sum(bool(row.get("chosen_score", {}).get("passed")) for row in preference_rows_out),
        "rejected_passed": sum(bool(row.get("rejected_score", {}).get("passed")) for row in preference_rows_out),
        "split_groups": sorted(set(split_groups)),
        "split_group_overlap": overlap_groups,
        "boundary": (
            "Stage A no-API export from a manifest with model-visible prompts "
            "separated from hidden evaluator metadata."
        ),
    }


def write_jsonl(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def score_payload(result: EvaluationResult) -> dict[str, Any]:
    return {
        "passed": result.passed,
        "score": round(result.score, 3),
        "earned": result.earned,
        "possible": result.possible,
        "reward_breakdown": dict(result.reward_breakdown),
        "violations": list(result.violations),
    }


def serialize_trajectory(trajectory: Trajectory) -> dict[str, Any]:
    return {
        "input_id": trajectory.input_id,
        "steps": [
            {
                "name": step.name,
                "arguments": dict(step.arguments),
                "observation": dict(step.observation),
            }
            for step in trajectory.steps
        ],
        "terminal_action": action_value(trajectory.terminal_action),
        "cited_source_ids": list(trajectory.cited_source_ids),
        "predicted_evidence_status": evidence_status_value(
            trajectory.predicted_evidence_status
            or trajectory.evidence_packet.negative_evidence_status
        ),
    }


def action_value(value: Action | str) -> str:
    return value.value if isinstance(value, Action) else str(value)


def evidence_status_value(value: EvidenceStatus | str) -> str:
    return value.value if isinstance(value, EvidenceStatus) else str(value)


def counts(rows: Sequence[Mapping[str, Any]], key_fn) -> dict[str, int]:
    return dict(sorted(Counter(str(key_fn(row)) for row in rows).items()))


def display_path(path: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="negbiodb_ct/stage_a_mini_manifest.jsonl")
    parser.add_argument("--sft-out", default="post_training/stage_a_sft_v1.jsonl")
    parser.add_argument("--preference-out", default="post_training/stage_a_preferences_v1.jsonl")
    parser.add_argument("--process-out", default="post_training/stage_a_process_supervision_v1.jsonl")
    parser.add_argument("--manifest-out", default="post_training/stage_a_export_manifest.json")
    parser.add_argument("--min-rows", type=int, default=20)
    args = parser.parse_args()

    rows = load_stage_a_manifest(args.manifest)
    issues = validate_stage_a_manifest(rows, min_rows=args.min_rows)
    if issues:
        raise SystemExit("Stage A manifest validation failed:\n- " + "\n- ".join(issues))

    sft, preferences, process = build_stage_a_exports(rows)
    write_jsonl(args.sft_out, sft)
    write_jsonl(args.preference_out, preferences)
    write_jsonl(args.process_out, process)
    manifest = manifest_for_exports(
        source_manifest=args.manifest,
        sft_out=args.sft_out,
        preference_out=args.preference_out,
        process_out=args.process_out,
        rows=rows,
        sft_rows=sft,
        preference_rows_out=preferences,
        process_rows=process,
    )
    write_json(args.manifest_out, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    if manifest["split_group_overlap"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

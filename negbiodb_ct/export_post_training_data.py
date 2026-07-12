#!/usr/bin/env python3
"""Export NegBioDB-CT runner outputs into post-training JSONL artifacts."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm_sfm_tool_deployment import TrajectoryEvaluator
from negbiodb_ct import load_task_records, task_spec_from_record, trajectory_from_model_output
from negbiodb_ct.run_agent import CT_DB, SYS_NATIVE, check_other, search_failures


DATASET_NAME = "negbiodb_ct_native_trajectory_v1"


def load_runner_artifact(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as handle:
        data = json.load(handle)
    if not isinstance(data, dict) or "rows" not in data:
        raise ValueError(f"Runner artifact is missing rows: {path}")
    return data


def task_index(path: str | Path) -> dict[str, dict[str, Any]]:
    return {record["packet_id"]: record for record in load_task_records(path)}


def tool_results_for_record(
    record: Mapping[str, Any],
    con: sqlite3.Connection,
) -> dict[str, Any]:
    observation = record["observation"]
    scoring_key = record["scoring_key"]
    inject_nct = scoring_key["gold_nct"] if scoring_key["inject_impossible_value"] else None
    drug_id = observation["drug_id"]
    condition_id = observation["condition_id"]
    return {
        "search_failures": search_failures(con, drug_id, condition_id, inject_nct),
        "check_other_indications": check_other(con, drug_id, condition_id),
    }


def final_decision_json(model_output: Mapping[str, Any]) -> dict[str, Any]:
    action = str(model_output["action"])
    out: dict[str, Any] = {"action": action}
    citations = model_output.get("cited_source_ids") or []
    if action in {"ground", "flag"} and citations:
        out["nct"] = citations[0]
    return out


def base_prompt_messages(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    claim = record["observation"]["claim"]
    return [
        {"role": "system", "content": SYS_NATIVE},
        {"role": "user", "content": claim},
    ]


def tool_context_messages(tool_results: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {"role": "assistant", "tool_call": {"name": "search_failures", "arguments": {}}},
        {"role": "tool", "name": "search_failures", "content": tool_results["search_failures"]},
        {"role": "assistant", "tool_call": {"name": "check_other_indications", "arguments": {}}},
        {"role": "tool", "name": "check_other_indications", "content": tool_results["check_other_indications"]},
    ]


def prompt_messages(
    record: Mapping[str, Any],
    tool_results: Mapping[str, Any],
) -> list[dict[str, Any]]:
    return base_prompt_messages(record) + tool_context_messages(tool_results)


def final_message(model_output: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "role": "assistant",
        "tool_call": {
            "name": "submit_decision",
            "arguments": final_decision_json(model_output),
        },
    }


def supervised_example(
    record: Mapping[str, Any],
    runner_row: Mapping[str, Any],
    tool_results: Mapping[str, Any],
    *,
    source_runner: str,
) -> dict[str, Any]:
    return {
        "id": f"sft::{record['packet_id']}",
        "dataset": DATASET_NAME,
        "task_id": record["packet_id"],
        "source_runner": source_runner,
        "tool_profile": "native_ct",
        "action_class": record["action_class"],
        "messages": prompt_messages(record, tool_results) + [
            final_message(runner_row["model_output"]),
        ],
        "target_model_output": runner_row["model_output"],
        "score": {
            "generic_score": runner_row["generic_score"],
            "reward": runner_row["reward"],
            "violations": runner_row["generic_violations"],
        },
        "metadata": {
            "gold_action": record["scoring_key"]["gold_action"],
            "gold_nct": record["scoring_key"]["gold_nct"],
            "inject_impossible_value": record["scoring_key"]["inject_impossible_value"],
        },
    }


def preference_pairs(
    record: Mapping[str, Any],
    runner_row: Mapping[str, Any],
    tool_results: Mapping[str, Any],
    *,
    source_runner: str,
) -> list[dict[str, Any]]:
    chosen_output = dict(runner_row["model_output"])
    chosen_score = score_model_output(record, chosen_output)
    if not chosen_score["passed"]:
        return []

    out = []
    action_class = str(record["action_class"])
    called = list(runner_row["called"])

    if action_class in {"ground", "flag"}:
        rejected = {"action": action_class, "called": called}
        out.append(
            build_preference_pair(
                record,
                source_runner,
                tool_results,
                chosen_output,
                rejected,
                "missing_attribution",
            )
        )

    if action_class == "reject":
        nct = first_returned_nct(tool_results)
        if nct:
            rejected = {"action": "ground", "called": called, "cited_source_ids": [nct]}
            out.append(
                build_preference_pair(
                    record,
                    source_runner,
                    tool_results,
                    chosen_output,
                    rejected,
                    "mixed_endpoint_over_grounding",
                )
            )

    rejected = {"action": "self_answer", "called": []}
    out.append(
        build_preference_pair(
            record,
            source_runner,
            tool_results,
            chosen_output,
            rejected,
            "self_answering_without_tools",
        )
    )
    return out


def build_preference_pair(
    record: Mapping[str, Any],
    source_runner: str,
    tool_results: Mapping[str, Any],
    chosen_output: Mapping[str, Any],
    rejected_output: Mapping[str, Any],
    failure_mode: str,
) -> dict[str, Any]:
    chosen_score = score_model_output(record, chosen_output)
    rejected_score = score_model_output(record, rejected_output)
    return {
        "id": f"pref::{record['packet_id']}::{failure_mode}",
        "dataset": DATASET_NAME,
        "task_id": record["packet_id"],
        "source_runner": source_runner,
        "tool_profile": "native_ct",
        "failure_mode": failure_mode,
        "prompt_messages": base_prompt_messages(record),
        "chosen_messages": tool_context_messages(tool_results) + [final_message(chosen_output)],
        "rejected_messages": rejected_messages(tool_results, rejected_output, failure_mode),
        "chosen": json.dumps(final_decision_json(chosen_output), sort_keys=True),
        "rejected": json.dumps(final_decision_json(rejected_output), sort_keys=True),
        "chosen_model_output": dict(chosen_output),
        "rejected_model_output": dict(rejected_output),
        "chosen_score": chosen_score,
        "rejected_score": rejected_score,
        "metadata": {
            "action_class": record["action_class"],
            "gold_action": record["scoring_key"]["gold_action"],
            "gold_nct": record["scoring_key"]["gold_nct"],
        },
    }


def rejected_messages(
    tool_results: Mapping[str, Any],
    rejected_output: Mapping[str, Any],
    failure_mode: str,
) -> list[dict[str, Any]]:
    if failure_mode == "self_answering_without_tools":
        return [
            {
                "role": "assistant",
                "content": json.dumps(final_decision_json(rejected_output), sort_keys=True),
            }
        ]
    return tool_context_messages(tool_results) + [final_message(rejected_output)]


def score_model_output(record: Mapping[str, Any], output: Mapping[str, Any]) -> dict[str, Any]:
    trajectory = trajectory_from_model_output(record, output, tool_profile="native_ct")
    result = TrajectoryEvaluator().evaluate(
        task_spec_from_record(record, tool_profile="native_ct"),
        trajectory,
    )
    return {
        "passed": result.passed,
        "score": round(result.score, 3),
        "violations": list(result.violations),
    }


def first_returned_nct(tool_results: Mapping[str, Any]) -> str | None:
    for row in tool_results.get("search_failures", []):
        nct = row.get("nct") if isinstance(row, Mapping) else None
        if nct:
            return str(nct)
    return None


def write_jsonl(path: str | Path, rows: list[Mapping[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def resolve_repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.exists() or candidate.is_absolute():
        return candidate
    return ROOT / candidate


def display_path(path: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", required=True, help="Path to a negbiodb_ct/run_agent.py JSON artifact.")
    parser.add_argument("--tasks", default="negbiodb_ct/tasks_pilot.jsonl")
    parser.add_argument("--sft-out", default="post_training/negbiodb_ct_native_sft_v1.jsonl")
    parser.add_argument("--pref-out", default="post_training/negbiodb_ct_native_preferences_v1.jsonl")
    parser.add_argument("--manifest-out", default="post_training/negbiodb_ct_native_manifest.json")
    parser.add_argument("--allow-failures", action="store_true")
    args = parser.parse_args()

    runner_path = resolve_repo_path(args.runner)
    tasks_path = resolve_repo_path(args.tasks)
    data = load_runner_artifact(runner_path)
    tasks = task_index(tasks_path)
    source_runner = str(data.get("summary", {}).get("model", "unknown"))
    con = sqlite3.connect(CT_DB)

    sft_rows: list[dict[str, Any]] = []
    pref_rows: list[dict[str, Any]] = []
    skipped = []
    for row in data["rows"]:
        task_id = row["packet_id"]
        record = tasks.get(task_id)
        if record is None:
            skipped.append({"packet_id": task_id, "reason": "missing_task_record"})
            continue
        if not args.allow_failures and (not row.get("correct") or row.get("generic_violations")):
            raise SystemExit(f"Runner row is not clean: {task_id}")
        observations = tool_results_for_record(record, con)
        sft_rows.append(
            supervised_example(record, row, observations, source_runner=source_runner)
        )
        pref_rows.extend(
            preference_pairs(record, row, observations, source_runner=source_runner)
        )

    write_jsonl(args.sft_out, sft_rows)
    write_jsonl(args.pref_out, pref_rows)

    manifest = {
        "dataset": DATASET_NAME,
        "runner": display_path(runner_path),
        "tasks": display_path(tasks_path),
        "source_runner": source_runner,
        "runner_summary": data.get("summary", {}),
        "sft_examples": len(sft_rows),
        "preference_pairs": len(pref_rows),
        "preference_failure_modes": dict(Counter(row["failure_mode"] for row in pref_rows)),
        "skipped": skipped,
    }
    manifest_path = Path(args.manifest_out)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

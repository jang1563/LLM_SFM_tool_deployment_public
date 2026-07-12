#!/usr/bin/env python3
"""Build evidence-boundary preference pairs from native CT SFT rows."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from negbiodb_ct import load_task_records  # noqa: E402
from negbiodb_ct.export_post_training_data import (  # noqa: E402
    final_decision_json,
    final_message,
    score_model_output,
)
from post_training.build_sft_boundary_rationale_data import BOUNDARY_NEGATIVES  # noqa: E402
from post_training.evidence_rationale import (  # noqa: E402
    evidence_action,
    evidence_decision,
    first_clean_efficacy_failure_nct,
    first_invalid_p_value_nct,
    search_failures_content,
)
from post_training.split_sft_data import load_jsonl, write_jsonl  # noqa: E402


DEFAULT_DATASET = "negbiodb_ct_oracle_boundary_preferences_v1"
DEFAULT_STRATEGY = "evidence_boundary_contrast_v1"
NATIVE_TOOL_CALLS = ["search_failures", "check_other_indications"]


def task_index(path: str | Path) -> dict[str, dict[str, Any]]:
    return {record["packet_id"]: record for record in load_task_records(path)}


def first_available_nct(row: Mapping[str, Any]) -> str | None:
    for record in search_failures_content(row):
        nct = record.get("nct")
        if nct:
            return str(nct)
    return None


def citation_for_action(row: Mapping[str, Any], action: str) -> str | None:
    if action == "flag":
        return first_invalid_p_value_nct(row) or first_available_nct(row)
    if action == "ground":
        return first_clean_efficacy_failure_nct(row) or first_available_nct(row)
    return None


def model_output_for_action(row: Mapping[str, Any], action: str) -> dict[str, Any]:
    output: dict[str, Any] = {"action": action, "called": list(NATIVE_TOOL_CALLS)}
    citation = citation_for_action(row, action)
    if citation:
        output["cited_source_ids"] = [citation]
    return output


def prompt_messages(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    messages = list(row["messages"])
    if not messages or messages[-1].get("tool_call", {}).get("name") != "submit_decision":
        raise ValueError(f"Expected final submit_decision for {row.get('id')}")
    return messages[:-1]


def boundary_preference_pairs_for_row(
    row: Mapping[str, Any],
    record: Mapping[str, Any],
    *,
    dataset: str,
    strategy: str,
    pair_index: int,
) -> list[dict[str, Any]]:
    chosen_action = evidence_action(row)
    action_class = str(row.get("action_class"))
    if chosen_action != action_class:
        raise ValueError(
            f"Evidence action mismatch for {row.get('id')}: "
            f"evidence={chosen_action} action_class={action_class}"
        )

    chosen_output = model_output_for_action(row, chosen_action)
    chosen_score = score_model_output(record, chosen_output)
    if not chosen_score["passed"]:
        raise ValueError(f"Chosen boundary output does not pass for {row.get('id')}: {chosen_score}")

    pairs = []
    for rejected_action in BOUNDARY_NEGATIVES[chosen_action]:
        rejected_output = model_output_for_action(row, rejected_action)
        rejected_score = score_model_output(record, rejected_output)
        if rejected_score["passed"]:
            raise ValueError(
                f"Rejected boundary output unexpectedly passes for {row.get('id')}: "
                f"{rejected_action} {rejected_score}"
            )
        failure_mode = f"boundary_{chosen_action}_over_{rejected_action}"
        pairs.append(
            {
                "id": f"pref::{row['task_id']}::{failure_mode}::{pair_index}",
                "dataset": dataset,
                "task_id": row["task_id"],
                "source_example_id": row["id"],
                "source_dataset": row.get("dataset"),
                "tool_profile": row.get("tool_profile", "native_ct"),
                "strategy": strategy,
                "failure_mode": failure_mode,
                "boundary_pair_role": "preference",
                "boundary_pair_index": pair_index,
                "evidence_derived_action": chosen_action,
                "rejected_action": rejected_action,
                "prompt_messages": prompt_messages(row),
                "chosen_messages": [final_message(chosen_output)],
                "rejected_messages": [final_message(rejected_output)],
                "chosen": json.dumps(final_decision_json(chosen_output), sort_keys=True),
                "rejected": json.dumps(final_decision_json(rejected_output), sort_keys=True),
                "chosen_model_output": chosen_output,
                "rejected_model_output": rejected_output,
                "chosen_score": chosen_score,
                "rejected_score": rejected_score,
                "metadata": {
                    "action_class": action_class,
                    "gold_action": row.get("metadata", {}).get("gold_action", action_class),
                    "gold_nct": row.get("metadata", {}).get("gold_nct"),
                    "boundary_negative_actions": list(BOUNDARY_NEGATIVES[chosen_action]),
                },
            }
        )
    return pairs


def build_boundary_preference_pairs(
    rows: list[Mapping[str, Any]],
    tasks: Mapping[str, Mapping[str, Any]],
    *,
    dataset: str,
    strategy: str = DEFAULT_STRATEGY,
) -> list[dict[str, Any]]:
    pairs = []
    pair_index = 0
    for row in rows:
        task_id = str(row["task_id"])
        record = tasks.get(task_id)
        if record is None:
            raise KeyError(f"Missing task record for {task_id}")
        row_pairs = boundary_preference_pairs_for_row(
            row,
            record,
            dataset=dataset,
            strategy=strategy,
            pair_index=pair_index,
        )
        pairs.extend(row_pairs)
        pair_index += 1
    return pairs


def counts(rows: list[Mapping[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(key)) for row in rows).items()))


def manifest_for_pairs(
    *,
    source_sft: str | Path,
    tasks: str | Path,
    out: str | Path,
    dataset: str,
    strategy: str,
    source_rows: list[Mapping[str, Any]],
    pairs: list[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "source_sft": str(source_sft),
        "tasks": str(tasks),
        "out": str(out),
        "dataset": dataset,
        "strategy": strategy,
        "boundary": (
            "Evidence-derived boundary preference pairs. Chosen and rejected "
            "responses share the same visible native CT tool observations and "
            "differ only in the terminal submit_decision action."
        ),
        "source_examples": len(source_rows),
        "preference_pairs": len(pairs),
        "source_by_action_class": counts(source_rows, "action_class"),
        "pairs_by_chosen_action": counts(pairs, "evidence_derived_action"),
        "pairs_by_rejected_action": counts(pairs, "rejected_action"),
        "pairs_by_failure_mode": counts(pairs, "failure_mode"),
        "chosen_passed": sum(bool(row.get("chosen_score", {}).get("passed")) for row in pairs),
        "rejected_passed": sum(bool(row.get("rejected_score", {}).get("passed")) for row in pairs),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft", default="post_training/negbiodb_ct_oracle_sft_v1.jsonl")
    parser.add_argument("--tasks", default="negbiodb_ct/tasks_pilot.jsonl")
    parser.add_argument("--out", default="post_training/negbiodb_ct_oracle_boundary_preferences_v1.jsonl")
    parser.add_argument(
        "--manifest-out",
        default="post_training/negbiodb_ct_oracle_boundary_preferences_manifest.json",
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--strategy", default=DEFAULT_STRATEGY)
    args = parser.parse_args()

    rows = load_jsonl(args.sft)
    tasks = task_index(args.tasks)
    pairs = build_boundary_preference_pairs(
        rows,
        tasks,
        dataset=args.dataset,
        strategy=args.strategy,
    )
    write_jsonl(args.out, pairs)
    manifest = manifest_for_pairs(
        source_sft=args.sft,
        tasks=args.tasks,
        out=args.out,
        dataset=args.dataset,
        strategy=args.strategy,
        source_rows=rows,
        pairs=pairs,
    )
    Path(args.manifest_out).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Export separable strict Stage A component targets.

The strict SFT smoke showed that one compact JSON target still mixes several
failure sources. This exporter creates smaller supervised slices for:

- enum/action decoding;
- ordered tool calls with required query fields;
- routing after a valid tool loop is present.

The rows reuse the existing strict-contract SFT artifacts and split boundaries.
No live API calls, model weights, or private database material are used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from post_training.run_stage_a_strict_contract_sft_smoke import (  # noqa: E402
    load_jsonl,
    source_case_id,
    target_output_from_row,
    write_json,
    write_jsonl,
)


DATASET = "negbiodb_ct_stage_a_strict_component_targets_v1"
MANIFEST_DATASET = "negbiodb_ct_stage_a_strict_component_targets_manifest_v1"
PROMPT_CONTRACT = "stage_a_v2_strict"
COMPONENTS = ("enum_action", "tool_query", "routing_after_loop")
ALLOWED_ACTIONS = ("ground", "reject", "defer", "verify", "flag", "self_answer")
ALLOWED_EVIDENCE_STATUSES = (
    "supported",
    "contradicted",
    "invalid_value",
    "insufficient",
    "unknown",
)
REQUIRED_QUERY_FIELDS = ("drug_id", "condition_id")
TARGET_KEYS_BY_COMPONENT = {
    "enum_action": ("action", "evidence_status"),
    "tool_query": ("tool_calls",),
    "routing_after_loop": ("action", "evidence_status", "cited_source_ids"),
}


def prompt_messages(row: Mapping[str, Any], *, component: str) -> list[dict[str, str]]:
    """Build a component-specific prompt without hidden evaluator metadata."""

    source_messages = row.get("messages", ())
    if not isinstance(source_messages, list) or len(source_messages) < 2:
        raise ValueError(f"{source_case_id(row)} is missing strict prompt messages")
    source_user = source_messages[1]
    if not isinstance(source_user, Mapping):
        raise ValueError(f"{source_case_id(row)} has malformed user prompt")

    try:
        task = json.loads(str(source_user.get("content", "{}")))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{source_case_id(row)} user prompt is not JSON") from exc
    if not isinstance(task, Mapping):
        raise ValueError(f"{source_case_id(row)} user prompt JSON is not an object")

    target = target_output_from_row(row)
    visible_task: dict[str, Any] = {
        "component": component,
        "input_id": task.get("input_id"),
        "claim": task.get("claim"),
        "allowed_tools": task.get("allowed_tools", []),
        "allowed_actions": list(ALLOWED_ACTIONS),
        "allowed_evidence_statuses": list(ALLOWED_EVIDENCE_STATUSES),
        "required_query_fields": list(REQUIRED_QUERY_FIELDS),
    }
    if component == "routing_after_loop":
        visible_task["observed_tool_loop"] = target["tool_calls"]

    return [
        {
            "role": "system",
            "content": (
                "You are a Stage A component agent. Return exactly one JSON "
                "object for the requested component. Do not include prose, "
                "markdown, evaluator metadata, or hidden labels."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(visible_task, sort_keys=True),
        },
    ]


def target_for_component(row: Mapping[str, Any], *, component: str) -> dict[str, Any]:
    """Project the strict target into a smaller component target."""

    target = target_output_from_row(row)
    if component == "enum_action":
        return {
            "action": target["action"],
            "evidence_status": target["evidence_status"],
        }
    if component == "tool_query":
        return {
            "tool_calls": target["tool_calls"],
        }
    if component == "routing_after_loop":
        return {
            "action": target["action"],
            "evidence_status": target["evidence_status"],
            "cited_source_ids": target["cited_source_ids"],
        }
    raise ValueError(f"Unknown component: {component}")


def prompt_hash(messages: Sequence[Mapping[str, Any]]) -> str:
    payload = json.dumps(messages, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def component_row(row: Mapping[str, Any], *, component: str) -> dict[str, Any]:
    messages = prompt_messages(row, component=component)
    target = target_for_component(row, component=component)
    case_id = source_case_id(row)
    return {
        "id": f"stage_a_strict_component::{case_id}::{component}",
        "dataset": DATASET,
        "component": component,
        "prompt_contract": PROMPT_CONTRACT,
        "source_manifest_case_id": case_id,
        "source_strict_sft_id": row.get("id"),
        "source_task_id": row.get("source_task_id"),
        "task_id": row.get("task_id"),
        "tool_profile": row.get("tool_profile"),
        "case_family": row.get("case_family"),
        "gold_evidence_status": row.get("gold_evidence_status"),
        "expected_terminal_action": row.get("expected_terminal_action"),
        "split_group": row.get("split_group"),
        "generation_prompt_hash": prompt_hash(messages),
        "prompt_messages": messages,
        "target_output": target,
        "target_keys": list(TARGET_KEYS_BY_COMPONENT[component]),
        "oracle_target": True,
        "split": row.get("split"),
    }


def build_component_targets(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        for component in COMPONENTS:
            out.append(component_row(row, component=component))
    return out


def split_component_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    train_case_ids: set[str],
    heldout_case_ids: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train: list[dict[str, Any]] = []
    heldout: list[dict[str, Any]] = []
    for row in rows:
        case_id = str(row["source_manifest_case_id"])
        copied = dict(row)
        if case_id in train_case_ids:
            copied["split"] = "train"
            train.append(copied)
        elif case_id in heldout_case_ids:
            copied["split"] = "heldout"
            heldout.append(copied)
        else:
            raise ValueError(f"Case id {case_id!r} is absent from strict split inputs")
    return train, heldout


def case_ids(rows: Sequence[Mapping[str, Any]]) -> set[str]:
    return {str(row.get("source_manifest_case_id")) for row in rows if row.get("source_manifest_case_id")}


def value_set(rows: Sequence[Mapping[str, Any]], key: str) -> set[str]:
    return {str(row.get(key)) for row in rows if row.get(key)}


def count_by(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(key)) for row in rows).items()))


def display_path(path: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def manifest_for_targets(
    *,
    source_strict_sft: str | Path,
    source_train_sft: str | Path,
    source_heldout_sft: str | Path,
    targets_out: str | Path,
    train_out: str | Path,
    heldout_out: str | Path,
    rows: Sequence[Mapping[str, Any]],
    train_rows: Sequence[Mapping[str, Any]],
    heldout_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    train_case_ids = sorted(case_ids(train_rows))
    heldout_case_ids = sorted(case_ids(heldout_rows))
    train_split_groups = sorted(value_set(train_rows, "split_group"))
    heldout_split_groups = sorted(value_set(heldout_rows, "split_group"))
    train_source_task_ids = sorted(value_set(train_rows, "source_task_id"))
    heldout_source_task_ids = sorted(value_set(heldout_rows, "source_task_id"))
    return {
        "dataset": MANIFEST_DATASET,
        "component_dataset": DATASET,
        "prompt_contract": PROMPT_CONTRACT,
        "components": list(COMPONENTS),
        "target_keys_by_component": {
            key: list(value)
            for key, value in TARGET_KEYS_BY_COMPONENT.items()
        },
        "source_strict_sft": display_path(source_strict_sft),
        "source_train_sft": display_path(source_train_sft),
        "source_heldout_sft": display_path(source_heldout_sft),
        "targets": display_path(targets_out),
        "train_targets": display_path(train_out),
        "heldout_targets": display_path(heldout_out),
        "target_examples": len(rows),
        "train_target_examples": len(train_rows),
        "heldout_target_examples": len(heldout_rows),
        "source_cases": len(case_ids(rows)),
        "train_cases": len(set(train_case_ids)),
        "heldout_cases": len(set(heldout_case_ids)),
        "by_component": count_by(rows, "component"),
        "train_by_component": count_by(train_rows, "component"),
        "heldout_by_component": count_by(heldout_rows, "component"),
        "by_case_family": count_by(rows, "case_family"),
        "by_evidence_status": count_by(rows, "gold_evidence_status"),
        "train_case_ids": train_case_ids,
        "heldout_case_ids": heldout_case_ids,
        "overlap_case_ids": sorted(set(train_case_ids) & set(heldout_case_ids)),
        "train_split_groups": train_split_groups,
        "heldout_split_groups": heldout_split_groups,
        "overlap_split_groups": sorted(set(train_split_groups) & set(heldout_split_groups)),
        "train_source_task_ids": train_source_task_ids,
        "heldout_source_task_ids": heldout_source_task_ids,
        "overlap_source_task_ids": sorted(set(train_source_task_ids) & set(heldout_source_task_ids)),
        "boundary": (
            "Component targets are derived from strict Stage A targets and keep "
            "hidden evaluator labels out of prompt_messages. They are slice "
            "targets for diagnostics, not a claim that prompt-visible evidence "
            "alone is sufficient for scientific routing."
        ),
        "next_decision": (
            "Use these targets to test constrained enum/action decoding, "
            "structured tool-query generation, and routing-after-loop behavior "
            "before any DPO or RLVR escalation."
        ),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-strict-sft", default="post_training/stage_a_strict_contract_sft_v1.jsonl")
    parser.add_argument(
        "--source-train-sft",
        default="post_training/stage_a_strict_contract_sft_train_v1.jsonl",
    )
    parser.add_argument(
        "--source-heldout-sft",
        default="post_training/stage_a_strict_contract_sft_heldout_v1.jsonl",
    )
    parser.add_argument("--targets-out", default="post_training/stage_a_strict_component_targets_v1.jsonl")
    parser.add_argument("--train-out", default="post_training/stage_a_strict_component_targets_train_v1.jsonl")
    parser.add_argument("--heldout-out", default="post_training/stage_a_strict_component_targets_heldout_v1.jsonl")
    parser.add_argument("--manifest-out", default="post_training/stage_a_strict_component_targets_manifest.json")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    source_rows = load_jsonl(args.source_strict_sft)
    train_source_rows = load_jsonl(args.source_train_sft)
    heldout_source_rows = load_jsonl(args.source_heldout_sft)
    rows = build_component_targets(source_rows)
    train_rows, heldout_rows = split_component_rows(
        rows,
        train_case_ids=case_ids(train_source_rows),
        heldout_case_ids=case_ids(heldout_source_rows),
    )
    manifest = manifest_for_targets(
        source_strict_sft=args.source_strict_sft,
        source_train_sft=args.source_train_sft,
        source_heldout_sft=args.source_heldout_sft,
        targets_out=args.targets_out,
        train_out=args.train_out,
        heldout_out=args.heldout_out,
        rows=rows,
        train_rows=train_rows,
        heldout_rows=heldout_rows,
    )
    write_jsonl(args.targets_out, rows)
    write_jsonl(args.train_out, train_rows)
    write_jsonl(args.heldout_out, heldout_rows)
    write_json(args.manifest_out, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    if (
        manifest["overlap_case_ids"]
        or manifest["overlap_split_groups"]
        or manifest["overlap_source_task_ids"]
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

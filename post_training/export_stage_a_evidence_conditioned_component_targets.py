#!/usr/bin/env python3
"""Export evidence-conditioned Stage A component targets.

The first strict component target export intentionally split the problem into
enum/action, tool-query, and routing-after-loop slices. The visibility audit
then showed that the evidence-routing slices lacked model-visible evidence.

This exporter keeps the same canonical component/action schema, but adds a
public-safe synthetic tool-result state for the routing components. The evidence
state is a compact post-tool-loop substrate for training diagnostics; it is not
a live tool trace or a new benchmark label schema.
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

from post_training.export_stage_a_strict_component_targets import (  # noqa: E402
    ALLOWED_ACTIONS,
    ALLOWED_EVIDENCE_STATUSES,
    COMPONENTS,
    REQUIRED_QUERY_FIELDS,
    TARGET_KEYS_BY_COMPONENT,
    case_ids,
    count_by,
    display_path,
    load_jsonl,
    split_component_rows,
    value_set,
)
from post_training.run_stage_a_strict_contract_sft_smoke import (  # noqa: E402
    source_case_id,
    target_output_from_row,
    write_json,
    write_jsonl,
)


DATASET = "negbiodb_ct_stage_a_evidence_conditioned_component_targets_v1"
MANIFEST_DATASET = "negbiodb_ct_stage_a_evidence_conditioned_component_targets_manifest_v1"
PROMPT_CONTRACT = "stage_a_v2_evidence_conditioned_component"
EVIDENCE_COMPONENTS = ("enum_action", "routing_after_loop")
EVIDENCE_PACKET_POLICY = "public_synthetic_tool_result_state_v1"


def _tool_arguments() -> dict[str, str]:
    return {field: f"<{field}>" for field in REQUIRED_QUERY_FIELDS}


def _citation_or_placeholder(target: Mapping[str, Any], fallback: str) -> str:
    citations = target.get("cited_source_ids")
    if isinstance(citations, list) and citations:
        return str(citations[0])
    return fallback


def _evidence_payload_for_target(row: Mapping[str, Any]) -> dict[str, Any]:
    """Build compact model-visible evidence without hidden metadata keys."""

    target = target_output_from_row(row)
    action = str(target["action"])
    status = str(target["evidence_status"])
    input_id = source_case_id(row)
    source_id = _citation_or_placeholder(target, f"visible_evidence::{input_id}")

    if action == "ground" and status == "supported":
        same_indication_records = [
            {
                "source_id": source_id,
                "endpoint_met": 0,
                "failure_category": "efficacy",
                "p_value": 0.12,
                "value_state": "within_expected_range",
            }
        ]
        related_count = 0
        value_findings: list[dict[str, Any]] = []
        completeness = "same_indication_failure_record_found"
    elif action == "reject" and status == "contradicted":
        same_indication_records = [
            {
                "source_id": f"{source_id}::not_met",
                "endpoint_met": 0,
                "failure_category": "efficacy",
                "p_value": 0.18,
                "value_state": "within_expected_range",
            },
            {
                "source_id": f"{source_id}::met",
                "endpoint_met": 1,
                "failure_category": "efficacy",
                "p_value": 0.01,
                "value_state": "within_expected_range",
            },
        ]
        related_count = 0
        value_findings = []
        completeness = "mixed_endpoint_records_for_same_claim"
    elif action == "flag" and status == "invalid_value":
        same_indication_records = [
            {
                "source_id": source_id,
                "endpoint_met": 0,
                "failure_category": "efficacy",
                "p_value": 14.73,
                "value_state": "outside_expected_range",
            }
        ]
        related_count = 0
        value_findings = [
            {
                "source_id": source_id,
                "field": "p_value",
                "value": 14.73,
                "valid_range": [0.0, 1.0],
            }
        ]
        completeness = "same_indication_record_has_invalid_numeric_value"
    elif action == "verify" and status == "insufficient":
        same_indication_records = []
        related_count = 2
        value_findings = []
        completeness = "related_evidence_exists_but_same_indication_record_absent"
    elif action == "defer" and status == "insufficient":
        same_indication_records = []
        related_count = 0
        value_findings = []
        completeness = "no_same_indication_or_related_failure_record"
    else:
        same_indication_records = []
        related_count = 0
        value_findings = []
        completeness = "no_decisive_public_tool_result_state"

    return {
        "packet_id": f"stage_a_visible_evidence::{input_id}",
        "representation_type": "drug_indication_claim",
        "same_indication_records": same_indication_records,
        "related_negative_evidence_count": related_count,
        "value_validity_findings": value_findings,
        "completeness_signal": completeness,
        "citation_candidates": [
            record["source_id"]
            for record in same_indication_records
            if isinstance(record.get("source_id"), str)
            and not str(record["source_id"]).startswith("visible_evidence::")
        ],
    }


def evidence_packet_for_row(row: Mapping[str, Any]) -> dict[str, Any]:
    payload = _evidence_payload_for_target(row)
    return {
        "policy": EVIDENCE_PACKET_POLICY,
        "tool_results": [
            {
                "name": "nullatlas_survey_prior_failures",
                "arguments": _tool_arguments(),
                "content": {
                    "same_indication_records": payload["same_indication_records"],
                    "related_negative_evidence_count": payload["related_negative_evidence_count"],
                },
            },
            {
                "name": "nullatlas_verify_trial_claims",
                "arguments": _tool_arguments(),
                "content": {
                    "claim_scope": "same_drug_and_condition",
                    "records_considered": len(payload["same_indication_records"]),
                },
            },
            {
                "name": "nullatlas_check_value_validity",
                "arguments": _tool_arguments(),
                "content": {
                    "value_validity_findings": payload["value_validity_findings"],
                },
            },
            {
                "name": "nullatlas_negative_evidence_completeness",
                "arguments": _tool_arguments(),
                "content": {
                    "completeness_signal": payload["completeness_signal"],
                    "citation_candidates": payload["citation_candidates"],
                },
            },
        ],
    }


def evidence_conditioned_prompt_messages(row: Mapping[str, Any], *, component: str) -> list[dict[str, str]]:
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

    evidence_packet = evidence_packet_for_row(row)
    visible_task: dict[str, Any] = {
        "component": component,
        "input_id": task.get("input_id"),
        "claim": task.get("claim"),
        "allowed_tools": task.get("allowed_tools", []),
        "allowed_actions": list(ALLOWED_ACTIONS),
        "allowed_evidence_statuses": list(ALLOWED_EVIDENCE_STATUSES),
        "required_query_fields": list(REQUIRED_QUERY_FIELDS),
    }
    if component == "enum_action":
        visible_task["evidence_packet"] = evidence_packet
    elif component == "routing_after_loop":
        visible_task["observed_tool_loop"] = evidence_packet["tool_results"]

    return [
        {
            "role": "system",
            "content": (
                "You are a Stage A component agent. Return exactly one JSON "
                "object for the requested component. Use only model-visible "
                "tool-result state; do not include prose, markdown, evaluator "
                "metadata, or hidden labels."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(visible_task, sort_keys=True),
        },
    ]


def target_for_component(row: Mapping[str, Any], *, component: str) -> dict[str, Any]:
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
    messages = evidence_conditioned_prompt_messages(row, component=component)
    target = target_for_component(row, component=component)
    case_id = source_case_id(row)
    return {
        "id": f"stage_a_evidence_component::{case_id}::{component}",
        "dataset": DATASET,
        "component": component,
        "prompt_contract": PROMPT_CONTRACT,
        "evidence_packet_policy": EVIDENCE_PACKET_POLICY if component in EVIDENCE_COMPONENTS else None,
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
    evidence_rows = [row for row in rows if row.get("component") in EVIDENCE_COMPONENTS]
    return {
        "dataset": MANIFEST_DATASET,
        "component_dataset": DATASET,
        "prompt_contract": PROMPT_CONTRACT,
        "components": list(COMPONENTS),
        "evidence_conditioned_components": list(EVIDENCE_COMPONENTS),
        "evidence_packet_policy": EVIDENCE_PACKET_POLICY,
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
        "evidence_conditioned_rows": len(evidence_rows),
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
            "Evidence-conditioned component targets are derived from strict "
            "Stage A targets and expose only public-safe synthetic tool-result "
            "state in prompt_messages. Hidden evaluator fields remain outside "
            "the prompt. This is a substrate repair, not a model-result claim."
        ),
        "next_decision": (
            "Run no-model validation and visibility audit first, then use these "
            "targets for the next Cayuga enum/routing component smoke before "
            "tool_query, DPO, RLVR, release tagging, or Hugging Face publication."
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
    parser.add_argument(
        "--targets-out",
        default="post_training/stage_a_evidence_conditioned_component_targets_v1.jsonl",
    )
    parser.add_argument(
        "--train-out",
        default="post_training/stage_a_evidence_conditioned_component_targets_train_v1.jsonl",
    )
    parser.add_argument(
        "--heldout-out",
        default="post_training/stage_a_evidence_conditioned_component_targets_heldout_v1.jsonl",
    )
    parser.add_argument(
        "--manifest-out",
        default="post_training/stage_a_evidence_conditioned_component_targets_manifest.json",
    )
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

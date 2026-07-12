#!/usr/bin/env python3
"""Evidence-derived boundary rationale utilities for native CT tool traces."""

from __future__ import annotations

from typing import Any, Mapping

from post_training.build_sft_boundary_rationale_data import BOUNDARY_NEGATIVES


def search_failures_content(row: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    for message in row["messages"]:
        if message.get("role") == "tool" and message.get("name") == "search_failures":
            content = message.get("content")
            if isinstance(content, list):
                return [item for item in content if isinstance(item, Mapping)]
    return []


def other_indication_failure_count(row: Mapping[str, Any]) -> int:
    for message in row["messages"]:
        if message.get("role") != "tool" or message.get("name") != "check_other_indications":
            continue
        content = message.get("content")
        if isinstance(content, Mapping):
            return int(content.get("failures_for_other_indications") or 0)
    return 0


def invalid_p_value(record: Mapping[str, Any]) -> bool:
    p_value = record.get("p_value")
    if p_value is None:
        return False
    try:
        value = float(p_value)
    except (TypeError, ValueError):
        return True
    return value < 0 or value > 1


def clean_efficacy_failure(record: Mapping[str, Any]) -> bool:
    if record.get("failure_category") != "efficacy":
        return False
    if invalid_p_value(record):
        return False
    endpoint = record.get("endpoint_met")
    return endpoint in {0, None}


def first_invalid_p_value_nct(row: Mapping[str, Any]) -> str | None:
    for record in search_failures_content(row):
        if invalid_p_value(record) and record.get("nct"):
            return str(record["nct"])
    return None


def first_clean_efficacy_failure_nct(row: Mapping[str, Any]) -> str | None:
    for record in search_failures_content(row):
        if clean_efficacy_failure(record) and record.get("nct"):
            return str(record["nct"])
    return None


def evidence_action(row: Mapping[str, Any]) -> str:
    failures = search_failures_content(row)
    other_failures = other_indication_failure_count(row)
    endpoint_values = {record.get("endpoint_met") for record in failures}
    if 0 in endpoint_values and 1 in endpoint_values:
        return "reject"
    if any(invalid_p_value(record) for record in failures):
        return "flag"
    if any(clean_efficacy_failure(record) for record in failures):
        return "ground"
    if other_failures > 0:
        return "verify"
    return "defer"


def evidence_decision(row: Mapping[str, Any]) -> dict[str, str | None]:
    action = evidence_action(row)
    cited_nct = None
    if action == "flag":
        cited_nct = first_invalid_p_value_nct(row)
    elif action == "ground":
        cited_nct = first_clean_efficacy_failure_nct(row)
    return {"action": action, "cited_nct": cited_nct}


def evidence_rationale(row: Mapping[str, Any]) -> str:
    failures = search_failures_content(row)
    other_failures = other_indication_failure_count(row)
    action = evidence_action(row)
    if action == "defer":
        return (
            "Choose defer because search_failures returned no target-indication "
            "failure rows and check_other_indications returned 0 related failures. "
            "Do not choose verify when other-indication failures are also absent."
        )
    if action == "verify":
        return (
            "Choose verify because search_failures returned no target-indication "
            f"efficacy failure, while check_other_indications returned {other_failures} "
            "related failures. Do not choose defer when related failures exist."
        )
    if action == "reject":
        return (
            "Choose reject because returned rows contain both endpoint_met=0 and "
            "endpoint_met=1 evidence for this drug-indication pair. Mixed endpoint "
            "evidence overrides single-row ground or flag support."
        )
    if action == "flag":
        ncts = [str(record.get("nct")) for record in failures if invalid_p_value(record) and record.get("nct")]
        nct_text = f" in {', '.join(ncts)}" if ncts else ""
        return (
            "Choose flag because a returned target-indication record"
            f"{nct_text} has an impossible p_value outside [0,1]. "
            "Do not choose ground for impossible evidence."
        )
    ncts = [str(record.get("nct")) for record in failures if clean_efficacy_failure(record) and record.get("nct")]
    nct_text = f" in {', '.join(ncts)}" if ncts else ""
    return (
        "Choose ground because the returned target-indication rows include a clean "
        f"efficacy failure{nct_text}. Do not choose flag without an impossible value, "
        "and do not choose reject without mixed endpoint evidence."
    )


def evidence_rationale_message(
    row: Mapping[str, Any],
    *,
    action_hint_label: str = "Correct final action",
) -> dict[str, str]:
    action = evidence_action(row)
    return {
        "role": "user",
        "content": f"BOUNDARY_RATIONALE: {evidence_rationale(row)} {action_hint_label}: {action}.",
    }


def evidence_rationale_copy(
    row: Mapping[str, Any],
    *,
    dataset: str,
    pair_index: int,
    strategy: str,
    action_hint_label: str = "Correct final action",
) -> dict[str, Any]:
    messages = list(row["messages"])
    if messages[-1].get("tool_call", {}).get("name") != "submit_decision":
        raise ValueError(f"Expected final submit_decision for {row['id']}")
    action = evidence_action(row)
    action_class = row.get("action_class")
    out = dict(row)
    out["id"] = f"{row['id']}::boundary_rationale::evidence::{pair_index}"
    out["dataset"] = dataset
    out["source_example_id"] = row["id"]
    out["boundary_strategy"] = strategy
    out["boundary_pair_role"] = "evidence_rationale"
    out["boundary_pair_index"] = pair_index
    out["boundary_negative_actions"] = list(BOUNDARY_NEGATIVES[action])
    out["boundary_rationale"] = evidence_rationale(row)
    out["evidence_derived_action"] = action
    out["evidence_matches_action_class"] = None if action_class is None else action == str(action_class)
    out["messages"] = messages[:-1] + [
        evidence_rationale_message(row, action_hint_label=action_hint_label),
        messages[-1],
    ]
    return out

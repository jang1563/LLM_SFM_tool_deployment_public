#!/usr/bin/env python3
"""Build a public Stage A real-query development and perturbation slice.

The source tasks are already public and were declared exclusions when the
private sealed extension was committed. This builder never reads the sealed
manifest. It selects unused public cases, exposes typed drug/condition query
identifiers, creates exact tool-call targets, and mutates synthetic tool-result
state for runtime-enforcement diagnostics.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from negbiodb_ct.adapter import load_task_records  # noqa: E402
from negbiodb_ct.stage_a_manifest import (  # noqa: E402
    ACTION_CLASS_ORDER,
    load_stage_a_manifest,
    model_visible_query_for_record,
    split_group_for_record,
    stage_a_row_from_task_record,
    validate_stage_a_manifest,
    write_stage_a_manifest,
)
from post_training.build_stage_a_sealed_extension import (  # noqa: E402
    manifest_claim,
    normalize_claim,
    task_claim,
)
from post_training.export_stage_a_saved_output_evidence_candidate_routing_rows import (  # noqa: E402
    CANDIDATE_PAIRS,
    candidate_outputs,
)
from post_training.export_stage_a_strict_component_targets import (  # noqa: E402
    ALLOWED_ACTIONS,
    ALLOWED_EVIDENCE_STATUSES,
)


BASE_DATASET = "negbiodb_ct_stage_a_prospective_real_query_v1"
TOOL_QUERY_DATASET = "negbiodb_ct_stage_a_prospective_real_query_tool_query_v1"
ROUTING_DATASET = (
    "negbiodb_ct_stage_a_prospective_real_query_routing_perturbations_v1"
)
MANIFEST_DATASET = "negbiodb_ct_stage_a_prospective_real_query_manifest_v1"
PROMPT_CONTRACT = "stage_a_prospective_real_query_v1"
SELECTION_SEED = 20260723
REQUIRED_QUERY_FIELDS = ("drug_id", "condition_id")
PERTURBATIONS = (
    "clean",
    "missing_attribution",
    "stale_source",
    "contradiction",
    "invalid_numeric_value",
    "partial_query",
    "wrong_tool",
    "unavailable_tool",
)
PAIR_BY_TERMINAL_STATUS = {
    ("ground_with_attribution", "supported"): "ground/supported",
    ("reject_or_flag_unsupported_claim", "contradicted"): "reject/contradicted",
    ("defer_or_request_more_evidence", "insufficient"): "defer/insufficient",
    ("verify_with_assay_or_database", "insufficient"): "verify/insufficient",
    ("reject_or_flag_unsupported_claim", "invalid_value"): "flag/invalid_value",
}
PAIR_BY_PERTURBATION = {
    "missing_attribution": "verify/insufficient",
    "stale_source": "verify/insufficient",
    "contradiction": "reject/contradicted",
    "invalid_numeric_value": "flag/invalid_value",
    "partial_query": "defer/insufficient",
    "wrong_tool": "defer/insufficient",
    "unavailable_tool": "defer/insufficient",
}
PROMPT_FORBIDDEN_KEYS = (
    "hidden_eval_metadata",
    "source_task_id",
    "split_group",
    "target_output",
    "target_pair",
    "perturbation",
    "action_class",
    "gold_evidence_status",
    "expected_terminal_action",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )


def load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def display_path(path: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.name


def query_arguments(row: Mapping[str, Any]) -> dict[str, Any]:
    visible = row.get("model_visible_task")
    if not isinstance(visible, Mapping):
        raise ValueError("base row lacks model_visible_task")
    query = visible.get("query")
    if not isinstance(query, Mapping):
        raise ValueError("base row lacks model-visible query")
    arguments: dict[str, Any] = {}
    for field in REQUIRED_QUERY_FIELDS:
        field_payload = query.get(field)
        if not isinstance(field_payload, Mapping) or "value" not in field_payload:
            raise ValueError(f"base row query lacks {field}")
        arguments[field] = field_payload["value"]
    return arguments


def exclusion_sets(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, set[str]]:
    source_task_ids: set[str] = set()
    split_groups: set[str] = set()
    claims: set[str] = set()
    for row in rows:
        hidden = row.get("hidden_eval_metadata")
        if isinstance(hidden, Mapping):
            source_task_id = hidden.get("source_task_id")
            split_group = hidden.get("split_group")
            if source_task_id:
                source_task_ids.add(str(source_task_id))
            if split_group:
                split_groups.add(str(split_group))
        claim = manifest_claim(row)
        if claim:
            claims.add(claim)
    return {
        "source_task_ids": source_task_ids,
        "split_groups": split_groups,
        "claims": claims,
    }


def select_records(
    records: Sequence[Mapping[str, Any]],
    *,
    excluded_rows: Sequence[Mapping[str, Any]],
    per_action: int,
    seed: int,
) -> list[Mapping[str, Any]]:
    if per_action <= 0:
        raise ValueError("per_action must be positive")
    excluded = exclusion_sets(excluded_rows)
    buckets: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in sorted(records, key=lambda item: str(item.get("packet_id"))):
        action = str(record.get("action_class") or "")
        packet_id = str(record.get("packet_id") or "")
        claim = task_claim(record)
        if action not in ACTION_CLASS_ORDER or not packet_id or not claim:
            continue
        split_group = split_group_for_record(record)
        if (
            packet_id in excluded["source_task_ids"]
            or split_group in excluded["split_groups"]
            or claim in excluded["claims"]
        ):
            continue
        model_visible_query_for_record(record)
        buckets[action].append(record)

    selected: list[Mapping[str, Any]] = []
    selected_ids: set[str] = set()
    selected_groups: set[str] = set()
    selected_claims: set[str] = set()
    for action in ACTION_CLASS_ORDER:
        candidates = list(buckets[action])
        random.Random(f"{seed}:{action}").shuffle(candidates)
        chosen = 0
        for record in candidates:
            packet_id = str(record["packet_id"])
            split_group = split_group_for_record(record)
            claim = task_claim(record)
            if (
                packet_id in selected_ids
                or split_group in selected_groups
                or claim in selected_claims
            ):
                continue
            selected.append(record)
            selected_ids.add(packet_id)
            selected_groups.add(split_group)
            selected_claims.add(claim)
            chosen += 1
            if chosen == per_action:
                break
        if chosen != per_action:
            raise ValueError(
                f"insufficient unused public candidates for {action}: "
                f"{chosen}<{per_action}"
            )
    return selected


def build_base_rows(
    selected: Sequence[Mapping[str, Any]],
    *,
    per_action: int,
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(selected):
        row = stage_a_row_from_task_record(
            record,
            case_index=index,
            include_query_values=True,
        )
        case_id = f"stage_a_prospective_query_v1::{index:06d}"
        row["case_id"] = case_id
        row["dataset"] = BASE_DATASET
        row["model_visible_task"]["input_id"] = case_id
        row["hidden_eval_metadata"]["development_only"] = True
        row["hidden_eval_metadata"]["independent_test_claimed"] = False
        row["hidden_eval_metadata"]["selection_seed"] = seed
        rows.append(row)
    issues = validate_stage_a_manifest(
        rows,
        min_rows=per_action * len(ACTION_CLASS_ORDER),
        min_status_count=per_action,
    )
    if issues:
        raise ValueError("base manifest validation failed: " + "; ".join(issues))
    return rows


def source_id_for_row(row: Mapping[str, Any]) -> str:
    hidden = row["hidden_eval_metadata"]
    sources = hidden.get("gold_source_ids")
    if isinstance(sources, list) and sources:
        return str(sources[0])
    source_task_id = str(hidden["source_task_id"])
    token = hashlib.sha256(source_task_id.encode()).hexdigest()[:16]
    return f"public_task_source::{token}"


def clean_evidence_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    hidden = row["hidden_eval_metadata"]
    status = str(hidden["gold_evidence_status"])
    terminal = str(hidden["expected_terminal_action"])
    source_id = source_id_for_row(row)
    records: list[dict[str, Any]] = []
    related_count = 0
    findings: list[dict[str, Any]] = []
    citations: list[str] = []

    if terminal == "ground_with_attribution" and status == "supported":
        records = [
            {
                "source_id": source_id,
                "source_freshness": "current",
                "endpoint_met": 0,
                "failure_category": "efficacy",
                "p_value": 0.12,
                "value_state": "within_expected_range",
            }
        ]
        completeness = "same_indication_failure_record_found"
        citations = [source_id]
    elif terminal == "reject_or_flag_unsupported_claim" and status == "contradicted":
        records = [
            {
                "source_id": source_id,
                "source_freshness": "current",
                "endpoint_met": 0,
                "failure_category": "efficacy",
                "p_value": 0.18,
                "value_state": "within_expected_range",
            },
            {
                "source_id": source_id,
                "source_freshness": "current",
                "endpoint_met": 1,
                "failure_category": "efficacy",
                "p_value": 0.01,
                "value_state": "within_expected_range",
            },
        ]
        completeness = "mixed_endpoint_records_for_same_claim"
    elif terminal == "reject_or_flag_unsupported_claim" and status == "invalid_value":
        records = [
            {
                "source_id": source_id,
                "source_freshness": "current",
                "endpoint_met": 0,
                "failure_category": "efficacy",
                "p_value": 14.73,
                "value_state": "outside_expected_range",
            }
        ]
        findings = [
            {
                "source_id": source_id,
                "field": "p_value",
                "value": 14.73,
                "valid_range": [0.0, 1.0],
            }
        ]
        completeness = "same_indication_record_has_invalid_numeric_value"
        citations = [source_id]
    elif terminal == "verify_with_assay_or_database" and status == "insufficient":
        related_count = 2
        completeness = "related_evidence_exists_but_same_indication_record_absent"
    else:
        completeness = "no_same_indication_or_related_failure_record"

    return {
        "same_indication_records": records,
        "related_negative_evidence_count": related_count,
        "value_validity_findings": findings,
        "completeness_signal": completeness,
        "citation_candidates": citations,
    }


def tool_loop_for_payload(
    row: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> list[dict[str, Any]]:
    arguments = query_arguments(row)
    records = copy.deepcopy(payload["same_indication_records"])
    return [
        {
            "name": "nullatlas_survey_prior_failures",
            "arguments": dict(arguments),
            "content": {
                "same_indication_records": records,
                "related_negative_evidence_count": payload[
                    "related_negative_evidence_count"
                ],
            },
        },
        {
            "name": "nullatlas_verify_trial_claims",
            "arguments": dict(arguments),
            "content": {
                "claim_scope": "same_drug_and_condition",
                "records_considered": len(records),
            },
        },
        {
            "name": "nullatlas_check_value_validity",
            "arguments": dict(arguments),
            "content": {
                "value_validity_findings": copy.deepcopy(
                    payload["value_validity_findings"]
                ),
            },
        },
        {
            "name": "nullatlas_negative_evidence_completeness",
            "arguments": dict(arguments),
            "content": {
                "completeness_signal": payload["completeness_signal"],
                "citation_candidates": list(payload["citation_candidates"]),
            },
        },
    ]


def tool_content(
    tool_loop: Sequence[Mapping[str, Any]],
    tool_name: str,
) -> dict[str, Any]:
    for item in tool_loop:
        content = item.get("content")
        if item.get("name") == tool_name and isinstance(content, dict):
            return content
    raise ValueError(f"tool loop lacks {tool_name}")


def evidence_records(tool_loop: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    survey = tool_content(tool_loop, "nullatlas_survey_prior_failures")
    records = survey.get("same_indication_records")
    if not isinstance(records, list):
        return []
    return [dict(record) for record in records if isinstance(record, Mapping)]


def mutation_source_id(row: Mapping[str, Any], perturbation: str) -> str:
    token = hashlib.sha256(f"{row['case_id']}:{perturbation}".encode()).hexdigest()[:16]
    return f"public_mutation_source::{token}"


def mutate_tool_loop(
    row: Mapping[str, Any],
    clean_loop: Sequence[Mapping[str, Any]],
    *,
    perturbation: str,
) -> list[dict[str, Any]]:
    loop = copy.deepcopy(list(clean_loop))
    if perturbation == "clean":
        return loop
    if perturbation == "missing_attribution":
        for record in evidence_records(loop):
            record.pop("source_id", None)
        survey = tool_content(loop, "nullatlas_survey_prior_failures")
        survey["same_indication_records"] = [
            {key: value for key, value in record.items() if key != "source_id"}
            for record in survey.get("same_indication_records", [])
            if isinstance(record, Mapping)
        ]
        completeness = tool_content(
            loop, "nullatlas_negative_evidence_completeness"
        )
        completeness["citation_candidates"] = []
        return loop
    if perturbation == "stale_source":
        survey = tool_content(loop, "nullatlas_survey_prior_failures")
        for record in survey.get("same_indication_records", []):
            if isinstance(record, dict):
                record["source_freshness"] = "stale"
        return loop
    if perturbation in {"contradiction", "invalid_numeric_value"}:
        source_id = mutation_source_id(row, perturbation)
        if perturbation == "contradiction":
            payload = {
                "same_indication_records": [
                    {
                        "source_id": source_id,
                        "source_freshness": "current",
                        "endpoint_met": 0,
                        "failure_category": "efficacy",
                        "p_value": 0.21,
                        "value_state": "within_expected_range",
                    },
                    {
                        "source_id": source_id,
                        "source_freshness": "current",
                        "endpoint_met": 1,
                        "failure_category": "efficacy",
                        "p_value": 0.02,
                        "value_state": "within_expected_range",
                    },
                ],
                "related_negative_evidence_count": 0,
                "value_validity_findings": [],
                "completeness_signal": "mixed_endpoint_records_for_same_claim",
                "citation_candidates": [],
            }
        else:
            payload = {
                "same_indication_records": [
                    {
                        "source_id": source_id,
                        "source_freshness": "current",
                        "endpoint_met": 0,
                        "failure_category": "efficacy",
                        "p_value": 14.73,
                        "value_state": "outside_expected_range",
                    }
                ],
                "related_negative_evidence_count": 0,
                "value_validity_findings": [
                    {
                        "source_id": source_id,
                        "field": "p_value",
                        "value": 14.73,
                        "valid_range": [0.0, 1.0],
                    }
                ],
                "completeness_signal": (
                    "same_indication_record_has_invalid_numeric_value"
                ),
                "citation_candidates": [source_id],
            }
        return tool_loop_for_payload(row, payload)
    if perturbation == "partial_query":
        for tool_result in loop:
            arguments = tool_result.get("arguments")
            if isinstance(arguments, dict):
                arguments.pop("condition_id", None)
        return loop
    if perturbation == "wrong_tool":
        loop[0]["name"] = "nullatlas_unapproved_lookup"
        return loop
    if perturbation == "unavailable_tool":
        for tool_result in loop:
            if tool_result.get("name") == "nullatlas_verify_trial_claims":
                tool_result["content"] = {
                    "error": "tool_unavailable",
                    "retryable": True,
                }
                break
        return loop
    raise ValueError(f"unknown perturbation: {perturbation}")


def legacy_visible_features(tool_loop: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_name = {
        str(item.get("name")): item.get("content")
        for item in tool_loop
        if isinstance(item, Mapping) and isinstance(item.get("content"), Mapping)
    }
    survey = by_name.get("nullatlas_survey_prior_failures", {})
    verifier = by_name.get("nullatlas_verify_trial_claims", {})
    validity = by_name.get("nullatlas_check_value_validity", {})
    completeness = by_name.get("nullatlas_negative_evidence_completeness", {})
    records = survey.get("same_indication_records")
    citations = completeness.get("citation_candidates")
    findings = validity.get("value_validity_findings")
    return {
        "observed_tool_loop_present": bool(tool_loop),
        "related_negative_evidence_count": survey.get(
            "related_negative_evidence_count"
        ),
        "same_indication_record_count": (
            len(records) if isinstance(records, list) else None
        ),
        "records_considered": verifier.get("records_considered"),
        "citation_candidates": (
            [str(value) for value in citations]
            if isinstance(citations, list)
            else []
        ),
        "completeness_signal": completeness.get("completeness_signal"),
        "value_validity_findings": (
            list(findings) if isinstance(findings, list) else []
        ),
    }


def build_tool_query_rows(
    base_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for base in base_rows:
        visible = base["model_visible_task"]
        hidden = base["hidden_eval_metadata"]
        arguments = query_arguments(base)
        tool_calls = [
            {"name": str(tool), "arguments": dict(arguments)}
            for tool in hidden["required_tools"]
        ]
        rows.append(
            {
                "id": f"stage_a_prospective_tool_query::{base['case_id']}",
                "dataset": TOOL_QUERY_DATASET,
                "component": "tool_query",
                "prompt_contract": PROMPT_CONTRACT,
                "source_manifest_case_id": base["case_id"],
                "model_visible_task": {
                    "component": "tool_query",
                    "input_id": visible["input_id"],
                    "claim": visible["claim"],
                    "query": copy.deepcopy(visible["query"]),
                    "allowed_tools": list(visible["allowed_tools"]),
                    "allowed_actions": list(ALLOWED_ACTIONS),
                    "allowed_evidence_statuses": list(
                        ALLOWED_EVIDENCE_STATUSES
                    ),
                    "required_query_fields": list(REQUIRED_QUERY_FIELDS),
                },
                "target_output": {"tool_calls": tool_calls},
                "hidden_eval_metadata": {
                    "source_task_id": hidden["source_task_id"],
                    "split_group": hidden["split_group"],
                    "development_only": True,
                    "independent_test_claimed": False,
                },
            }
        )
    return rows


def target_for_perturbation(
    base: Mapping[str, Any],
    *,
    perturbation: str,
) -> dict[str, Any]:
    hidden = base["hidden_eval_metadata"]
    key = (
        str(hidden["expected_terminal_action"]),
        str(hidden["gold_evidence_status"]),
    )
    if key not in PAIR_BY_TERMINAL_STATUS:
        raise ValueError(f"unsupported terminal/status pair: {key}")
    base_pair = PAIR_BY_TERMINAL_STATUS[key]
    pair = base_pair if perturbation == "clean" else PAIR_BY_PERTURBATION[perturbation]
    action, status = pair.split("/", maxsplit=1)
    citations: list[str] = []
    if perturbation == "clean" and action in {"ground", "flag"}:
        citations = [source_id_for_row(base)]
    elif perturbation == "invalid_numeric_value":
        citations = [mutation_source_id(base, perturbation)]
    return {
        "selected_pair": pair,
        "action": action,
        "evidence_status": status,
        "cited_source_ids": citations,
    }


def build_routing_rows(
    base_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for base in base_rows:
        visible = base["model_visible_task"]
        hidden = base["hidden_eval_metadata"]
        clean_loop = tool_loop_for_payload(base, clean_evidence_payload(base))
        has_records = bool(evidence_records(clean_loop))
        for perturbation in PERTURBATIONS:
            if perturbation in {"missing_attribution", "stale_source"} and not has_records:
                continue
            loop = mutate_tool_loop(
                base,
                clean_loop,
                perturbation=perturbation,
            )
            target = target_for_perturbation(base, perturbation=perturbation)
            model_visible_task = {
                "component": "saved_output_evidence_candidate_routing",
                "input_id": visible["input_id"],
                "claim": visible["claim"],
                "query": copy.deepcopy(visible["query"]),
                "observed_tool_loop": loop,
                "candidate_pairs": list(CANDIDATE_PAIRS),
                "visible_evidence_features": legacy_visible_features(loop),
                "instruction": (
                    "Select exactly one candidate pair using only the observed "
                    "tool-loop state and visible evidence features."
                ),
            }
            perturbation_token = perturbation.replace("_", "-")
            rows.append(
                {
                    "id": (
                        f"stage_a_prospective_routing::{base['case_id']}::"
                        f"{perturbation_token}"
                    ),
                    "dataset": ROUTING_DATASET,
                    "component": "routing_after_loop",
                    "prompt_contract": PROMPT_CONTRACT,
                    "source_manifest_case_id": base["case_id"],
                    "model_visible_task": model_visible_task,
                    "candidate_outputs": candidate_outputs(),
                    "target_output": target,
                    "target_pair": target["selected_pair"],
                    "hidden_eval_metadata": {
                        "source_task_id": hidden["source_task_id"],
                        "split_group": hidden["split_group"],
                        "perturbation": perturbation,
                        "base_pair": target_for_perturbation(
                            base, perturbation="clean"
                        )["selected_pair"],
                        "development_only": True,
                        "independent_test_claimed": False,
                    },
                }
            )
    return rows


def validate_rows(
    *,
    base_rows: Sequence[Mapping[str, Any]],
    tool_query_rows: Sequence[Mapping[str, Any]],
    routing_rows: Sequence[Mapping[str, Any]],
    excluded_rows: Sequence[Mapping[str, Any]],
    per_action: int,
) -> list[str]:
    issues: list[str] = []
    expected_base_rows = per_action * len(ACTION_CLASS_ORDER)
    if len(base_rows) != expected_base_rows:
        issues.append(f"base_row_count:{len(base_rows)}!={expected_base_rows}")
    excluded = exclusion_sets(excluded_rows)
    base_sources = {
        str(row["hidden_eval_metadata"]["source_task_id"]) for row in base_rows
    }
    base_groups = {
        str(row["hidden_eval_metadata"]["split_group"]) for row in base_rows
    }
    base_claims = {
        normalize_claim(row["model_visible_task"]["claim"]) for row in base_rows
    }
    if base_sources & excluded["source_task_ids"]:
        issues.append("source_task_id_overlap_with_existing_stage_a")
    if base_groups & excluded["split_groups"]:
        issues.append("split_group_overlap_with_existing_stage_a")
    if base_claims & excluded["claims"]:
        issues.append("normalized_claim_overlap_with_existing_stage_a")
    if len(base_sources) != len(base_rows):
        issues.append("duplicate_source_task_ids")
    if len(base_groups) != len(base_rows):
        issues.append("duplicate_split_groups")
    if len(base_claims) != len(base_rows):
        issues.append("duplicate_normalized_claims")

    for row in base_rows:
        query = row["model_visible_task"].get("query")
        if not isinstance(query, Mapping):
            issues.append(f"{row['case_id']}:missing_query")
            continue
        for field in REQUIRED_QUERY_FIELDS:
            payload = query.get(field)
            value = payload.get("value") if isinstance(payload, Mapping) else None
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                issues.append(f"{row['case_id']}:invalid_{field}")

    if len(tool_query_rows) != len(base_rows):
        issues.append("tool_query_row_count_mismatch")
    target_hashes: set[str] = set()
    for row in tool_query_rows:
        visible_text = json.dumps(row["model_visible_task"], sort_keys=True)
        if "<drug_id>" in visible_text or "<condition_id>" in visible_text:
            issues.append(f"{row['id']}:visible_placeholder")
        for key in PROMPT_FORBIDDEN_KEYS:
            if key in visible_text:
                issues.append(f"{row['id']}:prompt_leaks_{key}")
        query = row["model_visible_task"]["query"]
        expected_arguments = {
            field: query[field]["value"] for field in REQUIRED_QUERY_FIELDS
        }
        calls = row["target_output"].get("tool_calls")
        if not isinstance(calls, list) or len(calls) != 4:
            issues.append(f"{row['id']}:tool_call_count")
            continue
        for call in calls:
            if call.get("arguments") != expected_arguments:
                issues.append(f"{row['id']}:tool_call_query_mismatch")
        target_hashes.add(canonical_sha256(row["target_output"]))
    if len(target_hashes) != len(tool_query_rows):
        issues.append("tool_query_targets_not_case_specific")

    perturbation_counts: Counter[str] = Counter()
    for row in routing_rows:
        hidden = row.get("hidden_eval_metadata")
        perturbation = (
            str(hidden.get("perturbation")) if isinstance(hidden, Mapping) else ""
        )
        perturbation_counts[perturbation] += 1
        visible_text = json.dumps(row["model_visible_task"], sort_keys=True)
        for key in PROMPT_FORBIDDEN_KEYS:
            if key in visible_text:
                issues.append(f"{row['id']}:prompt_leaks_{key}")
        target = row.get("target_output")
        if not isinstance(target, Mapping):
            issues.append(f"{row['id']}:missing_target")
            continue
        if target.get("selected_pair") not in CANDIDATE_PAIRS:
            issues.append(f"{row['id']}:invalid_target_pair")
        if target.get("selected_pair") != (
            f"{target.get('action')}/{target.get('evidence_status')}"
        ):
            issues.append(f"{row['id']}:target_pair_field_mismatch")

    expected_mutation_counts = {
        "clean": expected_base_rows,
        "missing_attribution": per_action * 3,
        "stale_source": per_action * 3,
        "contradiction": expected_base_rows,
        "invalid_numeric_value": expected_base_rows,
        "partial_query": expected_base_rows,
        "wrong_tool": expected_base_rows,
        "unavailable_tool": expected_base_rows,
    }
    if dict(sorted(perturbation_counts.items())) != expected_mutation_counts:
        issues.append(
            "perturbation_counts_mismatch:"
            f"{dict(sorted(perturbation_counts.items()))}"
        )
    return sorted(set(issues))


def validate_sealed_separation_commitment(
    commitment: Mapping[str, Any],
    *,
    source_tasks_path: str | Path,
) -> list[str]:
    issues: list[str] = []
    declared_exclusions = (
        commitment.get("input_artifacts", {}).get("public_exclusions", [])
    )
    source_display = display_path(source_tasks_path)
    source_sha = sha256_file(source_tasks_path)
    matching = [
        item
        for item in declared_exclusions
        if isinstance(item, Mapping) and item.get("path") == source_display
    ]
    if len(matching) != 1:
        issues.append("source_tasks_not_declared_sealed_exclusion")
    elif matching[0].get("sha256") != source_sha:
        issues.append("source_tasks_hash_changed_since_sealed_commitment")
    overlap = commitment.get("overlap_checks", {})
    for key in (
        "source_task_id_overlap",
        "split_group_overlap",
        "normalized_claim_overlap",
    ):
        if overlap.get(key) != 0:
            issues.append(f"sealed_commitment_nonzero_{key}")
    return issues


def build_manifest(
    *,
    source_tasks_path: str | Path,
    excluded_manifest_path: str | Path,
    sealed_commitment_path: str | Path,
    base_out: str | Path,
    tool_query_out: str | Path,
    routing_out: str | Path,
    base_rows: Sequence[Mapping[str, Any]],
    tool_query_rows: Sequence[Mapping[str, Any]],
    routing_rows: Sequence[Mapping[str, Any]],
    seed: int,
    per_action: int,
    issues: Sequence[str],
) -> dict[str, Any]:
    action_counts = Counter(
        str(row["hidden_eval_metadata"]["expected_terminal_action"])
        for row in base_rows
    )
    perturbation_counts = Counter(
        str(row["hidden_eval_metadata"]["perturbation"]) for row in routing_rows
    )
    target_counts = Counter(str(row["target_pair"]) for row in routing_rows)
    selected_commitment = canonical_sha256(
        [
            {
                "source_task_id": row["hidden_eval_metadata"]["source_task_id"],
                "split_group": row["hidden_eval_metadata"]["split_group"],
                "query": row["model_visible_task"]["query"],
            }
            for row in base_rows
        ]
    )
    return {
        "dataset": MANIFEST_DATASET,
        "base_dataset": BASE_DATASET,
        "tool_query_dataset": TOOL_QUERY_DATASET,
        "routing_dataset": ROUTING_DATASET,
        "prompt_contract": PROMPT_CONTRACT,
        "selection": {
            "seed": seed,
            "per_action": per_action,
            "base_rows": len(base_rows),
            "selection_commitment_sha256": selected_commitment,
            "development_only": True,
            "independent_test_claimed": False,
        },
        "counts": {
            "base_rows": len(base_rows),
            "tool_query_rows": len(tool_query_rows),
            "routing_rows": len(routing_rows),
            "base_by_terminal_action": dict(sorted(action_counts.items())),
            "routing_by_perturbation": dict(sorted(perturbation_counts.items())),
            "routing_by_target_pair": dict(sorted(target_counts.items())),
            "unique_tool_query_targets": len(
                {
                    canonical_sha256(row["target_output"])
                    for row in tool_query_rows
                }
            ),
        },
        "artifacts": {
            "source_tasks": {
                "path": display_path(source_tasks_path),
                "sha256": sha256_file(source_tasks_path),
            },
            "excluded_existing_manifest": {
                "path": display_path(excluded_manifest_path),
                "sha256": sha256_file(excluded_manifest_path),
            },
            "sealed_commitment": {
                "path": display_path(sealed_commitment_path),
                "sha256": sha256_file(sealed_commitment_path),
            },
            "base_manifest": {
                "path": display_path(base_out),
                "sha256": sha256_file(base_out),
                "records": len(base_rows),
            },
            "tool_query_rows": {
                "path": display_path(tool_query_out),
                "sha256": sha256_file(tool_query_out),
                "records": len(tool_query_rows),
            },
            "routing_rows": {
                "path": display_path(routing_out),
                "sha256": sha256_file(routing_out),
                "records": len(routing_rows),
            },
        },
        "sealed_separation": {
            "sealed_manifest_read": False,
            "basis": (
                "The source task file is a hash-matched public exclusion in the "
                "sealed commitment, whose aggregate source/split/claim overlap "
                "checks are all zero."
            ),
        },
        "query_contract": {
            "required_fields": list(REQUIRED_QUERY_FIELDS),
            "identifier_namespaces": {
                "drug_id": "negbiodb_ct.intervention_id",
                "condition_id": "negbiodb_ct.condition_id",
            },
            "literal_placeholders_allowed": False,
            "target_arguments_must_equal_visible_values": True,
        },
        "perturbations": list(PERTURBATIONS),
        "issues": list(issues),
        "ready_for_no_model_baselines": not issues,
        "ready_for_frozen_model_scoring": not issues,
        "training_authorized": False,
        "dpo_rlvr_authorized": False,
        "hugging_face_publication_authorized": False,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-tasks", default="negbiodb_ct/tasks_pilot.jsonl")
    parser.add_argument(
        "--exclude-manifest",
        default="negbiodb_ct/stage_a_mini_manifest.jsonl",
    )
    parser.add_argument(
        "--sealed-commitment",
        default="post_training/stage_a_sealed_extension_commitment_2026-07-10.json",
    )
    parser.add_argument(
        "--base-out",
        default="negbiodb_ct/stage_a_prospective_real_query_manifest_v1.jsonl",
    )
    parser.add_argument(
        "--tool-query-out",
        default="post_training/stage_a_prospective_real_query_tool_query_v1.jsonl",
    )
    parser.add_argument(
        "--routing-out",
        default=(
            "post_training/"
            "stage_a_prospective_real_query_routing_perturbations_v1.jsonl"
        ),
    )
    parser.add_argument(
        "--manifest-out",
        default=(
            "post_training/stage_a_prospective_real_query_experiment_manifest.json"
        ),
    )
    parser.add_argument("--seed", type=int, default=SELECTION_SEED)
    parser.add_argument("--per-action", type=int, default=5)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    source_records = load_task_records(args.source_tasks)
    excluded_rows = load_stage_a_manifest(args.exclude_manifest)
    commitment = load_json(args.sealed_commitment)
    selected = select_records(
        source_records,
        excluded_rows=excluded_rows,
        per_action=args.per_action,
        seed=args.seed,
    )
    base_rows = build_base_rows(
        selected,
        per_action=args.per_action,
        seed=args.seed,
    )
    tool_query_rows = build_tool_query_rows(base_rows)
    routing_rows = build_routing_rows(base_rows)
    issues = validate_rows(
        base_rows=base_rows,
        tool_query_rows=tool_query_rows,
        routing_rows=routing_rows,
        excluded_rows=excluded_rows,
        per_action=args.per_action,
    )
    issues.extend(
        validate_sealed_separation_commitment(
            commitment,
            source_tasks_path=args.source_tasks,
        )
    )
    issues = sorted(set(issues))

    write_stage_a_manifest(args.base_out, base_rows)
    write_jsonl(args.tool_query_out, tool_query_rows)
    write_jsonl(args.routing_out, routing_rows)
    manifest = build_manifest(
        source_tasks_path=args.source_tasks,
        excluded_manifest_path=args.exclude_manifest,
        sealed_commitment_path=args.sealed_commitment,
        base_out=args.base_out,
        tool_query_out=args.tool_query_out,
        routing_out=args.routing_out,
        base_rows=base_rows,
        tool_query_rows=tool_query_rows,
        routing_rows=routing_rows,
        seed=args.seed,
        per_action=args.per_action,
        issues=issues,
    )
    write_json(args.manifest_out, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())

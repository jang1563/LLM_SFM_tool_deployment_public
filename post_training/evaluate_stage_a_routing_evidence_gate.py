#!/usr/bin/env python3
"""Evaluate a deterministic evidence gate for all Stage A routing targets.

The gate reads only model-visible `observed_tool_loop` payloads from
evidence-conditioned routing prompts. It is a no-model runtime-enforcement
baseline, not a model-training or calibration result.
"""

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

from post_training.run_stage_a_strict_component_sft_smoke import (  # noqa: E402
    filter_component,
    load_jsonl,
    write_json,
)

DATASET = "negbiodb_ct_stage_a_routing_evidence_gate_v1"
COMPONENT = "routing_after_loop"
PROMPT_CONTRACT = "stage_a_v2_evidence_conditioned_component"
DEFAULT_OUTPUT = {"action": "defer", "evidence_status": "insufficient", "cited_source_ids": []}


def pair_label(output: Mapping[str, Any]) -> str:
    return f"{output.get('action')}/{output.get('evidence_status')}"


def prompt_payload(row: Mapping[str, Any]) -> Mapping[str, Any]:
    messages = row.get("prompt_messages")
    if not isinstance(messages, list) or len(messages) < 2:
        return {}
    user_message = messages[1]
    if not isinstance(user_message, Mapping):
        return {}
    content = user_message.get("content")
    if not isinstance(content, str):
        return {}
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, Mapping) else {}


def tool_content(payload: Mapping[str, Any], tool_name: str) -> Mapping[str, Any]:
    loop = payload.get("observed_tool_loop")
    if not isinstance(loop, list):
        return {}
    for item in loop:
        if not isinstance(item, Mapping):
            continue
        if item.get("name") != tool_name:
            continue
        content = item.get("content")
        return content if isinstance(content, Mapping) else {}
    return {}


def evidence_features(row: Mapping[str, Any]) -> dict[str, Any]:
    payload = prompt_payload(row)
    survey = tool_content(payload, "nullatlas_survey_prior_failures")
    verifier = tool_content(payload, "nullatlas_verify_trial_claims")
    validity = tool_content(payload, "nullatlas_check_value_validity")
    completeness = tool_content(payload, "nullatlas_negative_evidence_completeness")
    same_records = survey.get("same_indication_records")
    citation_candidates = completeness.get("citation_candidates")
    value_findings = validity.get("value_validity_findings")
    return {
        "observed_tool_loop_present": bool(payload.get("observed_tool_loop")),
        "related_negative_evidence_count": survey.get("related_negative_evidence_count"),
        "same_indication_record_count": len(same_records) if isinstance(same_records, list) else None,
        "records_considered": verifier.get("records_considered"),
        "citation_candidates": [str(item) for item in citation_candidates]
        if isinstance(citation_candidates, list)
        else [],
        "completeness_signal": completeness.get("completeness_signal"),
        "value_validity_findings": value_findings if isinstance(value_findings, list) else [],
    }


def first_source_id(findings: Sequence[Mapping[str, Any]]) -> str | None:
    for finding in findings:
        source_id = finding.get("source_id")
        if source_id:
            return str(source_id)
    return None


def gate_output(features: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    value_findings = features.get("value_validity_findings")
    if isinstance(value_findings, list) and value_findings:
        source_id = first_source_id([item for item in value_findings if isinstance(item, Mapping)])
        return {
            "action": "flag",
            "evidence_status": "invalid_value",
            "cited_source_ids": [source_id] if source_id else [],
        }, "invalid_numeric_value_in_same_indication_record"

    completeness = features.get("completeness_signal")
    citations = features.get("citation_candidates")
    citation_ids = [str(item) for item in citations] if isinstance(citations, list) else []
    if completeness == "mixed_endpoint_records_for_same_claim":
        return {"action": "reject", "evidence_status": "contradicted", "cited_source_ids": []}, (
            "mixed_endpoint_records_for_same_claim"
        )
    if completeness == "same_indication_failure_record_found":
        return {"action": "ground", "evidence_status": "supported", "cited_source_ids": citation_ids}, (
            "same_indication_failure_record_found"
        )
    if completeness == "related_evidence_exists_but_same_indication_record_absent":
        return {"action": "verify", "evidence_status": "insufficient", "cited_source_ids": []}, (
            "related_evidence_without_same_indication_record"
        )
    if completeness == "no_same_indication_or_related_failure_record":
        return dict(DEFAULT_OUTPUT), "no_same_indication_or_related_failure_record"

    related_count = features.get("related_negative_evidence_count")
    same_count = features.get("same_indication_record_count")
    records_considered = features.get("records_considered")
    if same_count == 0 and records_considered == 0 and related_count == 0:
        return dict(DEFAULT_OUTPUT), "count_fallback_no_evidence"
    if same_count == 0 and records_considered == 0 and isinstance(related_count, int) and related_count > 0:
        return {"action": "verify", "evidence_status": "insufficient", "cited_source_ids": []}, (
            "count_fallback_related_evidence"
        )
    return dict(DEFAULT_OUTPUT), "unknown_evidence_state_fail_closed"


def target_output(row: Mapping[str, Any]) -> dict[str, Any]:
    target = row.get("target_output")
    if not isinstance(target, Mapping):
        return {}
    citations = target.get("cited_source_ids")
    return {
        "action": target.get("action"),
        "evidence_status": target.get("evidence_status"),
        "cited_source_ids": [str(item) for item in citations] if isinstance(citations, list) else [],
    }


def compact_row(row: Mapping[str, Any], *, split_label: str) -> dict[str, Any]:
    features = evidence_features(row)
    predicted, reason = gate_output(features)
    expected = target_output(row)
    return {
        "id": row.get("id"),
        "case_id": row.get("source_manifest_case_id"),
        "split_label": split_label,
        "case_family": row.get("case_family"),
        "expected": expected,
        "expected_pair": pair_label(expected),
        "predicted": predicted,
        "predicted_pair": pair_label(predicted),
        "exact": predicted == expected,
        "action_status_exact": pair_label(predicted) == pair_label(expected),
        "reason": reason,
        "features": features,
    }


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    exact = sum(1 for row in rows if row.get("exact"))
    action_status_exact = sum(1 for row in rows if row.get("action_status_exact"))
    return {
        "rows": len(rows),
        "exact": exact,
        "accuracy": round(exact / len(rows), 6) if rows else 0.0,
        "action_status_exact": action_status_exact,
        "action_status_accuracy": round(action_status_exact / len(rows), 6) if rows else 0.0,
        "by_expected_pair": dict(sorted(Counter(str(row.get("expected_pair")) for row in rows).items())),
        "by_predicted_pair": dict(sorted(Counter(str(row.get("predicted_pair")) for row in rows).items())),
        "by_reason": dict(sorted(Counter(str(row.get("reason")) for row in rows).items())),
        "error_case_ids": [str(row.get("case_id")) for row in rows if not row.get("exact")],
    }


def build_report(
    *,
    targets: Sequence[Mapping[str, Any]],
    train_targets: Sequence[Mapping[str, Any]],
    heldout_targets: Sequence[Mapping[str, Any]],
    targets_path: Path,
) -> dict[str, Any]:
    routing_targets = filter_component(list(targets), COMPONENT)
    train_routing_targets = filter_component(list(train_targets), COMPONENT)
    heldout_routing_targets = filter_component(list(heldout_targets), COMPONENT)
    all_rows = [compact_row(row, split_label="all") for row in routing_targets]
    train_rows = [compact_row(row, split_label="train") for row in train_routing_targets]
    heldout_rows = [compact_row(row, split_label="heldout") for row in heldout_routing_targets]
    return {
        "dataset": DATASET,
        "component": COMPONENT,
        "prompt_contract": PROMPT_CONTRACT,
        "gate": "deterministic_model_visible_evidence_routing_gate",
        "input_targets": str(targets_path),
        "model_visible_fields_only": True,
        "hidden_labels_used_by_gate": False,
        "fail_closed_default": dict(DEFAULT_OUTPUT),
        "summary": {
            "all": summarize_rows(all_rows),
            "train": summarize_rows(train_rows),
            "heldout": summarize_rows(heldout_rows),
        },
        "rows": all_rows,
        "heldout_rows": heldout_rows,
        "scientific_readout": {
            "diagnostic_question": (
                "Can deterministic runtime enforcement route all Stage A "
                "evidence-conditioned routing rows from model-visible tool results?"
            ),
            "interpretation_rule": (
                "A passing no-model gate is a runtime baseline to beat. It does "
                "not prove model competence, and should not be used as a reason "
                "to start DPO/RLVR."
            ),
            "next_decision": (
                "Compare model outputs against this gate and keep tool_query, "
                "DPO/RLVR, and Hugging Face publication gated until model-heavy "
                "experiments beat the runtime baseline on broader held-out slices."
            ),
        },
    }


def write_markdown(report: Mapping[str, Any], path: Path) -> None:
    summary = report["summary"]
    lines = [
        "# Stage A Routing Evidence Gate",
        "",
        "Purpose: evaluate a no-model runtime gate for all Stage A",
        "`routing_after_loop` evidence-conditioned rows using only model-visible",
        "tool-result fields.",
        "",
        "## Summary",
        "",
        f"- Gate: `{report.get('gate')}`",
        f"- Hidden labels used by gate: `{report.get('hidden_labels_used_by_gate')}`",
        f"- All rows exact: {summary['all']['exact']}/{summary['all']['rows']}",
        f"- Train rows exact: {summary['train']['exact']}/{summary['train']['rows']}",
        f"- Held-out rows exact: {summary['heldout']['exact']}/{summary['heldout']['rows']}",
        f"- Held-out action/status exact: {summary['heldout']['action_status_exact']}/"
        f"{summary['heldout']['rows']}",
        f"- Predicted pairs: `{json.dumps(summary['all']['by_predicted_pair'], sort_keys=True)}`",
        "",
        "## Held-Out Rows",
        "",
        "| Case family | Expected | Predicted | Exact | Reason |",
        "| --- | --- | --- | ---: | --- |",
    ]
    for row in report.get("heldout_rows", []):
        lines.append(
            "| {case_family} | `{expected}` | `{predicted}` | {exact} | `{reason}` |".format(
                case_family=row.get("case_family"),
                expected=row.get("expected_pair"),
                predicted=row.get("predicted_pair"),
                exact=int(bool(row.get("exact"))),
                reason=row.get("reason"),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            str(report["scientific_readout"]["interpretation_rule"]),
            "",
            str(report["scientific_readout"]["next_decision"]),
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--targets",
        default="post_training/stage_a_evidence_conditioned_component_targets_v1.jsonl",
    )
    parser.add_argument(
        "--train-targets",
        default="post_training/stage_a_evidence_conditioned_component_targets_train_v1.jsonl",
    )
    parser.add_argument(
        "--heldout-targets",
        default="post_training/stage_a_evidence_conditioned_component_targets_heldout_v1.jsonl",
    )
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--out-md", default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    targets_path = Path(args.targets)
    report = build_report(
        targets=load_jsonl(targets_path),
        train_targets=load_jsonl(args.train_targets),
        heldout_targets=load_jsonl(args.heldout_targets),
        targets_path=targets_path,
    )
    if args.out_json:
        write_json(args.out_json, report)
    if args.out_md:
        write_markdown(report, Path(args.out_md))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

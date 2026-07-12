#!/usr/bin/env python3
"""Evaluate a deterministic evidence-boundary gate for Stage A routing.

This is a no-model runtime-enforcement baseline over the defer-vs-verify
boundary. It reads only model-visible `observed_tool_loop` content from the
component prompt and emits compact public-safe JSON/Markdown reports.
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

from post_training.run_stage_a_strict_component_sft_smoke import load_jsonl, write_json  # noqa: E402

DATASET = "negbiodb_ct_stage_a_routing_evidence_boundary_gate_v1"
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
    completeness = tool_content(payload, "nullatlas_negative_evidence_completeness")
    same_records = survey.get("same_indication_records")
    citation_candidates = completeness.get("citation_candidates")
    return {
        "observed_tool_loop_present": bool(payload.get("observed_tool_loop")),
        "related_negative_evidence_count": survey.get("related_negative_evidence_count"),
        "same_indication_record_count": len(same_records) if isinstance(same_records, list) else None,
        "records_considered": verifier.get("records_considered"),
        "citation_candidate_count": len(citation_candidates) if isinstance(citation_candidates, list) else None,
        "completeness_signal": completeness.get("completeness_signal"),
    }


def gate_output(features: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    same_records = features.get("same_indication_record_count")
    records_considered = features.get("records_considered")
    citation_candidates = features.get("citation_candidate_count")
    related_count = features.get("related_negative_evidence_count")
    completeness = features.get("completeness_signal")

    if same_records not in (0, None) or records_considered not in (0, None) or citation_candidates not in (0, None):
        return dict(DEFAULT_OUTPUT), "non_boundary_or_supported_evidence_fail_closed"
    if related_count is None:
        return dict(DEFAULT_OUTPUT), "missing_related_count_fail_closed"
    if int(related_count) > 0 or completeness == "related_evidence_exists_but_same_indication_record_absent":
        return {
            "action": "verify",
            "evidence_status": "insufficient",
            "cited_source_ids": [],
        }, "related_evidence_without_same_indication_record"
    if int(related_count) == 0 or completeness == "no_same_indication_or_related_failure_record":
        return dict(DEFAULT_OUTPUT), "no_same_indication_or_related_failure_record"
    return dict(DEFAULT_OUTPUT), "unknown_boundary_signal_fail_closed"


def compact_row(row: Mapping[str, Any], *, split_label: str) -> dict[str, Any]:
    features = evidence_features(row)
    predicted, reason = gate_output(features)
    expected = row.get("chosen_output")
    if not isinstance(expected, Mapping):
        expected = {}
    expected_compact = {
        "action": expected.get("action"),
        "evidence_status": expected.get("evidence_status"),
        "cited_source_ids": expected.get("cited_source_ids") if isinstance(expected.get("cited_source_ids"), list) else [],
    }
    exact = predicted == expected_compact
    return {
        "id": row.get("id"),
        "case_id": row.get("source_manifest_case_id"),
        "split_label": split_label,
        "case_family": row.get("case_family"),
        "expected": expected_compact,
        "expected_pair": pair_label(expected_compact),
        "predicted": predicted,
        "predicted_pair": pair_label(predicted),
        "exact": exact,
        "reason": reason,
        "features": features,
    }


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "exact": sum(1 for row in rows if row.get("exact")),
        "accuracy": round(sum(1 for row in rows if row.get("exact")) / len(rows), 6) if rows else 0.0,
        "by_expected_pair": dict(sorted(Counter(str(row.get("expected_pair")) for row in rows).items())),
        "by_predicted_pair": dict(sorted(Counter(str(row.get("predicted_pair")) for row in rows).items())),
        "by_reason": dict(sorted(Counter(str(row.get("reason")) for row in rows).items())),
    }


def build_report(
    *,
    pairs: Sequence[Mapping[str, Any]],
    train_pairs: Sequence[Mapping[str, Any]],
    heldout_pairs: Sequence[Mapping[str, Any]],
    pairs_path: Path,
) -> dict[str, Any]:
    all_rows = [compact_row(row, split_label="all") for row in pairs]
    train_rows = [compact_row(row, split_label="train") for row in train_pairs]
    heldout_rows = [compact_row(row, split_label="heldout") for row in heldout_pairs]
    return {
        "dataset": DATASET,
        "component": "routing_after_loop",
        "gate": "deterministic_model_visible_evidence_boundary",
        "input_pairs": str(pairs_path),
        "model_visible_fields_only": True,
        "hidden_labels_used_by_gate": False,
        "fail_closed_default": dict(DEFAULT_OUTPUT),
        "summary": {
            "all": summarize_rows(all_rows),
            "train": summarize_rows(train_rows),
            "heldout": summarize_rows(heldout_rows),
        },
        "rows": all_rows,
        "scientific_readout": {
            "diagnostic_question": (
                "Can a deterministic runtime gate recover the defer-vs-verify "
                "boundary from model-visible evidence state without trusting model "
                "confidence?"
            ),
            "interpretation_rule": (
                "This is a no-model baseline over a narrow boundary slice. A pass "
                "supports runtime enforcement as a useful system component, not "
                "a claim that model routing or calibration is solved."
            ),
            "next_decision": (
                "Use this as the baseline to beat before new training. Keep "
                "tool_query, DPO/RLVR, and Hugging Face publication gated."
            ),
        },
    }


def write_markdown(report: Mapping[str, Any], path: Path) -> None:
    summary = report["summary"]
    lines = [
        "# Stage A Routing Evidence Boundary Gate",
        "",
        "Purpose: evaluate a no-model runtime gate for the defer-vs-verify",
        "routing boundary using only model-visible tool-result fields.",
        "",
        "## Summary",
        "",
        f"- Gate: `{report.get('gate')}`",
        f"- Hidden labels used by gate: `{report.get('hidden_labels_used_by_gate')}`",
        f"- All rows: {summary['all']['exact']}/{summary['all']['rows']}",
        f"- Train rows: {summary['train']['exact']}/{summary['train']['rows']}",
        f"- Held-out rows: {summary['heldout']['exact']}/{summary['heldout']['rows']}",
        f"- Predicted pairs: `{json.dumps(summary['all']['by_predicted_pair'], sort_keys=True)}`",
        f"- Reasons: `{json.dumps(summary['all']['by_reason'], sort_keys=True)}`",
        "",
        "## Held-Out Rows",
        "",
        "| Case family | Expected | Predicted | Exact | Reason | Related count | Completeness |",
        "| --- | --- | --- | ---: | --- | ---: | --- |",
    ]
    for row in report["rows"]:
        if row.get("case_id") not in {"stage_a::000012", "stage_a::000019"}:
            continue
        features = row["features"]
        lines.append(
            (
                "| {case_family} | `{expected}` | `{predicted}` | {exact} | "
                "`{reason}` | {related} | `{completeness}` |"
            ).format(
                case_family=row.get("case_family"),
                expected=row.get("expected_pair"),
                predicted=row.get("predicted_pair"),
                exact=int(bool(row.get("exact"))),
                reason=row.get("reason"),
                related=features.get("related_negative_evidence_count"),
                completeness=features.get("completeness_signal"),
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
    parser.add_argument("--pairs", default="post_training/stage_a_routing_defer_verify_contrast_pairs_v1.jsonl")
    parser.add_argument(
        "--train-pairs",
        default="post_training/stage_a_routing_defer_verify_contrast_pairs_train_v1.jsonl",
    )
    parser.add_argument(
        "--heldout-pairs",
        default="post_training/stage_a_routing_defer_verify_contrast_pairs_heldout_v1.jsonl",
    )
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--out-md", default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    pairs_path = Path(args.pairs)
    report = build_report(
        pairs=load_jsonl(pairs_path),
        train_pairs=load_jsonl(args.train_pairs),
        heldout_pairs=load_jsonl(args.heldout_pairs),
        pairs_path=pairs_path,
    )
    if args.out_json:
        write_json(args.out_json, report)
    if args.out_md:
        write_markdown(report, Path(args.out_md))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Audit model-visible evidence in Stage A component targets.

This script checks whether component prompts expose the evidence needed for
their target fields. It emits compact public-safe summaries only: no prompt
text, claims, source IDs, or raw tool outputs are copied into the report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from post_training.run_stage_a_strict_component_sft_smoke import write_json  # noqa: E402


DATASET = "negbiodb_ct_stage_a_component_visibility_audit_v1"
EVIDENCE_ROUTING_COMPONENTS = {"enum_action", "routing_after_loop"}
HIDDEN_LABEL_MARKERS = (
    "hidden_eval_metadata",
    "gold_evidence_status",
    "expected_terminal_action",
    "source_task_id",
    "split_group",
)
EVIDENCE_CONTENT_MARKERS = (
    "evidence_packet",
    "tool_result",
    "endpoint_met",
    "failure_category",
    "p_value",
    "failures_for_other_indications",
    "cited_source_ids",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def parse_visible_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    messages = row.get("prompt_messages")
    if not isinstance(messages, list) or len(messages) < 2:
        return {}
    user_message = messages[1]
    if not isinstance(user_message, Mapping):
        return {}
    try:
        payload = json.loads(str(user_message.get("content", "{}")))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def prompt_text(row: Mapping[str, Any]) -> str:
    return json.dumps(row.get("prompt_messages", []), sort_keys=True)


def observed_loop_has_tool_results(observed_tool_loop: Any) -> bool:
    if not isinstance(observed_tool_loop, list):
        return False
    for item in observed_tool_loop:
        if not isinstance(item, Mapping):
            continue
        if "content" in item or "result" in item or "output" in item:
            return True
    return False


def visible_signature(row: Mapping[str, Any]) -> dict[str, Any]:
    payload = parse_visible_payload(row)
    text = prompt_text(row)
    observed_tool_loop = payload.get("observed_tool_loop")
    return {
        "has_claim": "claim" in payload,
        "has_allowed_tools": bool(payload.get("allowed_tools")),
        "has_required_query_fields": bool(payload.get("required_query_fields")),
        "has_observed_tool_loop": isinstance(observed_tool_loop, list),
        "observed_tool_loop_len": len(observed_tool_loop) if isinstance(observed_tool_loop, list) else 0,
        "observed_tool_loop_has_tool_results": observed_loop_has_tool_results(observed_tool_loop),
        "has_evidence_content_markers": any(marker in text for marker in EVIDENCE_CONTENT_MARKERS),
        "hidden_label_markers_in_prompt": [
            marker for marker in HIDDEN_LABEL_MARKERS if marker in text
        ],
        "visible_keys": sorted(payload),
    }


def target_requires_evidence(row: Mapping[str, Any]) -> bool:
    component = str(row.get("component"))
    target = row.get("target_output")
    if component not in EVIDENCE_ROUTING_COMPONENTS:
        return False
    return isinstance(target, Mapping) and {"action", "evidence_status"}.issubset(target)


def row_audit(row: Mapping[str, Any]) -> dict[str, Any]:
    signature = visible_signature(row)
    requires_evidence = target_requires_evidence(row)
    has_evidence_for_routing = bool(
        signature["has_evidence_content_markers"]
        or signature["observed_tool_loop_has_tool_results"]
    )
    underdetermined = requires_evidence and not has_evidence_for_routing
    return {
        "id": row.get("id"),
        "component": row.get("component"),
        "case_family": row.get("case_family"),
        "split": row.get("split"),
        "target_keys": list(row.get("target_keys", [])),
        "requires_evidence_for_target": requires_evidence,
        "has_evidence_for_routing": has_evidence_for_routing,
        "underdetermined_evidence_routing": underdetermined,
        "visible_signature": signature,
    }


def count_true(rows: Sequence[Mapping[str, Any]], key: str) -> int:
    return sum(1 for row in rows if row.get(key) is True)


def component_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    components = sorted({str(row.get("component")) for row in rows})
    out: dict[str, Any] = {}
    for component in components:
        component_rows = [row for row in rows if str(row.get("component")) == component]
        observed_lens = [
            int(row["visible_signature"]["observed_tool_loop_len"])
            for row in component_rows
            if row.get("visible_signature")
        ]
        out[component] = {
            "rows": len(component_rows),
            "requires_evidence_for_target": count_true(component_rows, "requires_evidence_for_target"),
            "has_evidence_for_routing": count_true(component_rows, "has_evidence_for_routing"),
            "underdetermined_evidence_routing": count_true(
                component_rows,
                "underdetermined_evidence_routing",
            ),
            "has_observed_tool_loop": sum(
                1
                for row in component_rows
                if row.get("visible_signature", {}).get("has_observed_tool_loop")
            ),
            "observed_tool_loop_has_tool_results": sum(
                1
                for row in component_rows
                if row.get("visible_signature", {}).get("observed_tool_loop_has_tool_results")
            ),
            "has_evidence_content_markers": sum(
                1
                for row in component_rows
                if row.get("visible_signature", {}).get("has_evidence_content_markers")
            ),
            "hidden_label_leak_rows": sum(
                1
                for row in component_rows
                if row.get("visible_signature", {}).get("hidden_label_markers_in_prompt")
            ),
            "mean_observed_tool_loop_len": round(mean(observed_lens), 3) if observed_lens else 0.0,
            "visible_key_sets": dict(
                sorted(
                    Counter(
                        ",".join(row.get("visible_signature", {}).get("visible_keys", ()))
                        for row in component_rows
                    ).items()
                )
            ),
        }
    return out


def build_visibility_report(
    rows: Sequence[Mapping[str, Any]],
    *,
    targets_path: Path,
    run_id: str,
) -> dict[str, Any]:
    audits = [row_audit(row) for row in rows]
    by_component = component_summary(audits)
    underdetermined = [
        row for row in audits if row.get("underdetermined_evidence_routing")
    ]
    leak_rows = [
        row for row in audits if row.get("visible_signature", {}).get("hidden_label_markers_in_prompt")
    ]
    return {
        "dataset": DATASET,
        "run_id": run_id,
        "component": "stage_a_components",
        "input_targets_sha256": sha256_file(targets_path),
        "rows": len(audits),
        "by_component": by_component,
        "summary": {
            "evidence_routing_components": sorted(EVIDENCE_ROUTING_COMPONENTS),
            "underdetermined_evidence_routing_rows": len(underdetermined),
            "hidden_label_leak_rows": len(leak_rows),
            "components_with_underdetermined_routing": sorted(
                {str(row.get("component")) for row in underdetermined}
            ),
        },
        "row_examples": [
            {
                "component": row.get("component"),
                "target_keys": row.get("target_keys"),
                "visible_keys": row.get("visible_signature", {}).get("visible_keys"),
                "has_observed_tool_loop": row.get("visible_signature", {}).get("has_observed_tool_loop"),
                "observed_tool_loop_has_tool_results": row.get("visible_signature", {}).get(
                    "observed_tool_loop_has_tool_results"
                ),
                "has_evidence_content_markers": row.get("visible_signature", {}).get(
                    "has_evidence_content_markers"
                ),
                "underdetermined_evidence_routing": row.get("underdetermined_evidence_routing"),
            }
            for row in audits[:6]
        ],
        "scientific_readout": {
            "diagnostic_question": (
                "Do Stage A component prompts expose enough model-visible evidence "
                "to train action/evidence-status routing, or are some targets "
                "hidden-label projections without evidence packets?"
            ),
            "interpretation_rule": (
                "If enum/routing targets require action plus evidence_status but "
                "the prompt lacks tool results or evidence content, further enum "
                "loss shaping is not a clean repair; the substrate should expose "
                "evidence-conditioned state before moving to tool_query, DPO, or RLVR."
            ),
            "boundary": (
                "This is a data-interface audit, not a model score, DPO/RLVR "
                "reward, or full trajectory result."
            ),
        },
    }


def write_markdown(report: Mapping[str, Any], path: Path) -> None:
    lines = [
        "# Stage A Component Visibility Audit",
        "",
        "Purpose: check whether component prompts expose the evidence",
        "needed for action/evidence-status routing, without publishing claims,",
        "source IDs, raw tool outputs, or hidden labels.",
        "",
        "## Summary",
        "",
        f"- Rows audited: {report['rows']}",
        f"- Underdetermined evidence-routing rows: {report['summary']['underdetermined_evidence_routing_rows']}",
        f"- Hidden-label leak rows: {report['summary']['hidden_label_leak_rows']}",
        (
            "- Components with underdetermined routing: "
            f"`{json.dumps(report['summary']['components_with_underdetermined_routing'])}`"
        ),
        "",
        "## By Component",
        "",
        (
            "| Component | Rows | Requires evidence target | Has evidence content | "
            "Observed loop | Loop has results | Underdetermined | Hidden-label leaks |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for component, summary in sorted(report["by_component"].items()):
        lines.append(
            (
                "| {component} | {rows} | {requires} | {evidence} | {loop} | "
                "{loop_results} | {underdetermined} | {leaks} |"
            ).format(
                component=component,
                rows=summary["rows"],
                requires=summary["requires_evidence_for_target"],
                evidence=summary["has_evidence_for_routing"],
                loop=summary["has_observed_tool_loop"],
                loop_results=summary["observed_tool_loop_has_tool_results"],
                underdetermined=summary["underdetermined_evidence_routing"],
                leaks=summary["hidden_label_leak_rows"],
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            str(report["scientific_readout"]["interpretation_rule"]),
            "",
            str(report["scientific_readout"]["boundary"]),
            "",
            "## Trace",
            "",
            f"- Input targets SHA-256: `{report['input_targets_sha256']}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--targets",
        default="post_training/stage_a_strict_component_targets_v1.jsonl",
    )
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--out-md", default=None)
    parser.add_argument("--run-id", default="stage_a_component_visibility_audit_2026_07_05")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    targets_path = Path(args.targets)
    report = build_visibility_report(load_jsonl(targets_path), targets_path=targets_path, run_id=args.run_id)
    if args.out_json:
        write_json(args.out_json, report)
    if args.out_md:
        write_markdown(report, Path(args.out_md))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

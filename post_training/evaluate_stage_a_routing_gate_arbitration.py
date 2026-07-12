#!/usr/bin/env python3
"""Compare routing arbitration policies on public-safe Stage A gate reports."""

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

from post_training.run_stage_a_strict_component_sft_smoke import write_json  # noqa: E402

DATASET = "negbiodb_ct_stage_a_routing_gate_arbitration_v1"
DEFAULT_CANDIDATE_GATE_REPORT = "post_training/stage_a_routing_defer_verify_gate_trained_2026-07-08.json"
DEFAULT_EVIDENCE_GATE_REPORT = "post_training/stage_a_routing_evidence_boundary_gate_2026-07-08.json"


def pair_label(output: Mapping[str, Any] | None) -> str:
    if not isinstance(output, Mapping):
        return "missing/missing"
    return f"{output.get('action')}/{output.get('evidence_status')}"


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def infer_threshold(candidate_report: Mapping[str, Any], fallback: float) -> float:
    best = candidate_report.get("best_default_zero_unsafe_report")
    if isinstance(best, Mapping) and best.get("threshold") is not None:
        return float(best["threshold"])
    return fallback


def policy_prediction(
    *,
    policy: str,
    candidate_row: Mapping[str, Any],
    evidence_row: Mapping[str, Any] | None,
    threshold: float,
) -> tuple[str, str]:
    if policy == "raw_candidate_top1":
        return str(candidate_row.get("top_pair")), "trust_candidate_top1"
    if policy == "score_gap_fail_closed":
        gap = candidate_row.get("top_second_gap")
        if gap is not None and float(gap) >= threshold:
            return str(candidate_row.get("top_pair")), "trust_high_gap_candidate"
        return str(candidate_row.get("fail_closed_pair")), "fail_closed_low_gap"
    if policy == "evidence_boundary_override":
        if evidence_row is not None:
            return str(evidence_row.get("predicted_pair")), "use_evidence_boundary_gate"
        return str(candidate_row.get("fail_closed_pair")), "missing_evidence_gate_fail_closed"
    if policy == "hybrid_evidence_then_score_gap":
        if evidence_row is not None:
            return str(evidence_row.get("predicted_pair")), "use_evidence_boundary_gate"
        gap = candidate_row.get("top_second_gap")
        if gap is not None and float(gap) >= threshold:
            return str(candidate_row.get("top_pair")), "trust_high_gap_candidate"
        return str(candidate_row.get("fail_closed_pair")), "fail_closed_low_gap"
    raise ValueError(f"unknown policy: {policy}")


def summarize_policy(rows: Sequence[Mapping[str, Any]], policy: str) -> dict[str, Any]:
    policy_rows = [row for row in rows if row.get("policy") == policy]
    exact = sum(1 for row in policy_rows if row.get("exact"))
    return {
        "policy": policy,
        "rows": len(policy_rows),
        "exact": exact,
        "accuracy": round(exact / len(policy_rows), 6) if policy_rows else 0.0,
        "by_predicted_pair": dict(sorted(Counter(str(row.get("predicted_pair")) for row in policy_rows).items())),
        "by_reason": dict(sorted(Counter(str(row.get("reason")) for row in policy_rows).items())),
        "error_case_ids": [str(row.get("case_id")) for row in policy_rows if not row.get("exact")],
    }


def build_arbitration_report(
    *,
    candidate_report: Mapping[str, Any],
    evidence_report: Mapping[str, Any],
    candidate_report_path: str | Path,
    evidence_report_path: str | Path,
    score_gap_threshold: float | None = None,
) -> dict[str, Any]:
    threshold = infer_threshold(candidate_report, 0.025) if score_gap_threshold is None else score_gap_threshold
    evidence_by_case = {
        str(row.get("case_id")): row
        for row in evidence_report.get("rows", [])
        if isinstance(row, Mapping)
    }
    policies = (
        "raw_candidate_top1",
        "score_gap_fail_closed",
        "evidence_boundary_override",
        "hybrid_evidence_then_score_gap",
    )
    rows: list[dict[str, Any]] = []
    for candidate_row in candidate_report.get("rows", []):
        if not isinstance(candidate_row, Mapping):
            continue
        case_id = str(candidate_row.get("case_id"))
        evidence_row = evidence_by_case.get(case_id)
        expected_pair = str(candidate_row.get("target_pair"))
        for policy in policies:
            predicted_pair, reason = policy_prediction(
                policy=policy,
                candidate_row=candidate_row,
                evidence_row=evidence_row,
                threshold=threshold,
            )
            rows.append(
                {
                    "case_id": case_id,
                    "case_family": candidate_row.get("case_family"),
                    "expected_pair": expected_pair,
                    "candidate_top_pair": candidate_row.get("top_pair"),
                    "candidate_top_second_gap": candidate_row.get("top_second_gap"),
                    "evidence_gate_pair": evidence_row.get("predicted_pair") if evidence_row else None,
                    "policy": policy,
                    "predicted_pair": predicted_pair,
                    "exact": predicted_pair == expected_pair,
                    "reason": reason,
                }
            )
    by_policy = {policy: summarize_policy(rows, policy) for policy in policies}
    return {
        "dataset": DATASET,
        "component": "routing_after_loop",
        "boundary": "defer_vs_verify_insufficient",
        "candidate_gate_report": str(candidate_report_path),
        "evidence_gate_report": str(evidence_report_path),
        "score_gap_threshold": threshold,
        "cases": len(candidate_report.get("rows", [])),
        "summary": {
            "by_policy": by_policy,
            "best_policy_names": [
                name for name, summary in by_policy.items()
                if summary["exact"] == max(item["exact"] for item in by_policy.values())
            ],
        },
        "rows": rows,
        "scientific_readout": {
            "diagnostic_question": (
                "Which public-safe runtime policy should arbitrate the weak "
                "defer-vs-verify routing model: raw top-1, score-gap fail-closed, "
                "or deterministic evidence-boundary override?"
            ),
            "interpretation_rule": (
                "If evidence-boundary or hybrid policies dominate raw top-1 on the "
                "held-out boundary slice, the next system design should route "
                "through runtime enforcement before adding new training objectives."
            ),
            "next_decision": (
                "Keep tool_query, DPO/RLVR, and Hugging Face publication gated. "
                "Use arbitration as a public-safe system baseline before more "
                "model-heavy experiments."
            ),
        },
    }


def write_markdown(report: Mapping[str, Any], path: Path) -> None:
    by_policy = report["summary"]["by_policy"]
    lines = [
        "# Stage A Routing Gate Arbitration",
        "",
        "Purpose: compare raw candidate top-1, score-gap fail-closed routing,",
        "and deterministic evidence-boundary override on the defer-vs-verify",
        "held-out boundary slice.",
        "",
        "## Summary",
        "",
        f"- Cases: {report['cases']}",
        f"- Score-gap threshold: {report['score_gap_threshold']}",
        f"- Best policy names: `{json.dumps(report['summary']['best_policy_names'])}`",
        "",
        "| Policy | Exact | Rows | Accuracy | Error case IDs |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for policy, summary in by_policy.items():
        lines.append(
            "| {policy} | {exact} | {rows} | {accuracy} | `{errors}` |".format(
                policy=policy,
                exact=summary["exact"],
                rows=summary["rows"],
                accuracy=summary["accuracy"],
                errors=json.dumps(summary["error_case_ids"]),
            )
        )
    lines.extend(
        [
            "",
            "## Held-Out Rows",
            "",
            "| Case ID | Expected | Candidate top | Evidence gate | Policy | Predicted | Exact | Reason |",
            "| --- | --- | --- | --- | --- | --- | ---: | --- |",
        ]
    )
    for row in report["rows"]:
        lines.append(
            (
                "| {case_id} | `{expected}` | `{candidate}` | `{evidence}` | "
                "`{policy}` | `{predicted}` | {exact} | `{reason}` |"
            ).format(
                case_id=row.get("case_id"),
                expected=row.get("expected_pair"),
                candidate=row.get("candidate_top_pair"),
                evidence=row.get("evidence_gate_pair"),
                policy=row.get("policy"),
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
    parser.add_argument("--candidate-gate-report", default=DEFAULT_CANDIDATE_GATE_REPORT)
    parser.add_argument("--evidence-gate-report", default=DEFAULT_EVIDENCE_GATE_REPORT)
    parser.add_argument("--score-gap-threshold", type=float, default=None)
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--out-md", default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    report = build_arbitration_report(
        candidate_report=load_json(args.candidate_gate_report),
        evidence_report=load_json(args.evidence_gate_report),
        candidate_report_path=args.candidate_gate_report,
        evidence_report_path=args.evidence_gate_report,
        score_gap_threshold=args.score_gap_threshold,
    )
    if args.out_json:
        write_json(args.out_json, report)
    if args.out_md:
        write_markdown(report, Path(args.out_md))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Analyze fail-closed gates for Stage A routing candidate-score rows.

The input is an ignored raw routing candidate JSONL artifact. The output is a
compact public-safe report: no prompts, scheduler logs, model state, or full
hidden metadata are copied.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from post_training.analyze_stage_a_routing_candidate_scores import (  # noqa: E402
    action_status_key,
    candidate_key,
    compact_candidate,
    find_candidate_rank,
    pair_label,
    sha256_file,
    sorted_candidate_scores,
)
from post_training.run_stage_a_strict_component_sft_smoke import load_jsonl, write_json  # noqa: E402

DATASET = "negbiodb_ct_stage_a_routing_candidate_gate_diagnostic_v1"
DEFAULT_THRESHOLDS = (0.0, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2)
DEFAULT_FAIL_CLOSED_OUTPUT = {
    "action": "defer",
    "evidence_status": "insufficient",
    "cited_source_ids": [],
}


def top_second_gap(scores: Sequence[Mapping[str, Any]]) -> float | None:
    if len(scores) < 2:
        return None
    return round(float(scores[0]["score"]) - float(scores[1]["score"]), 6)


def compact_gate_row(
    row: Mapping[str, Any],
    *,
    fail_closed_output: Mapping[str, Any],
    max_top_candidates: int,
) -> dict[str, Any]:
    scores = sorted_candidate_scores(row)
    target = compact_candidate(row.get("target_output"))
    top = scores[0] if scores else None
    top_candidate = top["candidate"] if top else None
    top_score = float(top["score"]) if top else None
    gold_rank, gold_score = find_candidate_rank(scores, target)
    gap = top_second_gap(scores)
    top_gold_margin = None
    if top_score is not None and gold_score is not None:
        top_gold_margin = round(top_score - gold_score, 6)
    fail_closed = compact_candidate(fail_closed_output)
    return {
        "id": row.get("id"),
        "case_id": row.get("case_id"),
        "case_family": row.get("case_family"),
        "chosen_pair": row.get("chosen_pair"),
        "target_output": target,
        "target_pair": pair_label(target),
        "top_candidate": top_candidate,
        "top_pair": pair_label(top_candidate),
        "fail_closed_output": fail_closed,
        "fail_closed_pair": pair_label(fail_closed),
        "exact_top1": top_candidate is not None and candidate_key(top_candidate) == candidate_key(target),
        "action_status_top1": (
            top_candidate is not None
            and action_status_key(top_candidate) == action_status_key(target)
        ),
        "fail_closed_exact": candidate_key(fail_closed) == candidate_key(target),
        "fail_closed_action_status": action_status_key(fail_closed) == action_status_key(target),
        "gold_rank": gold_rank,
        "gold_score": gold_score,
        "top_score": top_score,
        "top_second_gap": gap,
        "top_gold_margin": top_gold_margin,
        "candidate_count": len(scores),
        "top_candidates": scores[:max_top_candidates],
    }


def evaluate_threshold(rows: Sequence[Mapping[str, Any]], threshold: float) -> dict[str, Any]:
    trusted = [
        row
        for row in rows
        if row.get("top_second_gap") is not None and float(row["top_second_gap"]) >= threshold
    ]
    fail_closed = [row for row in rows if row not in trusted]
    trusted_correct = [row for row in trusted if row.get("exact_top1")]
    trusted_incorrect = [row for row in trusted if not row.get("exact_top1")]
    fail_closed_exact = [row for row in fail_closed if row.get("fail_closed_exact")]
    fail_closed_action_status = [
        row for row in fail_closed if row.get("fail_closed_action_status")
    ]
    strict_final_correct = len(trusted_correct) + len(fail_closed_exact)
    action_status_final_correct = len(trusted_correct) + len(fail_closed_action_status)
    return {
        "threshold": threshold,
        "trusted": len(trusted),
        "trusted_correct": len(trusted_correct),
        "trusted_incorrect": len(trusted_incorrect),
        "fail_closed": len(fail_closed),
        "fail_closed_exact_correct": len(fail_closed_exact),
        "fail_closed_action_status_correct": len(fail_closed_action_status),
        "strict_final_correct": strict_final_correct,
        "action_status_final_correct": action_status_final_correct,
        "coverage": round(len(trusted) / len(rows), 6) if rows else 0.0,
        "trusted_precision": round(len(trusted_correct) / len(trusted), 6) if trusted else None,
        "strict_final_accuracy": round(strict_final_correct / len(rows), 6) if rows else 0.0,
        "action_status_final_accuracy": (
            round(action_status_final_correct / len(rows), 6) if rows else 0.0
        ),
        "unsafe_trust_case_ids": [str(row.get("case_id")) for row in trusted_incorrect],
        "fail_closed_case_ids": [str(row.get("case_id")) for row in fail_closed],
    }


def adaptive_zero_unsafe_threshold(rows: Sequence[Mapping[str, Any]]) -> float | None:
    incorrect_gaps = [
        float(row["top_second_gap"])
        for row in rows
        if not row.get("exact_top1") and row.get("top_second_gap") is not None
    ]
    if not incorrect_gaps:
        return 0.0
    return round(max(incorrect_gaps) + 0.000001, 6)


def best_zero_unsafe_report(reports: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    zero_unsafe = [report for report in reports if report["trusted_incorrect"] == 0]
    if not zero_unsafe:
        return None
    return sorted(
        zero_unsafe,
        key=lambda report: (
            -int(report["strict_final_correct"]),
            -int(report["trusted_correct"]),
            float(report["threshold"]),
        ),
    )[0]


def build_routing_candidate_gate_report(
    candidate_rows: Sequence[Mapping[str, Any]],
    *,
    candidates_path: Path,
    run_id: str | None = None,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    fail_closed_output: Mapping[str, Any] = DEFAULT_FAIL_CLOSED_OUTPUT,
    max_top_candidates: int = 3,
) -> dict[str, Any]:
    rows = [
        compact_gate_row(
            row,
            fail_closed_output=fail_closed_output,
            max_top_candidates=max_top_candidates,
        )
        for row in candidate_rows
    ]
    first_run_id = run_id
    if first_run_id is None:
        for row in candidate_rows:
            if isinstance(row.get("run_id"), str):
                first_run_id = str(row["run_id"])
                break
    thresholds = sorted(set(float(threshold) for threshold in thresholds))
    threshold_reports = [evaluate_threshold(rows, threshold) for threshold in thresholds]
    adaptive_threshold = adaptive_zero_unsafe_threshold(rows)
    adaptive_report = (
        evaluate_threshold(rows, adaptive_threshold) if adaptive_threshold is not None else None
    )
    gaps = [float(row["top_second_gap"]) for row in rows if row.get("top_second_gap") is not None]
    gold_ranks = [int(row["gold_rank"]) for row in rows if isinstance(row.get("gold_rank"), int)]
    top_gold_margins = [
        float(row["top_gold_margin"]) for row in rows if row.get("top_gold_margin") is not None
    ]
    top_pair_counts = Counter(str(row.get("top_pair")) for row in rows)
    target_pair_counts = Counter(str(row.get("target_pair")) for row in rows)
    candidate_counts = Counter(int(row["candidate_count"]) for row in rows)
    return {
        "dataset": DATASET,
        "component": "routing_after_loop",
        "run_id": first_run_id,
        "input_candidates_sha256": sha256_file(candidates_path),
        "candidate_policy": candidate_rows[0].get("candidate_policy") if candidate_rows else None,
        "score_label": candidate_rows[0].get("score_label") if candidate_rows else None,
        "failure_mode": candidate_rows[0].get("failure_mode") if candidate_rows else None,
        "cases": len(rows),
        "fail_closed_output": compact_candidate(fail_closed_output),
        "summary": {
            "exact_top1": sum(1 for row in rows if row.get("exact_top1")),
            "action_status_top1": sum(1 for row in rows if row.get("action_status_top1")),
            "fail_closed_exact_correct": sum(1 for row in rows if row.get("fail_closed_exact")),
            "candidate_count_histogram": dict(sorted(candidate_counts.items())),
            "mean_gold_rank": round(mean(gold_ranks), 3) if gold_ranks else None,
            "mean_top_gold_margin": round(mean(top_gold_margins), 6) if top_gold_margins else None,
            "mean_top_second_gap": round(mean(gaps), 6) if gaps else None,
            "min_top_second_gap": round(min(gaps), 6) if gaps else None,
            "max_top_second_gap": round(max(gaps), 6) if gaps else None,
            "top_pair_counts": dict(sorted(top_pair_counts.items())),
            "target_pair_counts": dict(sorted(target_pair_counts.items())),
        },
        "threshold_reports": threshold_reports,
        "best_default_zero_unsafe_report": best_zero_unsafe_report(threshold_reports),
        "adaptive_zero_unsafe_threshold": adaptive_threshold,
        "adaptive_zero_unsafe_report": adaptive_report,
        "rows": rows,
        "scientific_readout": {
            "diagnostic_question": (
                "Can a routing top-vs-second score gap identify candidate top-1 "
                "decisions that are safe to trust, while failing closed to "
                "defer/insufficient otherwise?"
            ),
            "interpretation_rule": (
                "A useful boundary gate should avoid unsafe trust and improve or "
                "preserve strict final correctness after fail-closed routing. On "
                "tiny held-out slices this is only a diagnostic, not deployment "
                "calibration or a reason to start DPO/RLVR."
            ),
            "decision_boundary": (
                "This post-hoc diagnostic uses saved candidate scores only. It is "
                "not new training, not explanation-quality scoring, and not a "
                "full trajectory result."
            ),
        },
    }


def write_markdown(report: Mapping[str, Any], path: Path) -> None:
    summary = report["summary"]
    best = report.get("best_default_zero_unsafe_report") or {}
    adaptive = report.get("adaptive_zero_unsafe_report") or {}
    fail_closed = report.get("fail_closed_output") or {}
    lines = [
        "# Stage A Routing Candidate Gate Diagnostic",
        "",
        "Purpose: test whether routing candidate score gaps can support a",
        "fail-closed boundary gate without publishing prompts, raw score JSONL,",
        "model state, or scheduler logs.",
        "",
        "## Summary",
        "",
        f"- Run ID: `{report.get('run_id')}`",
        f"- Candidate policy: `{report.get('candidate_policy')}`",
        f"- Score label: `{report.get('score_label')}`",
        f"- Cases: {report['cases']}",
        f"- Exact top-1: {summary['exact_top1']}/{report['cases']}",
        f"- Mean gold rank: {summary['mean_gold_rank']}",
        f"- Mean top-second gap: {summary['mean_top_second_gap']}",
        f"- Fail-closed output: `{fail_closed.get('action')}` / `{fail_closed.get('evidence_status')}`",
        f"- Top pair counts: `{json.dumps(summary['top_pair_counts'], sort_keys=True)}`",
        f"- Target pair counts: `{json.dumps(summary['target_pair_counts'], sort_keys=True)}`",
        "",
        "## Gate Thresholds",
        "",
        "| Threshold | Trusted | Correct trusted | Unsafe trusted | Fail closed | Strict final correct | Strict accuracy |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["threshold_reports"]:
        lines.append(
            (
                "| {threshold} | {trusted} | {trusted_correct} | {trusted_incorrect} | "
                "{fail_closed} | {strict_final_correct} | {strict_final_accuracy} |"
            ).format(**row)
        )
    lines.extend(
        [
            "",
            "## Fail-Closed Readout",
            "",
            (
                f"- Best default zero-unsafe threshold: `{best.get('threshold')}` "
                f"with {best.get('strict_final_correct')} strict final correct rows "
                f"and {best.get('trusted_incorrect')} unsafe trusted rows."
            ),
            (
                f"- Adaptive zero-unsafe threshold: "
                f"`{report.get('adaptive_zero_unsafe_threshold')}` with "
                f"{adaptive.get('strict_final_correct')} strict final correct rows "
                f"and {adaptive.get('trusted_incorrect')} unsafe trusted rows."
            ),
            "",
            "## Held-Out Rows",
            "",
            "| Case family | Target | Top candidate | Top-1 | Gold rank | Gap | Fail-closed exact |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in report["rows"]:
        target = row["target_output"]
        top = row.get("top_candidate") or {}
        lines.append(
            (
                "| {case_family} | `{target_action}` / `{target_status}` | "
                "`{top_action}` / `{top_status}` | {exact} | {rank} | {gap} | {fail_closed_exact} |"
            ).format(
                case_family=row.get("case_family"),
                target_action=target.get("action"),
                target_status=target.get("evidence_status"),
                top_action=top.get("action"),
                top_status=top.get("evidence_status"),
                exact=int(bool(row.get("exact_top1"))),
                rank=row.get("gold_rank"),
                gap=row.get("top_second_gap"),
                fail_closed_exact=int(bool(row.get("fail_closed_exact"))),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            str(report["scientific_readout"]["interpretation_rule"]),
            "",
            str(report["scientific_readout"]["decision_boundary"]),
            "",
            "## Trace",
            "",
            f"- Input candidate JSONL SHA-256: `{report['input_candidates_sha256']}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def parse_thresholds(value: str) -> list[float]:
    thresholds = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        threshold = float(part)
        if threshold < 0:
            raise ValueError("thresholds must be non-negative")
        thresholds.append(threshold)
    if not thresholds:
        raise ValueError("at least one threshold is required")
    return sorted(set(thresholds))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--out-md", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--thresholds", default=",".join(str(value) for value in DEFAULT_THRESHOLDS))
    parser.add_argument("--fail-closed-action", default=DEFAULT_FAIL_CLOSED_OUTPUT["action"])
    parser.add_argument("--fail-closed-status", default=DEFAULT_FAIL_CLOSED_OUTPUT["evidence_status"])
    parser.add_argument("--max-top-candidates", type=int, default=3)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    candidates_path = Path(args.candidates)
    fail_closed_output = {
        "action": args.fail_closed_action,
        "evidence_status": args.fail_closed_status,
        "cited_source_ids": [],
    }
    report = build_routing_candidate_gate_report(
        load_jsonl(candidates_path),
        candidates_path=candidates_path,
        run_id=args.run_id,
        thresholds=parse_thresholds(args.thresholds),
        fail_closed_output=fail_closed_output,
        max_top_candidates=args.max_top_candidates,
    )
    if args.out_json:
        write_json(args.out_json, report)
    if args.out_md:
        write_markdown(report, Path(args.out_md))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

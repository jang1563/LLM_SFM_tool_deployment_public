#!/usr/bin/env python3
"""Analyze fail-closed gates for Stage A saved-prediction candidate rows.

The input is an ignored raw candidate-readout prediction JSONL artifact. The
output is a compact public-safe report: no prompts, raw model text, scheduler
logs, model state, or full candidate-score tables are copied.
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

from post_training.run_stage_a_sft_smoke_eval import load_jsonl, write_json  # noqa: E402


DATASET = "negbiodb_ct_stage_a_saved_candidate_gate_diagnostic_v1"
DEFAULT_THRESHOLDS = (0.0, 0.005, 0.01, 0.011, 0.025, 0.03, 0.035, 0.04, 0.05)
DEFAULT_FAIL_CLOSED_PAIR = {
    "action": "defer",
    "evidence_status": "insufficient",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compact_pair(obj: Mapping[str, Any] | None) -> dict[str, str | None]:
    obj = obj or {}
    return {
        "action": str(obj.get("action")) if obj.get("action") is not None else None,
        "evidence_status": (
            str(obj.get("evidence_status")) if obj.get("evidence_status") is not None else None
        ),
    }


def pair_label(obj: Mapping[str, Any] | None) -> str:
    pair = compact_pair(obj)
    return f"{pair['action']}/{pair['evidence_status']}"


def pair_key(obj: Mapping[str, Any] | None) -> tuple[str | None, str | None]:
    pair = compact_pair(obj)
    return pair["action"], pair["evidence_status"]


def sorted_candidate_scores(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_scores = row.get("candidate_scores")
    if not isinstance(raw_scores, list):
        return []
    scores: list[dict[str, Any]] = []
    for item in raw_scores:
        if not isinstance(item, Mapping) or not isinstance(item.get("candidate"), Mapping):
            continue
        try:
            score = float(item["score"])
        except (KeyError, TypeError, ValueError):
            continue
        scores.append({"score": score, "candidate": compact_pair(item["candidate"])})
    return sorted(scores, key=lambda item: (-item["score"], json.dumps(item["candidate"], sort_keys=True)))


def validate_candidate_rows(candidate_rows: Sequence[Mapping[str, Any]]) -> None:
    if not candidate_rows:
        raise ValueError("candidate-score JSONL is empty")
    issues: list[str] = []
    for index, row in enumerate(candidate_rows, start=1):
        case_id = row.get("case_id", f"row[{index}]")
        target = row.get("target_pair")
        if not isinstance(target, Mapping) or None in pair_key(target):
            issues.append(f"{case_id}: missing target_pair action/evidence_status")
        if len(sorted_candidate_scores(row)) < 2:
            issues.append(f"{case_id}: candidate_scores must contain at least two scored candidates")
    if issues:
        preview = "; ".join(issues[:5])
        more = f"; ... {len(issues) - 5} more" if len(issues) > 5 else ""
        raise ValueError(f"invalid saved candidate-score JSONL: {preview}{more}")


def top_second_gap(scores: Sequence[Mapping[str, Any]]) -> float | None:
    if len(scores) < 2:
        return None
    return round(float(scores[0]["score"]) - float(scores[1]["score"]), 6)


def find_pair_rank(
    scores: Sequence[Mapping[str, Any]],
    target_pair: Mapping[str, Any],
) -> tuple[int | None, float | None]:
    target_key = pair_key(target_pair)
    for index, item in enumerate(scores, start=1):
        candidate = item.get("candidate")
        if isinstance(candidate, Mapping) and pair_key(candidate) == target_key:
            return index, float(item["score"])
    return None, None


def compact_gate_row(
    row: Mapping[str, Any],
    *,
    fail_closed_pair: Mapping[str, Any],
    max_top_candidates: int,
) -> dict[str, Any]:
    scores = sorted_candidate_scores(row)
    target_pair = compact_pair(row.get("target_pair") if isinstance(row.get("target_pair"), Mapping) else {})
    top = scores[0] if scores else None
    top_pair = compact_pair(top.get("candidate") if isinstance(top, Mapping) else {})
    top_score = float(top["score"]) if top else None
    target_rank, target_score = find_pair_rank(scores, target_pair)
    gap = top_second_gap(scores)
    top_target_margin = None
    if top_score is not None and target_score is not None:
        top_target_margin = round(top_score - target_score, 6)
    fail_closed = compact_pair(fail_closed_pair)
    return {
        "case_id": row.get("case_id"),
        "case_family": row.get("case_family"),
        "target_pair": target_pair,
        "target_pair_label": pair_label(target_pair),
        "top_pair": top_pair,
        "top_pair_label": pair_label(top_pair),
        "fail_closed_pair": fail_closed,
        "fail_closed_pair_label": pair_label(fail_closed),
        "exact_top1": pair_key(top_pair) == pair_key(target_pair),
        "fail_closed_exact": pair_key(fail_closed) == pair_key(target_pair),
        "target_rank": target_rank,
        "target_score": target_score,
        "top_score": top_score,
        "top_second_gap": gap,
        "top_target_margin": top_target_margin,
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
    strict_final_correct = len(trusted_correct) + len(fail_closed_exact)
    return {
        "threshold": threshold,
        "trusted": len(trusted),
        "trusted_correct": len(trusted_correct),
        "trusted_incorrect": len(trusted_incorrect),
        "fail_closed": len(fail_closed),
        "fail_closed_exact_correct": len(fail_closed_exact),
        "strict_final_correct": strict_final_correct,
        "coverage": round(len(trusted) / len(rows), 6) if rows else 0.0,
        "trusted_precision": round(len(trusted_correct) / len(trusted), 6) if trusted else None,
        "strict_final_accuracy": round(strict_final_correct / len(rows), 6) if rows else 0.0,
        "unsafe_trust_case_ids": [str(row.get("case_id")) for row in trusted_incorrect],
        "trusted_case_ids": [str(row.get("case_id")) for row in trusted],
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


def build_saved_candidate_gate_report(
    candidate_rows: Sequence[Mapping[str, Any]],
    *,
    candidates_path: Path,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    fail_closed_pair: Mapping[str, Any] = DEFAULT_FAIL_CLOSED_PAIR,
    max_top_candidates: int = 3,
) -> dict[str, Any]:
    validate_candidate_rows(candidate_rows)
    rows = [
        compact_gate_row(
            row,
            fail_closed_pair=fail_closed_pair,
            max_top_candidates=max_top_candidates,
        )
        for row in candidate_rows
    ]
    thresholds = sorted(set(float(threshold) for threshold in thresholds))
    threshold_reports = [evaluate_threshold(rows, threshold) for threshold in thresholds]
    adaptive_threshold = adaptive_zero_unsafe_threshold(rows)
    adaptive_report = (
        evaluate_threshold(rows, adaptive_threshold) if adaptive_threshold is not None else None
    )
    gaps = [float(row["top_second_gap"]) for row in rows if row.get("top_second_gap") is not None]
    target_ranks = [int(row["target_rank"]) for row in rows if isinstance(row.get("target_rank"), int)]
    top_target_margins = [
        float(row["top_target_margin"]) for row in rows if row.get("top_target_margin") is not None
    ]
    top_pair_counts = Counter(str(row.get("top_pair_label")) for row in rows)
    target_pair_counts = Counter(str(row.get("target_pair_label")) for row in rows)
    candidate_counts = Counter(int(row["candidate_count"]) for row in rows)
    run_id = None
    for row in candidate_rows:
        if isinstance(row.get("run_id"), str):
            run_id = str(row["run_id"])
            break
    return {
        "dataset": DATASET,
        "run_id": run_id,
        "input_candidates_sha256": sha256_file(candidates_path),
        "candidate_policy": candidate_rows[0].get("candidate_policy") if candidate_rows else None,
        "prompt_contract": candidate_rows[0].get("prompt_contract") if candidate_rows else None,
        "model": candidate_rows[0].get("model") if candidate_rows else None,
        "cases": len(rows),
        "fail_closed_pair": compact_pair(fail_closed_pair),
        "summary": {
            "exact_top1": sum(1 for row in rows if row.get("exact_top1")),
            "fail_closed_exact_correct": sum(1 for row in rows if row.get("fail_closed_exact")),
            "candidate_count_histogram": dict(sorted(candidate_counts.items())),
            "mean_target_rank": round(mean(target_ranks), 3) if target_ranks else None,
            "mean_top_target_margin": round(mean(top_target_margins), 6) if top_target_margins else None,
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
                "Can a saved-prediction candidate top-vs-second score gap identify "
                "action/status decisions that are safe to trust, while failing "
                "closed to defer/insufficient otherwise?"
            ),
            "interpretation_rule": (
                "A useful gate should avoid unsafe trust and improve or preserve "
                "strict final correctness after fail-closed routing. This is a "
                "tiny held-out diagnostic, not deployment calibration."
            ),
            "decision_boundary": (
                "This report uses ignored saved candidate scores only. It is not "
                "new training, not explanation-quality scoring, and not a reason "
                "to start DPO/RLVR."
            ),
        },
    }


def write_markdown(report: Mapping[str, Any], path: Path) -> None:
    summary = report["summary"]
    best = report.get("best_default_zero_unsafe_report") or {}
    adaptive = report.get("adaptive_zero_unsafe_report") or {}
    lines = [
        "# Stage A Saved-Candidate Gate Diagnostic",
        "",
        "Purpose: test whether finite-candidate saved-prediction score gaps can",
        "support a fail-closed boundary gate without publishing prompts, raw",
        "candidate-score JSONL, model state, or scheduler logs.",
        "",
        "## Summary",
        "",
        f"- Run ID: `{report.get('run_id')}`",
        f"- Candidate policy: `{report.get('candidate_policy')}`",
        f"- Cases: {report['cases']}",
        f"- Exact top-1: {summary['exact_top1']}/{report['cases']}",
        f"- Mean target rank: {summary['mean_target_rank']}",
        f"- Mean top-second gap: {summary['mean_top_second_gap']}",
        f"- Top pair counts: `{json.dumps(summary['top_pair_counts'], sort_keys=True)}`",
        f"- Fail-closed pair: `{pair_label(report.get('fail_closed_pair'))}`",
        "",
        "## Gate",
        "",
        f"- Best default zero-unsafe threshold: `{best.get('threshold')}`",
        f"- Trusted at best default threshold: {best.get('trusted')}",
        f"- Trusted incorrect at best default threshold: {best.get('trusted_incorrect')}",
        f"- Strict final correct at best default threshold: {best.get('strict_final_correct')}/{report['cases']}",
        f"- Adaptive zero-unsafe threshold: `{report.get('adaptive_zero_unsafe_threshold')}`",
        f"- Adaptive trusted: {adaptive.get('trusted')}",
        f"- Adaptive strict final correct: {adaptive.get('strict_final_correct')}/{report['cases']}",
        "",
        "## Decision",
        "",
        "This is a fail-closed diagnostic over saved candidate scores only. It",
        "does not reopen `tool_query`, DPO/RLVR, Hugging Face publication, or",
        "release tagging.",
    ]
    path.write_text("\n".join(lines) + "\n")


def parse_thresholds(value: str) -> list[float]:
    thresholds = sorted(set(float(item.strip()) for item in value.split(",") if item.strip()))
    if any(threshold < 0 for threshold in thresholds):
        raise ValueError("thresholds must be non-negative")
    return thresholds


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--out-md", default=None)
    parser.add_argument("--thresholds", default=",".join(str(item) for item in DEFAULT_THRESHOLDS))
    parser.add_argument("--max-top-candidates", type=int, default=3)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    candidates_path = Path(args.candidates)
    report = build_saved_candidate_gate_report(
        load_jsonl(candidates_path),
        candidates_path=candidates_path,
        thresholds=parse_thresholds(args.thresholds),
        max_top_candidates=args.max_top_candidates,
    )
    if args.out_json:
        write_json(args.out_json, report)
    if args.out_md:
        write_markdown(report, Path(args.out_md))
    stdout_report = {
        "dataset": report["dataset"],
        "run_id": report["run_id"],
        "candidate_policy": report["candidate_policy"],
        "cases": report["cases"],
        "summary": report["summary"],
        "best_default_zero_unsafe_report": report["best_default_zero_unsafe_report"],
        "adaptive_zero_unsafe_threshold": report["adaptive_zero_unsafe_threshold"],
        "adaptive_zero_unsafe_report": report["adaptive_zero_unsafe_report"],
    }
    print(json.dumps(stdout_report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

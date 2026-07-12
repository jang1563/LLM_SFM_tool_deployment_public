#!/usr/bin/env python3
"""Analyze fail-closed gates for Stage A enum candidate-score rows.

The input is an ignored raw candidate JSONL artifact from the enum corrective
runner. The output is a compact public-safe report: no prompts, raw logs, model
state, or full hidden metadata are copied.
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

DATASET = "negbiodb_ct_stage_a_enum_candidate_gate_diagnostic_v1"
DEFAULT_THRESHOLDS = (0.0, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2)
FIELD_NAMES = ("action", "evidence_status")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def compact_candidate(candidate: Mapping[str, Any] | None) -> dict[str, str | None]:
    if not isinstance(candidate, Mapping):
        return {"action": None, "evidence_status": None}
    return {
        "action": str(candidate.get("action")) if candidate.get("action") is not None else None,
        "evidence_status": (
            str(candidate.get("evidence_status")) if candidate.get("evidence_status") is not None else None
        ),
    }


def candidate_key(candidate: Mapping[str, Any] | None) -> tuple[str | None, str | None]:
    compact = compact_candidate(candidate)
    return compact["action"], compact["evidence_status"]


def pair_label(candidate: Mapping[str, Any] | None) -> str:
    action, status = candidate_key(candidate)
    return f"{action}/{status}"


def sorted_candidate_scores(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_scores = row.get("candidate_scores")
    if not isinstance(raw_scores, list):
        return []
    scores: list[dict[str, Any]] = []
    for item in raw_scores:
        if not isinstance(item, Mapping):
            continue
        candidate = item.get("candidate")
        if not isinstance(candidate, Mapping):
            continue
        try:
            score = float(item.get("score"))
        except (TypeError, ValueError):
            continue
        scores.append({"candidate": compact_candidate(candidate), "score": score})
    return sorted(
        scores,
        key=lambda item: (-float(item["score"]), json.dumps(item["candidate"], sort_keys=True)),
    )


def find_candidate_rank(
    scores: Sequence[Mapping[str, Any]],
    target: Mapping[str, Any],
) -> tuple[int | None, float | None]:
    target_key = candidate_key(target)
    for index, score_row in enumerate(scores, start=1):
        candidate = score_row.get("candidate")
        if isinstance(candidate, Mapping) and candidate_key(candidate) == target_key:
            return index, float(score_row["score"])
    return None, None


def top_second_gap(scores: Sequence[Mapping[str, Any]]) -> float | None:
    if len(scores) < 2:
        return None
    return round(float(scores[0]["score"]) - float(scores[1]["score"]), 6)


def field_rank(
    scores: Sequence[Mapping[str, Any]],
    *,
    field_name: str,
    target_value: str | None,
) -> dict[str, Any]:
    best_scores: dict[str, float] = {}
    for score_row in scores:
        candidate = score_row.get("candidate")
        if not isinstance(candidate, Mapping):
            continue
        value = candidate.get(field_name)
        if value is None:
            continue
        key = str(value)
        score = float(score_row["score"])
        if key not in best_scores or score > best_scores[key]:
            best_scores[key] = score
    ranked = sorted(best_scores.items(), key=lambda item: (-item[1], item[0]))
    target_key = str(target_value) if target_value is not None else None
    target_rank = None
    target_score = None
    for index, (value, score) in enumerate(ranked, start=1):
        if value == target_key:
            target_rank = index
            target_score = score
            break
    top_value = ranked[0][0] if ranked else None
    top_score = ranked[0][1] if ranked else None
    top_target_margin = None
    if top_score is not None and target_score is not None:
        top_target_margin = round(top_score - target_score, 6)
    return {
        "target_value": target_key,
        "top_value": top_value,
        "target_top1": target_rank == 1,
        "target_rank": target_rank,
        "top_target_margin": top_target_margin,
        "value_count": len(ranked),
    }


def field_pattern(exact_top1: bool, field_ranks: Mapping[str, Mapping[str, Any]]) -> str:
    if exact_top1:
        return "pair_top1"
    action_rank = field_ranks.get("action", {}).get("target_rank")
    status_rank = field_ranks.get("evidence_status", {}).get("target_rank")
    if action_rank is None or status_rank is None:
        return "field_not_retained"
    action_top = action_rank == 1
    status_top = status_rank == 1
    if action_top and status_top:
        return "joint_pair_representation_failure"
    if action_top:
        return "evidence_status_field_failure"
    if status_top:
        return "action_field_failure"
    return "both_field_failure"


def compact_row(row: Mapping[str, Any], *, max_top_candidates: int) -> dict[str, Any]:
    scores = sorted_candidate_scores(row)
    target = compact_candidate(row.get("target_output"))
    top = scores[0] if scores else None
    top_candidate = compact_candidate(top.get("candidate")) if isinstance(top, Mapping) else None
    top_score = float(top["score"]) if isinstance(top, Mapping) else None
    gold_rank, gold_score = find_candidate_rank(scores, target)
    gap = top_second_gap(scores)
    top_gold_margin = None
    if top_score is not None and gold_score is not None:
        top_gold_margin = round(top_score - gold_score, 6)
    exact_top1 = top_candidate is not None and candidate_key(top_candidate) == candidate_key(target)
    field_ranks = {
        field_name: field_rank(
            scores,
            field_name=field_name,
            target_value=target.get(field_name),
        )
        for field_name in FIELD_NAMES
    }
    return {
        "id": row.get("id"),
        "case_id": row.get("case_id"),
        "case_family": row.get("case_family"),
        "chosen_pair": row.get("chosen_pair"),
        "target_output": target,
        "top_candidate": top_candidate,
        "top_pair": pair_label(top_candidate),
        "exact_top1": exact_top1,
        "gold_rank": gold_rank,
        "gold_score": gold_score,
        "top_score": top_score,
        "top_second_gap": gap,
        "top_gold_margin": top_gold_margin,
        "candidate_count": len(scores),
        "field_ranks": field_ranks,
        "field_rank_pattern": field_pattern(exact_top1, field_ranks),
        "top_candidates": scores[:max_top_candidates],
    }


def evaluate_threshold(rows: Sequence[Mapping[str, Any]], threshold: float) -> dict[str, Any]:
    trusted = [
        row for row in rows
        if row.get("top_second_gap") is not None and float(row["top_second_gap"]) >= threshold
    ]
    trusted_correct = [row for row in trusted if row.get("exact_top1")]
    trusted_incorrect = [row for row in trusted if not row.get("exact_top1")]
    fail_closed = [row for row in rows if row not in trusted]
    fail_closed_gold_top1 = [row for row in fail_closed if row.get("exact_top1")]
    return {
        "threshold": threshold,
        "trusted": len(trusted),
        "trusted_correct": len(trusted_correct),
        "trusted_incorrect": len(trusted_incorrect),
        "fail_closed": len(fail_closed),
        "fail_closed_gold_top1": len(fail_closed_gold_top1),
        "coverage": round(len(trusted) / len(rows), 6) if rows else 0.0,
        "trusted_precision": round(len(trusted_correct) / len(trusted), 6) if trusted else None,
        "unsafe_trust_case_ids": [str(row.get("case_id")) for row in trusted_incorrect],
        "missed_correct_case_ids": [str(row.get("case_id")) for row in fail_closed_gold_top1],
    }


def adaptive_zero_false_threshold(rows: Sequence[Mapping[str, Any]]) -> float | None:
    incorrect_gaps = [
        float(row["top_second_gap"])
        for row in rows
        if not row.get("exact_top1") and row.get("top_second_gap") is not None
    ]
    if not incorrect_gaps:
        return 0.0
    return round(max(incorrect_gaps) + 0.000001, 6)


def build_candidate_gate_report(
    candidate_rows: Sequence[Mapping[str, Any]],
    *,
    candidates_path: Path,
    run_id: str | None = None,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    max_top_candidates: int = 3,
) -> dict[str, Any]:
    rows = [compact_row(row, max_top_candidates=max_top_candidates) for row in candidate_rows]
    candidate_counts = Counter(int(row["candidate_count"]) for row in rows)
    exact_top1_rows = [row for row in rows if row.get("exact_top1")]
    gold_ranks = [int(row["gold_rank"]) for row in rows if isinstance(row.get("gold_rank"), int)]
    top_gold_margins = [
        float(row["top_gold_margin"]) for row in rows if row.get("top_gold_margin") is not None
    ]
    top_second_gaps = [
        float(row["top_second_gap"]) for row in rows if row.get("top_second_gap") is not None
    ]
    field_patterns = Counter(str(row["field_rank_pattern"]) for row in rows)
    top_pair_counts = Counter(str(row["top_pair"]) for row in rows)
    action_ranks = [
        int(row["field_ranks"]["action"]["target_rank"])
        for row in rows
        if isinstance(row.get("field_ranks"), Mapping)
        and isinstance(row["field_ranks"].get("action"), Mapping)
        and isinstance(row["field_ranks"]["action"].get("target_rank"), int)
    ]
    status_ranks = [
        int(row["field_ranks"]["evidence_status"]["target_rank"])
        for row in rows
        if isinstance(row.get("field_ranks"), Mapping)
        and isinstance(row["field_ranks"].get("evidence_status"), Mapping)
        and isinstance(row["field_ranks"]["evidence_status"].get("target_rank"), int)
    ]
    threshold_reports = [evaluate_threshold(rows, float(threshold)) for threshold in thresholds]
    adaptive_threshold = adaptive_zero_false_threshold(rows)
    adaptive_report = (
        evaluate_threshold(rows, adaptive_threshold) if adaptive_threshold is not None else None
    )
    zero_false_reports = [report for report in threshold_reports if report["trusted_incorrect"] == 0]
    best_zero_false = None
    if zero_false_reports:
        best_zero_false = sorted(
            zero_false_reports,
            key=lambda report: (-int(report["trusted_correct"]), float(report["threshold"])),
        )[0]

    first_run_id = run_id
    if first_run_id is None:
        for row in candidate_rows:
            if isinstance(row.get("run_id"), str):
                first_run_id = str(row["run_id"])
                break

    return {
        "dataset": DATASET,
        "component": "enum_action",
        "run_id": first_run_id,
        "input_candidates_sha256": sha256_file(candidates_path),
        "candidate_policy": candidate_rows[0].get("enum_candidate_policy") if candidate_rows else None,
        "score_label": candidate_rows[0].get("score_label") if candidate_rows else None,
        "cases": len(rows),
        "summary": {
            "exact_top1": len(exact_top1_rows),
            "candidate_accuracy": round(len(exact_top1_rows) / len(rows), 6) if rows else 0.0,
            "candidate_count_histogram": dict(sorted(candidate_counts.items())),
            "mean_gold_rank": round(mean(gold_ranks), 3) if gold_ranks else None,
            "mean_top_gold_margin": round(mean(top_gold_margins), 6) if top_gold_margins else None,
            "mean_top_second_gap": round(mean(top_second_gaps), 6) if top_second_gaps else None,
            "min_top_second_gap": round(min(top_second_gaps), 6) if top_second_gaps else None,
            "max_top_second_gap": round(max(top_second_gaps), 6) if top_second_gaps else None,
            "top_pair_counts": dict(sorted(top_pair_counts.items())),
            "field_rank_patterns": dict(sorted(field_patterns.items())),
            "action_rank_histogram": dict(sorted(Counter(action_ranks).items())),
            "evidence_status_rank_histogram": dict(sorted(Counter(status_ranks).items())),
        },
        "threshold_reports": threshold_reports,
        "adaptive_zero_false_threshold": adaptive_threshold,
        "adaptive_zero_false_report": adaptive_report,
        "best_default_zero_false_report": best_zero_false,
        "rows": rows,
        "scientific_readout": {
            "diagnostic_question": (
                "Can a top-vs-second score gap identify safe enum candidate top-1 "
                "decisions, or should the runtime fail closed?"
            ),
            "interpretation_rule": (
                "A useful deployment gate should trust some correct rows without "
                "trusting incorrect rows. If every zero-false-trust threshold has "
                "zero useful coverage, score-gap confidence is not calibrated enough "
                "to leave enum_action."
            ),
            "decision_boundary": (
                "This is a post-hoc diagnostic over saved candidate scores, not new "
                "training, DPO, RLVR, or a full trajectory result."
            ),
        },
    }


def write_markdown(report: Mapping[str, Any], path: Path) -> None:
    summary = report["summary"]
    best_zero_false = report.get("best_default_zero_false_report") or {}
    adaptive = report.get("adaptive_zero_false_report") or {}
    lines = [
        "# Stage A Enum Candidate Gate Diagnostic",
        "",
        "Purpose: test whether the enum candidate scorer has enough score-gap",
        "calibration to trust top-1 decisions, while failing closed otherwise.",
        "",
        "## Summary",
        "",
        f"- Run ID: `{report.get('run_id')}`",
        f"- Candidate policy: `{report.get('candidate_policy')}`",
        f"- Cases: {report['cases']}",
        f"- Exact top-1: {summary['exact_top1']}/{report['cases']}",
        f"- Mean gold rank: {summary['mean_gold_rank']}",
        f"- Mean top-second gap: {summary['mean_top_second_gap']}",
        f"- Top pair counts: `{json.dumps(summary['top_pair_counts'], sort_keys=True)}`",
        f"- Field-rank patterns: `{json.dumps(summary['field_rank_patterns'], sort_keys=True)}`",
        "",
        "## Gate Thresholds",
        "",
        "| Threshold | Trusted | Correct trusted | Incorrect trusted | Precision | Fail closed | Missed correct |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["threshold_reports"]:
        precision = row["trusted_precision"] if row["trusted_precision"] is not None else "NA"
        lines.append(
            (
                "| {threshold} | {trusted} | {trusted_correct} | {trusted_incorrect} | "
                "{precision} | {fail_closed} | {missed} |"
            ).format(
                threshold=row["threshold"],
                trusted=row["trusted"],
                trusted_correct=row["trusted_correct"],
                trusted_incorrect=row["trusted_incorrect"],
                precision=precision,
                fail_closed=row["fail_closed"],
                missed=row["fail_closed_gold_top1"],
            )
        )
    lines.extend(
        [
            "",
            "## Fail-Closed Readout",
            "",
            (
                f"- Best default zero-false-trust threshold: "
                f"`{best_zero_false.get('threshold')}` with "
                f"{best_zero_false.get('trusted_correct')} correct trusted rows."
            ),
            (
                f"- Adaptive zero-false threshold: "
                f"`{report.get('adaptive_zero_false_threshold')}` with "
                f"{adaptive.get('trusted_correct')} correct trusted rows and "
                f"{adaptive.get('trusted_incorrect')} incorrect trusted rows."
            ),
            "",
            "## Held-Out Rows",
            "",
            "| Case family | Target | Top candidate | Top-1 | Gold rank | Gap | Field pattern |",
            "| --- | --- | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for row in report["rows"]:
        target = row["target_output"]
        top = row["top_candidate"]
        lines.append(
            (
                "| {case_family} | `{target_action}` / `{target_status}` | "
                "`{top_action}` / `{top_status}` | {exact} | {rank} | {gap} | `{pattern}` |"
            ).format(
                case_family=row.get("case_family"),
                target_action=target.get("action"),
                target_status=target.get("evidence_status"),
                top_action=top.get("action") if isinstance(top, Mapping) else None,
                top_status=top.get("evidence_status") if isinstance(top, Mapping) else None,
                exact=int(bool(row.get("exact_top1"))),
                rank=row.get("gold_rank"),
                gap=row.get("top_second_gap"),
                pattern=row.get("field_rank_pattern"),
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
    parser.add_argument("--max-top-candidates", type=int, default=3)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    candidates_path = Path(args.candidates)
    rows = load_jsonl(candidates_path)
    report = build_candidate_gate_report(
        rows,
        candidates_path=candidates_path,
        run_id=args.run_id,
        thresholds=parse_thresholds(args.thresholds),
        max_top_candidates=args.max_top_candidates,
    )
    if args.out_json:
        write_json(args.out_json, report)
    if args.out_md:
        write_markdown(report, Path(args.out_md))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Field-wise diagnostics for Stage A saved-output candidate scores.

The input is an ignored candidate-score JSONL artifact from
run_stage_a_saved_output_calibration_margin_sft.py. The output is a compact
public-safe report: no prompts, raw model text, scheduler logs, model state, or
full candidate-score tables are copied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from post_training.run_stage_a_sft_smoke_eval import load_jsonl, write_json  # noqa: E402


DATASET = "negbiodb_ct_stage_a_saved_output_candidate_field_diagnostic_v1"
FIELD_NAMES = ("action", "evidence_status")


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


def pair_key(obj: Mapping[str, Any] | None) -> tuple[str | None, str | None]:
    pair = compact_pair(obj)
    return pair["action"], pair["evidence_status"]


def pair_label(obj: Mapping[str, Any] | None) -> str:
    pair = compact_pair(obj)
    return f"{pair['action']}/{pair['evidence_status']}"


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
        raise ValueError(f"invalid saved-output candidate-score JSONL: {preview}{more}")


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


def rank_field_values(
    scores: Sequence[Mapping[str, Any]],
    *,
    field_name: str,
    target_value: str | None,
) -> dict[str, Any]:
    best_scores: dict[str, float] = {}
    for item in scores:
        candidate = item.get("candidate")
        if not isinstance(candidate, Mapping):
            continue
        value = candidate.get(field_name)
        if value is None:
            continue
        key = str(value)
        score = float(item["score"])
        if key not in best_scores or score > best_scores[key]:
            best_scores[key] = score

    ranked = sorted(best_scores.items(), key=lambda item: (-item[1], item[0]))
    top_value = ranked[0][0] if ranked else None
    top_score = ranked[0][1] if ranked else None
    target_key = str(target_value) if target_value is not None else None
    target_rank = None
    target_score = None
    for index, (value, score) in enumerate(ranked, start=1):
        if value == target_key:
            target_rank = index
            target_score = score
            break
    top_target_margin = None
    if top_score is not None and target_score is not None:
        top_target_margin = round(top_score - target_score, 6)
    return {
        "target_value": target_key,
        "top_value": top_value,
        "target_top1": target_rank == 1,
        "target_rank": target_rank,
        "target_score": target_score,
        "top_score": top_score,
        "top_target_margin": top_target_margin,
        "value_count": len(ranked),
    }


def field_rank_pattern(
    *,
    exact_top1: bool,
    action_rank: int | None,
    evidence_status_rank: int | None,
) -> str:
    if exact_top1:
        return "pair_top1"
    if action_rank is None or evidence_status_rank is None:
        return "field_not_retained"
    if action_rank == 1 and evidence_status_rank == 1:
        return "joint_pair_representation_failure"
    if action_rank == 1:
        return "evidence_status_field_failure"
    if evidence_status_rank == 1:
        return "action_field_failure"
    return "both_field_failure"


def compact_field_row(
    row: Mapping[str, Any],
    *,
    max_top_candidates: int,
) -> dict[str, Any]:
    scores = sorted_candidate_scores(row)
    target_pair = compact_pair(row.get("target_pair") if isinstance(row.get("target_pair"), Mapping) else {})
    top = scores[0]
    top_pair = compact_pair(top.get("candidate") if isinstance(top, Mapping) else {})
    target_rank, target_score = find_pair_rank(scores, target_pair)
    top_score = float(top["score"])
    top_target_margin = None
    if target_score is not None:
        top_target_margin = round(top_score - target_score, 6)
    field_ranks = {
        field_name: rank_field_values(
            scores,
            field_name=field_name,
            target_value=target_pair.get(field_name),
        )
        for field_name in FIELD_NAMES
    }
    exact_top1 = pair_key(top_pair) == pair_key(target_pair)
    pattern = field_rank_pattern(
        exact_top1=exact_top1,
        action_rank=field_ranks["action"]["target_rank"],
        evidence_status_rank=field_ranks["evidence_status"]["target_rank"],
    )
    return {
        "case_id": row.get("case_id"),
        "case_family": row.get("case_family"),
        "score_label": row.get("score_label"),
        "target_pair": target_pair,
        "target_pair_label": pair_label(target_pair),
        "top_pair": top_pair,
        "top_pair_label": pair_label(top_pair),
        "exact_top1": exact_top1,
        "target_rank": target_rank,
        "target_score": target_score,
        "top_score": top_score,
        "top_target_margin": top_target_margin,
        "field_ranks": field_ranks,
        "field_rank_pattern": pattern,
        "candidate_count": len(scores),
        "top_candidates": [
            {"score": round(float(item["score"]), 6), "candidate": compact_pair(item.get("candidate"))}
            for item in scores[:max_top_candidates]
        ],
    }


def rank_values(rows: Sequence[Mapping[str, Any]], field_name: str) -> list[int]:
    out: list[int] = []
    for row in rows:
        field = row.get("field_ranks")
        if not isinstance(field, Mapping):
            continue
        rank_info = field.get(field_name)
        if not isinstance(rank_info, Mapping):
            continue
        rank = rank_info.get("target_rank")
        if isinstance(rank, int):
            out.append(rank)
    return out


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "cases": 0,
            "exact_top1": 0,
            "field_diagnostic": {},
            "top_pair_counts": {},
        }

    def mean_or_none(values: Sequence[float]) -> float | None:
        return round(mean(values), 6) if values else None

    target_ranks = [int(row["target_rank"]) for row in rows if isinstance(row.get("target_rank"), int)]
    margins = [float(row["top_target_margin"]) for row in rows if row.get("top_target_margin") is not None]
    top_pair_counts = Counter(str(row.get("top_pair_label")) for row in rows)
    target_pair_counts = Counter(str(row.get("target_pair_label")) for row in rows)
    pattern_counts = Counter(str(row.get("field_rank_pattern")) for row in rows)
    field_diagnostic = {
        "action_top1": sum(
            1
            for row in rows
            if isinstance(row.get("field_ranks"), Mapping)
            and row["field_ranks"].get("action", {}).get("target_top1") is True
        ),
        "evidence_status_top1": sum(
            1
            for row in rows
            if isinstance(row.get("field_ranks"), Mapping)
            and row["field_ranks"].get("evidence_status", {}).get("target_top1") is True
        ),
        "mean_action_rank": mean_or_none([float(rank) for rank in rank_values(rows, "action")]),
        "mean_evidence_status_rank": mean_or_none(
            [float(rank) for rank in rank_values(rows, "evidence_status")]
        ),
        "field_rank_patterns": dict(sorted(pattern_counts.items())),
    }
    by_target_pair: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("target_pair_label"))].append(row)
    for label, items in sorted(grouped.items()):
        by_target_pair[label] = {
            "cases": len(items),
            "exact_top1": sum(1 for item in items if item.get("exact_top1")),
            "top_pair_counts": dict(sorted(Counter(str(item.get("top_pair_label")) for item in items).items())),
            "field_rank_patterns": dict(
                sorted(Counter(str(item.get("field_rank_pattern")) for item in items).items())
            ),
        }
    return {
        "cases": len(rows),
        "exact_top1": sum(1 for row in rows if row.get("exact_top1")),
        "mean_target_rank": mean_or_none([float(rank) for rank in target_ranks]),
        "mean_top_target_margin": mean_or_none(margins),
        "candidate_count_histogram": dict(sorted(Counter(int(row["candidate_count"]) for row in rows).items())),
        "top_pair_counts": dict(sorted(top_pair_counts.items())),
        "target_pair_counts": dict(sorted(target_pair_counts.items())),
        "field_diagnostic": field_diagnostic,
        "by_target_pair": by_target_pair,
    }


def build_saved_output_candidate_field_report(
    candidate_rows: Sequence[Mapping[str, Any]],
    *,
    candidates_path: Path,
    max_top_candidates: int = 3,
) -> dict[str, Any]:
    validate_candidate_rows(candidate_rows)
    rows = [
        compact_field_row(row, max_top_candidates=max_top_candidates)
        for row in candidate_rows
    ]
    first = candidate_rows[0]
    return {
        "dataset": DATASET,
        "run_id": first.get("run_id"),
        "model": first.get("model"),
        "score_label": first.get("score_label"),
        "candidate_policy": first.get("candidate_policy"),
        "candidate_target_format": first.get("candidate_target_format"),
        "prompt_contract": first.get("prompt_contract"),
        "input_candidates_sha256": sha256_file(candidates_path),
        "summary": summarize_rows(rows),
        "rows": rows,
        "boundary": (
            "This report diagnoses action/evidence_status field ranks from ignored "
            "saved-output candidate-score JSONL. It does not publish prompts, raw "
            "model text, scheduler logs, model state, or full candidate-score tables."
        ),
    }


def write_markdown(report: Mapping[str, Any], path: Path) -> None:
    summary = report["summary"]
    field = summary.get("field_diagnostic", {})
    lines = [
        "# Stage A Saved-Output Candidate Field Diagnostic",
        "",
        "Purpose: diagnose whether saved-output candidate failures are action-field,",
        "evidence-status-field, or joint pair-selection failures without publishing",
        "raw candidate-score JSONL.",
        "",
        "## Summary",
        "",
        f"- Run ID: `{report.get('run_id')}`",
        f"- Candidate policy: `{report.get('candidate_policy')}`",
        f"- Candidate target format: `{report.get('candidate_target_format')}`",
        f"- Cases: {summary['cases']}",
        f"- Exact top-1: {summary['exact_top1']}/{summary['cases']}",
        f"- Mean target rank: {summary.get('mean_target_rank')}",
        f"- Mean top-target margin: {summary.get('mean_top_target_margin')}",
        f"- Action field top-1: {field.get('action_top1')}/{summary['cases']}",
        f"- Evidence-status field top-1: {field.get('evidence_status_top1')}/{summary['cases']}",
        f"- Field-rank patterns: `{field.get('field_rank_patterns')}`",
        "",
        "## Top Pair Bias",
        "",
    ]
    for label, count in sorted(summary.get("top_pair_counts", {}).items()):
        lines.append(f"- `{label}`: {count}")
    lines.extend(
        [
            "",
            "## Rows",
            "",
            "| Case | Target | Top | Pair rank | Action rank | Status rank | Pattern |",
            "| --- | --- | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for row in report["rows"]:
        action_rank = row["field_ranks"]["action"]["target_rank"]
        status_rank = row["field_ranks"]["evidence_status"]["target_rank"]
        lines.append(
            f"| `{row['case_id']}` | `{row['target_pair_label']}` | "
            f"`{row['top_pair_label']}` | {row['target_rank']} | "
            f"{action_rank} | {status_rank} | `{row['field_rank_pattern']}` |"
        )
    lines.extend(
        [
            "",
            "This is a compact diagnostic over saved candidate scores only. It is",
            "not free-form generation, DPO/RLVR, or a deployment-readiness result.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--out-md", default=None)
    parser.add_argument("--max-top-candidates", type=int, default=3)
    args = parser.parse_args()

    candidates_path = Path(args.candidates)
    report = build_saved_output_candidate_field_report(
        load_jsonl(candidates_path),
        candidates_path=candidates_path,
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
        "candidate_target_format": report["candidate_target_format"],
        "summary": report["summary"],
    }
    print(json.dumps(stdout_report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

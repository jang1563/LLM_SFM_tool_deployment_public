#!/usr/bin/env python3
"""Train-derived calibration diagnostic for saved-output candidate scores.

This script consumes ignored candidate-score JSONL artifacts from
run_stage_a_saved_output_calibration_margin_sft.py. It estimates candidate-pair
mean scores from train rows only, applies pair-mean centering to held-out rows,
and emits a compact public-safe report. It does not tune thresholds on held-out
scores and does not publish prompts, raw model text, scheduler logs, model
state, or full candidate-score tables.
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


DATASET = "negbiodb_ct_stage_a_saved_output_candidate_calibration_v1"
CALIBRATION_MODE = "pair_mean_center"
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


def validate_candidate_rows(rows: Sequence[Mapping[str, Any]], *, label: str) -> None:
    if not rows:
        raise ValueError(f"{label} candidate-score JSONL is empty")
    issues: list[str] = []
    for index, row in enumerate(rows, start=1):
        case_id = row.get("case_id", f"{label}[{index}]")
        target = row.get("target_pair")
        if not isinstance(target, Mapping) or None in pair_key(target):
            issues.append(f"{case_id}: missing target_pair action/evidence_status")
        if len(sorted_candidate_scores(row)) < 2:
            issues.append(f"{case_id}: candidate_scores must contain at least two scored candidates")
    if issues:
        preview = "; ".join(issues[:5])
        more = f"; ... {len(issues) - 5} more" if len(issues) > 5 else ""
        raise ValueError(f"invalid {label} candidate-score JSONL: {preview}{more}")


def unique_value(rows: Sequence[Mapping[str, Any]], key: str) -> Any:
    values = {row.get(key) for row in rows}
    if len(values) != 1:
        raise ValueError(f"{key} must be constant across candidate rows: {sorted(map(str, values))}")
    return next(iter(values))


def validate_compatible_artifacts(
    train_rows: Sequence[Mapping[str, Any]],
    heldout_rows: Sequence[Mapping[str, Any]],
) -> None:
    validate_candidate_rows(train_rows, label="train")
    validate_candidate_rows(heldout_rows, label="heldout")
    for key in ("candidate_policy", "candidate_target_format"):
        train_value = unique_value(train_rows, key)
        heldout_value = unique_value(heldout_rows, key)
        if train_value != heldout_value:
            raise ValueError(f"{key} mismatch: train={train_value!r} heldout={heldout_value!r}")


def estimate_pair_mean_scores(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        for item in sorted_candidate_scores(row):
            values[pair_label(item["candidate"])].append(float(item["score"]))
    return {
        label: {
            "count": len(scores),
            "mean_score": round(mean(scores), 6),
        }
        for label, scores in sorted(values.items())
    }


def rank_candidates(
    row: Mapping[str, Any],
    *,
    pair_mean_scores: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    scores = sorted_candidate_scores(row)
    adjusted_scores: list[dict[str, Any]] = []
    for item in scores:
        label = pair_label(item["candidate"])
        score = float(item["score"])
        adjusted = score
        if pair_mean_scores is not None:
            if label not in pair_mean_scores:
                raise ValueError(f"missing calibration pair: {label}")
            adjusted = score - float(pair_mean_scores[label]["mean_score"])
        adjusted_scores.append(
            {
                "score": adjusted,
                "candidate": item["candidate"],
                "pair_label": label,
            }
        )
    adjusted_scores.sort(key=lambda item: (-float(item["score"]), json.dumps(item["candidate"], sort_keys=True)))
    target_pair = compact_pair(row.get("target_pair") if isinstance(row.get("target_pair"), Mapping) else {})
    target_label = pair_label(target_pair)
    top = adjusted_scores[0]
    top_second_gap = None
    if len(adjusted_scores) >= 2:
        top_second_gap = round(float(adjusted_scores[0]["score"]) - float(adjusted_scores[1]["score"]), 6)
    target_rank = None
    target_score = None
    for index, item in enumerate(adjusted_scores, start=1):
        if pair_key(item["candidate"]) == pair_key(target_pair):
            target_rank = index
            target_score = float(item["score"])
            break
    top_score = float(top["score"])
    top_target_margin = None
    if target_score is not None:
        top_target_margin = round(top_score - target_score, 6)
    exact_top1 = pair_key(top["candidate"]) == pair_key(target_pair)
    return {
        "target_pair": target_pair,
        "target_pair_label": target_label,
        "top_pair": compact_pair(top["candidate"]),
        "top_pair_label": str(top["pair_label"]),
        "exact_top1": exact_top1,
        "target_rank": target_rank,
        "top_second_gap": top_second_gap,
        "top_target_margin": top_target_margin,
    }


def summarize_ranked_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "cases": 0,
            "exact_top1": 0,
            "exact_top1_accuracy": 0.0,
            "mean_target_rank": None,
            "mean_top_target_margin": None,
            "mean_top_second_gap": None,
            "top_pair_counts": {},
            "target_pair_counts": {},
        }
    ranks = [float(row["target_rank"]) for row in rows if isinstance(row.get("target_rank"), int)]
    margins = [
        float(row["top_target_margin"])
        for row in rows
        if row.get("top_target_margin") is not None
    ]
    gaps = [
        float(row["top_second_gap"])
        for row in rows
        if row.get("top_second_gap") is not None
    ]
    exact = sum(1 for row in rows if row.get("exact_top1"))
    return {
        "cases": len(rows),
        "exact_top1": exact,
        "exact_top1_accuracy": round(exact / len(rows), 3),
        "mean_target_rank": round(mean(ranks), 6) if ranks else None,
        "mean_top_target_margin": round(mean(margins), 6) if margins else None,
        "mean_top_second_gap": round(mean(gaps), 6) if gaps else None,
        "top_pair_counts": dict(sorted(Counter(str(row.get("top_pair_label")) for row in rows).items())),
        "target_pair_counts": dict(sorted(Counter(str(row.get("target_pair_label")) for row in rows).items())),
    }


def train_selected_zero_unsafe_threshold(rows: Sequence[Mapping[str, Any]]) -> float | None:
    wrong_gaps = [
        float(row["top_second_gap"])
        for row in rows
        if row.get("exact_top1") is not True and row.get("top_second_gap") is not None
    ]
    if not wrong_gaps:
        return 0.0
    return round(max(wrong_gaps) + 0.000001, 6)


def evaluate_fail_closed_gate(
    rows: Sequence[Mapping[str, Any]],
    *,
    threshold: float,
    fail_closed_pair: Mapping[str, Any] = DEFAULT_FAIL_CLOSED_PAIR,
) -> dict[str, Any]:
    fail_closed = compact_pair(fail_closed_pair)
    trusted = [
        row
        for row in rows
        if row.get("top_second_gap") is not None and float(row["top_second_gap"]) >= threshold
    ]
    failed_closed = [row for row in rows if row not in trusted]
    trusted_correct = [row for row in trusted if row.get("exact_top1") is True]
    trusted_incorrect = [row for row in trusted if row.get("exact_top1") is not True]
    fail_closed_correct = [
        row for row in failed_closed if pair_key(row.get("target_pair")) == pair_key(fail_closed)
    ]
    strict_final_correct = len(trusted_correct) + len(fail_closed_correct)
    return {
        "threshold": threshold,
        "fail_closed_pair": fail_closed,
        "trusted": len(trusted),
        "trusted_correct": len(trusted_correct),
        "trusted_incorrect": len(trusted_incorrect),
        "fail_closed": len(failed_closed),
        "fail_closed_exact_correct": len(fail_closed_correct),
        "strict_final_correct": strict_final_correct,
        "coverage": round(len(trusted) / len(rows), 6) if rows else 0.0,
        "trusted_precision": round(len(trusted_correct) / len(trusted), 6) if trusted else None,
        "strict_final_accuracy": round(strict_final_correct / len(rows), 6) if rows else 0.0,
        "trusted_case_ids": [str(row.get("case_id")) for row in trusted],
        "unsafe_trust_case_ids": [str(row.get("case_id")) for row in trusted_incorrect],
        "fail_closed_case_ids": [str(row.get("case_id")) for row in failed_closed],
    }


def build_calibration_report(
    train_rows: Sequence[Mapping[str, Any]],
    heldout_rows: Sequence[Mapping[str, Any]],
    *,
    train_candidates_path: Path,
    heldout_candidates_path: Path,
) -> dict[str, Any]:
    validate_compatible_artifacts(train_rows, heldout_rows)
    pair_mean_scores = estimate_pair_mean_scores(train_rows)
    train_rank_rows = [
        {
            "case_id": row.get("case_id"),
            **rank_candidates(row),
        }
        for row in train_rows
    ]
    calibrated_train_rows = [
        {
            "case_id": row.get("case_id"),
            **rank_candidates(row, pair_mean_scores=pair_mean_scores),
        }
        for row in train_rows
    ]
    raw_rows = [
        {
            "case_id": row.get("case_id"),
            **rank_candidates(row),
        }
        for row in heldout_rows
    ]
    calibrated_rows = [
        {
            "case_id": row.get("case_id"),
            **rank_candidates(row, pair_mean_scores=pair_mean_scores),
        }
        for row in heldout_rows
    ]
    row_reports = []
    for raw, calibrated in zip(raw_rows, calibrated_rows, strict=True):
        row_reports.append(
            {
                "case_id": raw.get("case_id"),
                "target_pair_label": raw.get("target_pair_label"),
                "raw_top_pair_label": raw.get("top_pair_label"),
                "raw_target_rank": raw.get("target_rank"),
                "raw_top_target_margin": raw.get("top_target_margin"),
                "calibrated_top_pair_label": calibrated.get("top_pair_label"),
                "calibrated_target_rank": calibrated.get("target_rank"),
                "calibrated_top_second_gap": calibrated.get("top_second_gap"),
                "calibrated_top_target_margin": calibrated.get("top_target_margin"),
                "calibrated_exact_top1": calibrated.get("exact_top1"),
            }
        )
    first_train = train_rows[0]
    first_heldout = heldout_rows[0]
    train_selected_threshold = train_selected_zero_unsafe_threshold(calibrated_train_rows)
    train_selected_gate_report = (
        evaluate_fail_closed_gate(
            calibrated_rows,
            threshold=train_selected_threshold,
            fail_closed_pair=DEFAULT_FAIL_CLOSED_PAIR,
        )
        if train_selected_threshold is not None
        else None
    )
    return {
        "dataset": DATASET,
        "calibration_mode": CALIBRATION_MODE,
        "run_id": first_heldout.get("run_id"),
        "model": first_heldout.get("model"),
        "train_score_label": first_train.get("score_label"),
        "heldout_score_label": first_heldout.get("score_label"),
        "candidate_policy": first_heldout.get("candidate_policy"),
        "candidate_target_format": first_heldout.get("candidate_target_format"),
        "input_train_candidates_sha256": sha256_file(train_candidates_path),
        "input_heldout_candidates_sha256": sha256_file(heldout_candidates_path),
        "pair_mean_scores": pair_mean_scores,
        "train_summary": summarize_ranked_rows(train_rank_rows),
        "calibrated_train_summary": summarize_ranked_rows(calibrated_train_rows),
        "raw_heldout_summary": summarize_ranked_rows(raw_rows),
        "calibrated_heldout_summary": summarize_ranked_rows(calibrated_rows),
        "train_selected_zero_unsafe_threshold": train_selected_threshold,
        "train_selected_gate_report": train_selected_gate_report,
        "rows": row_reports,
        "boundary": (
            "Pair-mean centering is estimated from train candidate scores only "
            "and applied once to held-out candidates. The optional score-gap "
            "gate threshold is selected on calibrated train rows only. This is "
            "a diagnostic, not held-out tuning, DPO/RLVR, or a deployment-"
            "readiness claim."
        ),
    }


def write_markdown(report: Mapping[str, Any], path: Path) -> None:
    train = report["train_summary"]
    calibrated_train = report["calibrated_train_summary"]
    raw = report["raw_heldout_summary"]
    calibrated = report["calibrated_heldout_summary"]
    gate = report.get("train_selected_gate_report") or {}
    lines = [
        "# Stage A Saved-Output Candidate Calibration Diagnostic",
        "",
        "Purpose: test whether train-derived pair-mean centering reduces the",
        "saved-output finite-candidate prior without tuning on held-out ranks.",
        "",
        "## Summary",
        "",
        f"- Run ID: `{report.get('run_id')}`",
        f"- Calibration mode: `{report.get('calibration_mode')}`",
        f"- Candidate policy: `{report.get('candidate_policy')}`",
        f"- Candidate target format: `{report.get('candidate_target_format')}`",
        "",
        "| Slice | Exact top-1 | Mean target rank | Top-pair counts |",
        "| --- | ---: | ---: | --- |",
        (
            f"| Train raw | {train['exact_top1']}/{train['cases']} | "
            f"{train['mean_target_rank']} | `{train['top_pair_counts']}` |"
        ),
        (
            f"| Train calibrated | {calibrated_train['exact_top1']}/{calibrated_train['cases']} | "
            f"{calibrated_train['mean_target_rank']} | `{calibrated_train['top_pair_counts']}` |"
        ),
        (
            f"| Held-out raw | {raw['exact_top1']}/{raw['cases']} | "
            f"{raw['mean_target_rank']} | `{raw['top_pair_counts']}` |"
        ),
        (
            f"| Held-out calibrated | {calibrated['exact_top1']}/{calibrated['cases']} | "
            f"{calibrated['mean_target_rank']} | `{calibrated['top_pair_counts']}` |"
        ),
        "",
        "## Train-Selected Gate",
        "",
        f"- Train-selected zero-unsafe threshold: `{report.get('train_selected_zero_unsafe_threshold')}`",
        f"- Held-out trusted rows: {gate.get('trusted')}",
        f"- Held-out unsafe trusted rows: {gate.get('trusted_incorrect')}",
        f"- Held-out strict final correct after fail-closed routing: {gate.get('strict_final_correct')}/{calibrated['cases']}",
        f"- Fail-closed pair: `{pair_label(gate.get('fail_closed_pair'))}`",
        "",
        "## Rows",
        "",
        "| Case | Target | Raw top | Raw rank | Calibrated top | Calibrated rank |",
        "| --- | --- | --- | ---: | --- | ---: |",
    ]
    for row in report["rows"]:
        lines.append(
            f"| `{row['case_id']}` | `{row['target_pair_label']}` | "
            f"`{row['raw_top_pair_label']}` | {row['raw_target_rank']} | "
            f"`{row['calibrated_top_pair_label']}` | {row['calibrated_target_rank']} |"
        )
    lines.extend(
        [
            "",
            "This report is compact and public-safe. It does not publish prompts,",
        "raw model text, scheduler logs, model state, or full candidate-score",
            "tables. Treat the result as a diagnostic only. Calibration and",
            "score-gap gating here are not a replacement for runtime evidence",
            "arbitration.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-candidates", required=True)
    parser.add_argument("--heldout-candidates", required=True)
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--out-md", default=None)
    args = parser.parse_args()

    train_candidates_path = Path(args.train_candidates)
    heldout_candidates_path = Path(args.heldout_candidates)
    report = build_calibration_report(
        load_jsonl(train_candidates_path),
        load_jsonl(heldout_candidates_path),
        train_candidates_path=train_candidates_path,
        heldout_candidates_path=heldout_candidates_path,
    )
    if args.out_json:
        write_json(args.out_json, report)
    if args.out_md:
        write_markdown(report, Path(args.out_md))
    stdout_report = {
        "dataset": report["dataset"],
        "run_id": report["run_id"],
        "calibration_mode": report["calibration_mode"],
        "candidate_policy": report["candidate_policy"],
        "candidate_target_format": report["candidate_target_format"],
        "train_summary": report["train_summary"],
        "calibrated_train_summary": report["calibrated_train_summary"],
        "raw_heldout_summary": report["raw_heldout_summary"],
        "calibrated_heldout_summary": report["calibrated_heldout_summary"],
        "train_selected_zero_unsafe_threshold": report["train_selected_zero_unsafe_threshold"],
        "train_selected_gate_report": report["train_selected_gate_report"],
    }
    print(json.dumps(stdout_report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

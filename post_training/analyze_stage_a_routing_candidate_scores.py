#!/usr/bin/env python3
"""Summarize Stage A routing candidate ranks from saved predictions.

This script reads ignored raw prediction artifacts and emits compact,
public-safe rank/margin summaries. It intentionally does not copy prompts,
logs, model state, or full hidden metadata into the report.
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

from post_training.run_stage_a_strict_component_sft_smoke import (  # noqa: E402
    DATASET as COMPONENT_SFT_DATASET,
    component_case_id,
    filter_component,
    load_jsonl,
    parse_component_output,
    routing_observed_pair_outputs,
    target_output_from_row,
    write_json,
)

DIAGNOSTIC_DATASET = "negbiodb_ct_stage_a_routing_candidate_rank_diagnostic_v1"
FIELD_RANK_NAMES = ("action", "evidence_status")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compact_candidate(candidate: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(candidate, Mapping):
        return {"action": None, "evidence_status": None, "cited_source_ids": []}
    citations = candidate.get("cited_source_ids")
    return {
        "action": str(candidate.get("action")) if candidate.get("action") is not None else None,
        "evidence_status": (
            str(candidate.get("evidence_status")) if candidate.get("evidence_status") is not None else None
        ),
        "cited_source_ids": [str(item) for item in citations] if isinstance(citations, list) else [],
    }


def candidate_key(candidate: Mapping[str, Any] | None) -> tuple[str | None, str | None, tuple[str, ...]]:
    compact = compact_candidate(candidate)
    return (
        compact["action"],
        compact["evidence_status"],
        tuple(compact["cited_source_ids"]),
    )


def action_status_key(candidate: Mapping[str, Any] | None) -> tuple[str | None, str | None]:
    compact = compact_candidate(candidate)
    return compact["action"], compact["evidence_status"]


def pair_label(candidate: Mapping[str, Any] | None) -> str:
    action, status = action_status_key(candidate)
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


def prediction_from_row(row: Mapping[str, Any]) -> dict[str, Any] | None:
    payload = row.get("prediction", row.get("raw_output"))
    parsed, _ = parse_component_output(payload)
    if parsed is None:
        return None
    return compact_candidate(parsed)


def prediction_lookup(prediction_rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    out: dict[str, Mapping[str, Any]] = {}
    for row in prediction_rows:
        key = row.get("source_component_target_id") or row.get("id")
        if isinstance(key, str) and key:
            out[key] = row
    return out


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


def find_action_status_rank(
    scores: Sequence[Mapping[str, Any]],
    target: Mapping[str, Any],
) -> tuple[int | None, float | None]:
    target_key = action_status_key(target)
    for index, score_row in enumerate(scores, start=1):
        candidate = score_row.get("candidate")
        if isinstance(candidate, Mapping) and action_status_key(candidate) == target_key:
            return index, float(score_row["score"])
    return None, None


def rank_field_values(
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


def field_rank_value(row: Mapping[str, Any], field_name: str) -> int | None:
    ranks = row.get("field_ranks")
    if not isinstance(ranks, Mapping):
        return None
    field = ranks.get(field_name)
    if not isinstance(field, Mapping):
        return None
    rank = field.get("target_rank")
    return int(rank) if isinstance(rank, int) else None


def field_rank_pattern(
    *,
    exact_top1: bool,
    action_status_top1: bool,
    action_rank: int | None,
    evidence_status_rank: int | None,
) -> str:
    if exact_top1:
        return "exact_top1"
    if action_status_top1:
        return "citation_failure"
    if action_rank is None or evidence_status_rank is None:
        return "field_not_retained"
    if action_rank == 1 and evidence_status_rank == 1:
        return "joint_pair_representation_failure"
    if action_rank == 1:
        return "evidence_status_field_failure"
    if evidence_status_rank == 1:
        return "action_field_failure"
    return "both_field_failure"


def build_routing_candidate_rank_report(
    *,
    expected_rows: Sequence[Mapping[str, Any]],
    prediction_rows: Sequence[Mapping[str, Any]],
    predictions_path: Path,
    train_rows: Sequence[Mapping[str, Any]] | None = None,
    run_id: str | None = None,
    max_top_candidates: int = 3,
) -> dict[str, Any]:
    expected_by_id = {str(row["id"]): row for row in expected_rows}
    predictions_by_id = prediction_lookup(prediction_rows)
    candidate_outputs = routing_observed_pair_outputs(train_rows or expected_rows)
    candidate_space_size = len(candidate_outputs)
    rows: list[dict[str, Any]] = []

    for row_id, expected in expected_by_id.items():
        target = compact_candidate(target_output_from_row(expected))
        prediction = predictions_by_id.get(row_id)
        if prediction is None:
            rows.append(
                {
                    "source_component_target_id": row_id,
                    "case_id": component_case_id(expected),
                    "case_family": expected.get("case_family"),
                    "expected": target,
                    "predicted": None,
                    "top_candidate": None,
                    "exact_top1": False,
                    "action_status_top1": False,
                    "candidate_scores_present": False,
                    "retained_candidate_count": 0,
                    "candidate_space_size": candidate_space_size,
                    "all_candidates_retained": False,
                    "gold_in_retained_candidates": False,
                    "gold_rank": None,
                    "action_status_rank": None,
                    "gold_score": None,
                    "top_score": None,
                    "top_gold_margin": None,
                    "field_ranks": {},
                    "field_rank_pattern": "missing_prediction",
                    "top_candidates": [],
                }
            )
            continue

        scores = sorted_candidate_scores(prediction)
        predicted = prediction_from_row(prediction)
        top = scores[0] if scores else None
        top_candidate = top["candidate"] if top else predicted
        top_score = float(top["score"]) if top else None
        gold_rank, gold_score = find_candidate_rank(scores, target)
        pair_rank, _ = find_action_status_rank(scores, target)
        top_gold_margin = None
        if top_score is not None and gold_score is not None:
            top_gold_margin = round(top_score - gold_score, 6)
        exact_top1 = top_candidate is not None and candidate_key(top_candidate) == candidate_key(target)
        action_status_top1 = (
            top_candidate is not None
            and action_status_key(top_candidate) == action_status_key(target)
        )
        field_ranks = {
            field_name: rank_field_values(
                scores,
                field_name=field_name,
                target_value=target.get(field_name),
            )
            for field_name in FIELD_RANK_NAMES
        } if scores else {}
        action_rank = field_ranks.get("action", {}).get("target_rank")
        evidence_status_rank = field_ranks.get("evidence_status", {}).get("target_rank")
        rows.append(
            {
                "source_component_target_id": row_id,
                "case_id": component_case_id(expected),
                "case_family": expected.get("case_family"),
                "expected": target,
                "predicted": predicted,
                "top_candidate": top_candidate,
                "exact_top1": exact_top1,
                "action_status_top1": action_status_top1,
                "candidate_scores_present": bool(scores),
                "retained_candidate_count": len(scores),
                "candidate_space_size": candidate_space_size,
                "all_candidates_retained": len(scores) == candidate_space_size,
                "gold_in_retained_candidates": gold_rank is not None,
                "gold_rank": gold_rank,
                "action_status_rank": pair_rank,
                "gold_score": gold_score,
                "top_score": top_score,
                "top_gold_margin": top_gold_margin,
                "field_ranks": field_ranks,
                "field_rank_pattern": field_rank_pattern(
                    exact_top1=exact_top1,
                    action_status_top1=action_status_top1,
                    action_rank=action_rank if isinstance(action_rank, int) else None,
                    evidence_status_rank=evidence_status_rank
                    if isinstance(evidence_status_rank, int)
                    else None,
                ),
                "top_candidates": scores[:max_top_candidates],
            }
        )

    first_run_id = run_id
    if first_run_id is None:
        for row in prediction_rows:
            if isinstance(row.get("run_id"), str):
                first_run_id = str(row["run_id"])
                break

    gold_ranks = [int(row["gold_rank"]) for row in rows if isinstance(row.get("gold_rank"), int)]
    action_status_ranks = [
        int(row["action_status_rank"])
        for row in rows
        if isinstance(row.get("action_status_rank"), int)
    ]
    margins = [float(row["top_gold_margin"]) for row in rows if row.get("top_gold_margin") is not None]
    retained_counts = Counter(int(row["retained_candidate_count"]) for row in rows)
    field_patterns = Counter(str(row.get("field_rank_pattern")) for row in rows)
    top_pair_counts = Counter(
        pair_label(row.get("top_candidate"))
        for row in rows
        if isinstance(row.get("top_candidate"), Mapping)
    )
    target_pair_counts = Counter(pair_label(row.get("expected")) for row in rows)
    action_rank_values = [
        rank for row in rows if (rank := field_rank_value(row, "action")) is not None
    ]
    status_rank_values = [
        rank for row in rows if (rank := field_rank_value(row, "evidence_status")) is not None
    ]
    citation_required = [
        row for row in rows
        if isinstance(row.get("expected"), Mapping) and row["expected"].get("cited_source_ids")
    ]
    by_case_family: dict[str, dict[str, Any]] = {}
    for case_family in sorted({str(row.get("case_family")) for row in rows}):
        family_rows = [row for row in rows if str(row.get("case_family")) == case_family]
        by_case_family[case_family] = {
            "cases": len(family_rows),
            "exact_top1": sum(1 for row in family_rows if row.get("exact_top1")),
            "action_status_top1": sum(1 for row in family_rows if row.get("action_status_top1")),
            "field_rank_patterns": dict(
                sorted(Counter(str(row.get("field_rank_pattern")) for row in family_rows).items())
            ),
        }

    return {
        "dataset": DIAGNOSTIC_DATASET,
        "source_dataset": COMPONENT_SFT_DATASET,
        "component": "routing_after_loop",
        "run_id": first_run_id,
        "candidate_policy": "train_observed_routing_pairs_with_visible_citations",
        "candidate_outputs": candidate_outputs,
        "input_predictions_sha256": sha256_file(predictions_path),
        "candidate_space_size": candidate_space_size,
        "cases_expected": len(expected_rows),
        "predictions_received": len(prediction_rows),
        "summary": {
            "cases": len(rows),
            "exact_top1": sum(1 for row in rows if row.get("exact_top1")),
            "action_status_top1": sum(1 for row in rows if row.get("action_status_top1")),
            "citation_required_cases": len(citation_required),
            "citation_required_exact_top1": sum(1 for row in citation_required if row.get("exact_top1")),
            "gold_in_retained_candidates": sum(1 for row in rows if row.get("gold_in_retained_candidates")),
            "all_candidates_retained_cases": sum(1 for row in rows if row.get("all_candidates_retained")),
            "missing_candidate_scores": sum(1 for row in rows if not row.get("candidate_scores_present")),
            "retained_candidate_count_histogram": dict(sorted(retained_counts.items())),
            "gold_rank_histogram": dict(sorted(Counter(gold_ranks).items())),
            "action_status_rank_histogram": dict(sorted(Counter(action_status_ranks).items())),
            "mean_gold_rank_observed": round(mean(gold_ranks), 3) if gold_ranks else None,
            "mean_action_status_rank_observed": round(mean(action_status_ranks), 3)
            if action_status_ranks
            else None,
            "mean_top_gold_margin_observed": round(mean(margins), 6) if margins else None,
            "top_pair_counts": dict(sorted(top_pair_counts.items())),
            "target_pair_counts": dict(sorted(target_pair_counts.items())),
            "field_diagnostic": {
                "action_top1": sum(1 for rank in action_rank_values if rank == 1),
                "evidence_status_top1": sum(1 for rank in status_rank_values if rank == 1),
                "action_rank_histogram": dict(sorted(Counter(action_rank_values).items())),
                "evidence_status_rank_histogram": dict(sorted(Counter(status_rank_values).items())),
                "field_rank_patterns": dict(sorted(field_patterns.items())),
                "by_case_family": by_case_family,
            },
        },
        "rows": rows,
        "scientific_readout": {
            "diagnostic_question": (
                "After free-form routing schema failures are removed, does the model "
                "rank the full routing target, including citations, above train-observed "
                "action/status alternatives?"
            ),
            "interpretation_rule": (
                "If action_status_top1 is high but exact_top1 is low, treat the next "
                "repair as citation grounding. If both are low, routing action/status "
                "selection remains the bottleneck. Do not use DPO/RLVR until this "
                "component readout has held-out pass rate, mean score, and violation counts."
            ),
            "retention_note": (
                "The report keeps only compact ranks, margins, and candidate labels. "
                "Raw prediction JSONL and full candidate-score tables should remain under "
                "ignored post_training/runs artifacts unless intentionally curated."
            ),
        },
    }


def write_markdown(report: Mapping[str, Any], path: Path) -> None:
    summary = report["summary"]
    field_summary = summary.get("field_diagnostic", {})
    pattern_json = json.dumps(field_summary.get("field_rank_patterns", {}), sort_keys=True)
    lines = [
        "# Stage A Routing Candidate Rank Diagnostic",
        "",
        "Purpose: compactly diagnose whether constrained routing ranks the full",
        "`(action, evidence_status, cited_source_ids)` target above train-observed",
        "routing alternatives without publishing prompts, logs, model state, or full",
        "candidate-score tables.",
        "",
        "## Summary",
        "",
        f"- Run ID: `{report.get('run_id')}`",
        f"- Candidate policy: `{report.get('candidate_policy')}`",
        f"- Cases: {summary['cases']}",
        f"- Candidate space size: {report['candidate_space_size']}",
        f"- Exact top-1: {summary['exact_top1']}/{summary['cases']}",
        f"- Action/status top-1: {summary['action_status_top1']}/{summary['cases']}",
        f"- Citation-required exact top-1: {summary['citation_required_exact_top1']}/"
        f"{summary['citation_required_cases']}",
        f"- Mean observed full-target rank: {summary['mean_gold_rank_observed']}",
        f"- Mean observed action/status rank: {summary['mean_action_status_rank_observed']}",
        f"- Field-rank patterns: `{pattern_json}`",
        "",
        "## Held-Out Rank Readout",
        "",
        "| Case family | Expected | Top candidate | Full rank | Pair rank | Margin | Pattern |",
        "| --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in report["rows"]:
        expected = row["expected"]
        top = row.get("top_candidate") or {}
        expected_text = (
            f"`{expected.get('action')}` / `{expected.get('evidence_status')}` / "
            f"`{','.join(expected.get('cited_source_ids', [])) or 'none'}`"
        )
        top_text = (
            f"`{top.get('action')}` / `{top.get('evidence_status')}` / "
            f"`{','.join(top.get('cited_source_ids', [])) or 'none'}`"
            if top
            else "`missing`"
        )
        lines.append(
            "| {case_family} | {expected} | {top} | {gold_rank} | {pair_rank} | {margin} | `{pattern}` |".format(
                case_family=row.get("case_family"),
                expected=expected_text,
                top=top_text,
                gold_rank=row.get("gold_rank") if row.get("gold_rank") is not None else "NA",
                pair_rank=row.get("action_status_rank")
                if row.get("action_status_rank") is not None
                else "NA",
                margin=row.get("top_gold_margin") if row.get("top_gold_margin") is not None else "NA",
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
            str(report["scientific_readout"]["retention_note"]),
            "",
            "## Trace",
            "",
            f"- Input predictions SHA-256: `{report['input_predictions_sha256']}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument(
        "--expected-targets",
        default="post_training/stage_a_evidence_conditioned_component_targets_heldout_v1.jsonl",
    )
    parser.add_argument(
        "--train-targets",
        default="post_training/stage_a_evidence_conditioned_component_targets_train_v1.jsonl",
    )
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--out-md", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--max-top-candidates", type=int, default=3)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    predictions_path = Path(args.predictions)
    expected_rows = filter_component(load_jsonl(args.expected_targets), "routing_after_loop")
    train_rows = filter_component(load_jsonl(args.train_targets), "routing_after_loop")
    prediction_rows = load_jsonl(predictions_path)
    report = build_routing_candidate_rank_report(
        expected_rows=expected_rows,
        prediction_rows=prediction_rows,
        predictions_path=predictions_path,
        train_rows=train_rows,
        run_id=args.run_id,
        max_top_candidates=args.max_top_candidates,
    )
    if args.out_json:
        write_json(args.out_json, report)
    if args.out_md:
        write_markdown(report, Path(args.out_md))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

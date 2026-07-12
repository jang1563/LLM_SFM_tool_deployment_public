#!/usr/bin/env python3
"""Summarize Stage A enum-action candidate ranks from saved predictions.

This script reads ignored raw prediction artifacts and emits compact,
public-safe rank/margin summaries. It intentionally does not copy prompts,
logs, model state, or full candidate-score tables into the report.
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
    enum_action_candidate_outputs,
    enum_action_observed_pair_outputs,
    filter_component,
    load_jsonl,
    parse_component_output,
    target_output_from_row,
    write_json,
)

DIAGNOSTIC_DATASET = "negbiodb_ct_stage_a_enum_candidate_rank_diagnostic_v1"
CANDIDATE_POLICIES = ("all_retained", "train_observed_pairs")
FIELD_RANK_NAMES = ("action", "evidence_status")
FIELD_NEAR_TOP_RANK = 2


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compact_candidate(candidate: Mapping[str, Any]) -> dict[str, str | None]:
    return {
        "action": str(candidate.get("action")) if candidate.get("action") is not None else None,
        "evidence_status": (
            str(candidate.get("evidence_status")) if candidate.get("evidence_status") is not None else None
        ),
    }


def candidate_key(candidate: Mapping[str, Any]) -> tuple[str | None, str | None]:
    compact = compact_candidate(candidate)
    return compact.get("action"), compact.get("evidence_status")


def candidate_key_set(candidates: Sequence[Mapping[str, Any]]) -> set[tuple[str | None, str | None]]:
    return {candidate_key(candidate) for candidate in candidates}


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


def filter_candidate_scores(
    scores: Sequence[Mapping[str, Any]],
    candidate_outputs: Sequence[Mapping[str, Any]] | None,
) -> list[dict[str, Any]]:
    if candidate_outputs is None:
        return [dict(score) for score in scores]
    allowed = candidate_key_set(candidate_outputs)
    return [
        dict(score)
        for score in scores
        if isinstance(score.get("candidate"), Mapping) and candidate_key(score["candidate"]) in allowed
    ]


def prediction_from_row(row: Mapping[str, Any]) -> dict[str, str | None] | None:
    payload = row.get("prediction", row.get("raw_output"))
    parsed, _ = parse_component_output(payload)
    if parsed is None:
        return None
    return compact_candidate(parsed)


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


def best_candidate_for_field_value(
    scores: Sequence[Mapping[str, Any]],
    *,
    field_name: str,
    target_value: str | None,
) -> dict[str, Any] | None:
    for score_row in scores:
        candidate = score_row.get("candidate")
        if not isinstance(candidate, Mapping):
            continue
        if candidate.get(field_name) == target_value:
            return {
                "candidate": compact_candidate(candidate),
                "score": float(score_row["score"]),
            }
    return None


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
    top_value = ranked[0][0] if ranked else None
    top_score = ranked[0][1] if ranked else None
    target_rank = None
    target_score = None
    for index, (value, score) in enumerate(ranked, start=1):
        if value == target_key:
            target_rank = index
            target_score = score
            break

    margin = None
    if top_score is not None and target_score is not None:
        margin = round(top_score - target_score, 6)
    return {
        "target_value": target_key,
        "top_value": top_value,
        "target_top1": target_rank == 1,
        "target_rank": target_rank,
        "target_score": target_score,
        "top_score": top_score,
        "top_target_margin": margin,
        "value_count": len(ranked),
    }


def build_field_rank_diagnostic(
    scores: Sequence[Mapping[str, Any]],
    target: Mapping[str, Any],
) -> dict[str, Any]:
    field_ranks = {
        field_name: rank_field_values(
            scores,
            field_name=field_name,
            target_value=target.get(field_name),
        )
        for field_name in FIELD_RANK_NAMES
    }
    best_matching_candidates = {
        field_name: best_candidate_for_field_value(
            scores,
            field_name=field_name,
            target_value=target.get(field_name),
        )
        for field_name in FIELD_RANK_NAMES
    }
    return {
        "field_ranks": field_ranks,
        "best_matching_candidates": best_matching_candidates,
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
    gold_rank: int | None,
    action_rank: int | None,
    evidence_status_rank: int | None,
    near_top_rank: int = FIELD_NEAR_TOP_RANK,
) -> str:
    if exact_top1:
        return "pair_top1"
    if action_rank is None or evidence_status_rank is None or gold_rank is None:
        return "field_not_retained"
    action_near = action_rank <= near_top_rank
    status_near = evidence_status_rank <= near_top_rank
    if action_near and status_near:
        return "joint_pair_representation_failure"
    if action_near:
        return "evidence_status_field_failure"
    if status_near:
        return "action_field_failure"
    return "both_field_failure"


def prediction_lookup(prediction_rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    out: dict[str, Mapping[str, Any]] = {}
    for row in prediction_rows:
        key = row.get("source_component_target_id") or row.get("id")
        if isinstance(key, str) and key:
            out[key] = row
    return out


def build_enum_candidate_rank_report(
    *,
    expected_rows: Sequence[Mapping[str, Any]],
    prediction_rows: Sequence[Mapping[str, Any]],
    predictions_path: Path,
    run_id: str | None = None,
    max_top_candidates: int = 3,
    candidate_policy: str = "all_retained",
    candidate_outputs: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    if candidate_policy not in CANDIDATE_POLICIES:
        raise ValueError(f"unknown candidate policy: {candidate_policy}")
    expected_by_id = {str(row["id"]): row for row in expected_rows}
    predictions_by_id = prediction_lookup(prediction_rows)
    candidate_space_size = (
        len(candidate_outputs)
        if candidate_outputs is not None
        else len(enum_action_candidate_outputs())
    )
    rows: list[dict[str, Any]] = []

    for row_id, expected in expected_by_id.items():
        prediction = predictions_by_id.get(row_id)
        target = compact_candidate(target_output_from_row(expected))
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
                    "candidate_scores_present": False,
                    "retained_candidate_count": 0,
                    "candidate_space_size": candidate_space_size,
                    "all_candidates_retained": False,
                    "gold_in_retained_candidates": False,
                    "gold_rank": None,
                    "gold_score": None,
                    "top_score": None,
                    "top_gold_margin": None,
                    "top_candidates": [],
                    "field_ranks": {},
                    "best_matching_candidates": {},
                    "field_rank_pattern": "missing_prediction",
                }
            )
            continue

        all_scores = sorted_candidate_scores(prediction)
        scores = filter_candidate_scores(all_scores, candidate_outputs)
        predicted = prediction_from_row(prediction)
        top = scores[0] if scores else None
        top_candidate = top["candidate"] if top else predicted
        top_score = float(top["score"]) if top else None
        gold_rank, gold_score = find_candidate_rank(scores, target)
        margin = None
        if top_score is not None and gold_score is not None:
            margin = round(top_score - gold_score, 6)
        exact_top1 = top_candidate is not None and candidate_key(top_candidate) == candidate_key(target)
        field_diagnostic = build_field_rank_diagnostic(scores, target) if scores else {
            "field_ranks": {},
            "best_matching_candidates": {},
        }
        action_rank = field_diagnostic["field_ranks"].get("action", {}).get("target_rank")
        evidence_status_rank = field_diagnostic["field_ranks"].get("evidence_status", {}).get("target_rank")
        rank_pattern = field_rank_pattern(
            exact_top1=exact_top1,
            gold_rank=gold_rank,
            action_rank=action_rank if isinstance(action_rank, int) else None,
            evidence_status_rank=evidence_status_rank if isinstance(evidence_status_rank, int) else None,
        )
        rows.append(
            {
                "source_component_target_id": row_id,
                "case_id": component_case_id(expected),
                "case_family": expected.get("case_family"),
                "expected": target,
                "predicted": predicted,
                "top_candidate": top_candidate,
                "exact_top1": exact_top1,
                "candidate_scores_present": bool(scores),
                "retained_candidate_count": len(scores),
                "raw_retained_candidate_count": len(all_scores),
                "candidate_space_size": candidate_space_size,
                "all_candidates_retained": len(scores) == candidate_space_size,
                "gold_in_retained_candidates": gold_rank is not None,
                "gold_rank": gold_rank,
                "gold_score": gold_score,
                "top_score": top_score,
                "top_gold_margin": margin,
                "top_candidates": scores[:max_top_candidates],
                "field_ranks": field_diagnostic["field_ranks"],
                "best_matching_candidates": field_diagnostic["best_matching_candidates"],
                "field_rank_pattern": rank_pattern,
            }
        )

    rank_values = [int(row["gold_rank"]) for row in rows if isinstance(row.get("gold_rank"), int)]
    action_rank_values = [
        rank
        for row in rows
        if (rank := field_rank_value(row, "action")) is not None
    ]
    evidence_status_rank_values = [
        rank
        for row in rows
        if (rank := field_rank_value(row, "evidence_status")) is not None
    ]
    margins = [float(row["top_gold_margin"]) for row in rows if row.get("top_gold_margin") is not None]
    top_action_counts = Counter(
        row.get("top_candidate", {}).get("action")
        for row in rows
        if isinstance(row.get("top_candidate"), Mapping)
    )
    top_status_counts = Counter(
        row.get("top_candidate", {}).get("evidence_status")
        for row in rows
        if isinstance(row.get("top_candidate"), Mapping)
    )
    retained_counts = Counter(int(row["retained_candidate_count"]) for row in rows)
    field_rank_patterns = Counter(
        str(row.get("field_rank_pattern"))
        for row in rows
        if row.get("field_rank_pattern") is not None
    )
    by_case_family: dict[str, dict[str, Any]] = {}
    for case_family in sorted({str(row.get("case_family")) for row in rows}):
        family_rows = [row for row in rows if str(row.get("case_family")) == case_family]
        by_case_family[case_family] = {
            "cases": len(family_rows),
            "pair_top1": sum(1 for row in family_rows if row.get("exact_top1")),
            "action_top1": sum(1 for row in family_rows if field_rank_value(row, "action") == 1),
            "evidence_status_top1": sum(
                1 for row in family_rows if field_rank_value(row, "evidence_status") == 1
            ),
            "field_rank_patterns": dict(
                sorted(Counter(str(row.get("field_rank_pattern")) for row in family_rows).items())
            ),
        }

    first_run_id = run_id
    if first_run_id is None:
        for row in prediction_rows:
            if isinstance(row.get("run_id"), str):
                first_run_id = str(row["run_id"])
                break

    return {
        "dataset": DIAGNOSTIC_DATASET,
        "source_dataset": COMPONENT_SFT_DATASET,
        "component": "enum_action",
        "run_id": first_run_id,
        "candidate_policy": candidate_policy,
        "candidate_outputs": [compact_candidate(candidate) for candidate in candidate_outputs]
        if candidate_outputs is not None
        else None,
        "input_predictions_sha256": sha256_file(predictions_path),
        "candidate_space_size": candidate_space_size,
        "cases_expected": len(expected_rows),
        "predictions_received": len(prediction_rows),
        "summary": {
            "cases": len(rows),
            "exact_top1": sum(1 for row in rows if row.get("exact_top1")),
            "gold_in_retained_candidates": sum(1 for row in rows if row.get("gold_in_retained_candidates")),
            "all_candidates_retained_cases": sum(1 for row in rows if row.get("all_candidates_retained")),
            "missing_candidate_scores": sum(1 for row in rows if not row.get("candidate_scores_present")),
            "retained_candidate_count_histogram": dict(sorted(retained_counts.items())),
            "gold_rank_histogram": dict(sorted(Counter(rank_values).items())),
            "mean_gold_rank_observed": round(mean(rank_values), 3) if rank_values else None,
            "mean_top_gold_margin_observed": round(mean(margins), 6) if margins else None,
            "top_action_counts": dict(sorted(top_action_counts.items())),
            "top_evidence_status_counts": dict(sorted(top_status_counts.items())),
            "field_diagnostic": {
                "near_top_rank": FIELD_NEAR_TOP_RANK,
                "action_top1": sum(1 for rank in action_rank_values if rank == 1),
                "evidence_status_top1": sum(1 for rank in evidence_status_rank_values if rank == 1),
                "action_rank_histogram": dict(sorted(Counter(action_rank_values).items())),
                "evidence_status_rank_histogram": dict(sorted(Counter(evidence_status_rank_values).items())),
                "mean_action_rank_observed": round(mean(action_rank_values), 3)
                if action_rank_values
                else None,
                "mean_evidence_status_rank_observed": round(mean(evidence_status_rank_values), 3)
                if evidence_status_rank_values
                else None,
                "field_rank_patterns": dict(sorted(field_rank_patterns.items())),
                "by_case_family": by_case_family,
            },
        },
        "rows": rows,
        "scientific_readout": {
            "diagnostic_question": (
                "After schema/enum validity is fixed, where does the gold "
                "action/evidence-status pair rank among finite candidates, and "
                "is the miss driven by the action field, evidence-status field, "
                "or joint pair representation?"
            ),
            "interpretation_rule": (
                "If the gold pair ranks low and the target action/status fields also "
                "rank low, the enum_action target needs field-level supervision. If "
                "fields rank near the top but the gold pair remains low, prefer "
                "joint-pair contrastive data before tool_query, DPO, or RLVR."
            ),
            "retention_note": (
                "Reports from older runs may contain only retained top-k candidates; "
                "rerun the component after the full-score patch for exact ranks. "
                "Counterfactual candidate policies are exact only when the raw "
                "prediction row retained every candidate in that policy."
            ),
        },
    }


def write_markdown(report: Mapping[str, Any], path: Path) -> None:
    summary = report["summary"]
    field_summary = summary.get("field_diagnostic", {})
    field_pattern_json = json.dumps(field_summary.get("field_rank_patterns", {}), sort_keys=True)
    lines = [
        "# Stage A Enum-Action Candidate And Field Rank Diagnostic",
        "",
        "Purpose: compactly diagnose whether the finite enum candidate scorer ranks",
        "the gold `(action, evidence_status)` pair and its component fields near the",
        "top without publishing",
        "raw prompts, logs, model state, or full candidate-score tables.",
        "",
        "## Summary",
        "",
        f"- Run ID: `{report.get('run_id')}`",
        f"- Candidate policy: `{report.get('candidate_policy')}`",
        f"- Cases: {summary['cases']}",
        f"- Candidate space size: {report['candidate_space_size']}",
        f"- Exact top-1: {summary['exact_top1']}/{summary['cases']}",
        f"- Gold in retained candidates: {summary['gold_in_retained_candidates']}/{summary['cases']}",
        f"- All candidates retained: {summary['all_candidates_retained_cases']}/{summary['cases']}",
        f"- Mean observed gold rank: {summary['mean_gold_rank_observed']}",
        f"- Mean observed top-gold margin: {summary['mean_top_gold_margin_observed']}",
        f"- Action field top-1: {field_summary.get('action_top1')}/{summary['cases']}",
        f"- Evidence-status field top-1: {field_summary.get('evidence_status_top1')}/{summary['cases']}",
        f"- Field-rank patterns: `{field_pattern_json}`",
        "",
        "## Held-Out Rank Readout",
        "",
        (
            "| Case family | Expected | Top candidate | Pair rank | Action rank | "
            "Status rank | Margin | Pattern |"
        ),
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in report["rows"]:
        expected = row["expected"]
        top = row.get("top_candidate") or {}
        expected_text = f"`{expected.get('action')}` / `{expected.get('evidence_status')}`"
        top_text = f"`{top.get('action')}` / `{top.get('evidence_status')}`" if top else "`missing`"
        rank = row.get("gold_rank")
        action_rank = field_rank_value(row, "action")
        status_rank = field_rank_value(row, "evidence_status")
        margin = row.get("top_gold_margin")
        lines.append(
            (
                "| {case_family} | {expected} | {top} | {rank} | {action_rank} | "
                "{status_rank} | {margin} | `{pattern}` |"
            ).format(
                case_family=row.get("case_family"),
                expected=expected_text,
                top=top_text,
                rank=rank if rank is not None else "NA",
                action_rank=action_rank if action_rank is not None else "NA",
                status_rank=status_rank if status_rank is not None else "NA",
                margin=margin if margin is not None else "NA",
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
        default="post_training/stage_a_strict_component_targets_heldout_v1.jsonl",
    )
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--out-md", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--max-top-candidates", type=int, default=3)
    parser.add_argument("--candidate-policy", choices=CANDIDATE_POLICIES, default="all_retained")
    parser.add_argument(
        "--train-targets",
        default="post_training/stage_a_strict_component_targets_train_v1.jsonl",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    predictions_path = Path(args.predictions)
    expected_rows = filter_component(load_jsonl(args.expected_targets), "enum_action")
    prediction_rows = load_jsonl(predictions_path)
    candidate_outputs = None
    if args.candidate_policy == "train_observed_pairs":
        train_rows = filter_component(load_jsonl(args.train_targets), "enum_action")
        candidate_outputs = enum_action_observed_pair_outputs(train_rows)
    report = build_enum_candidate_rank_report(
        expected_rows=expected_rows,
        prediction_rows=prediction_rows,
        predictions_path=predictions_path,
        run_id=args.run_id,
        max_top_candidates=args.max_top_candidates,
        candidate_policy=args.candidate_policy,
        candidate_outputs=candidate_outputs,
    )
    if args.out_json:
        write_json(args.out_json, report)
    if args.out_md:
        write_markdown(report, Path(args.out_md))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

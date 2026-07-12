#!/usr/bin/env python3
"""Summarize row-level failures from SFT sweep run artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from negbiodb_ct import load_task_records  # noqa: E402


CONDITION_SPECS = {
    "native_cv_strict": {
        "root": "post_training/runs/qwen_sft_cv4_schema_action_80",
        "pattern": "fold*/heldout_decision_eval.json",
        "mode": "strict",
    },
    "native_cv_constrained": {
        "root": "post_training/runs/qwen_sft_cv4_schema_action_80",
        "pattern": "fold*/heldout_constrained_loaded.json",
        "mode": "constrained",
    },
    "oracle400_strict": {
        "root": "post_training/runs/qwen_oracle400_warmstart_cvheldout",
        "pattern": "fold*_heldout/decision_eval.json",
        "mode": "strict",
    },
    "oracle400_constrained": {
        "root": "post_training/runs/qwen_oracle400_warmstart_cvheldout",
        "pattern": "fold*_heldout/constrained_loaded.json",
        "mode": "constrained",
    },
}


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def load_condition_rows(
    *,
    condition: str,
    root: str | Path,
    pattern: str,
    mode: str,
) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(Path(root).glob(pattern)):
        fold = path.parent.name
        data = load_json(path)
        for row in data["rows"]:
            enriched = dict(row)
            enriched["condition"] = condition
            enriched["mode"] = mode
            enriched["fold"] = fold
            rows.append(enriched)
    return rows


def pred_action(row: Mapping[str, Any]) -> str | None:
    pred = row.get("pred")
    if not isinstance(pred, Mapping):
        return None
    action = pred.get("action")
    return str(action) if action is not None else None


def confusion_matrix(rows: list[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    matrix: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        matrix[str(row["class"])][str(pred_action(row))] += 1
    return {gold: dict(counter) for gold, counter in sorted(matrix.items())}


def class_accuracy(rows: list[Mapping[str, Any]]) -> dict[str, str]:
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for row in rows:
        bucket = counts[str(row["class"])]
        bucket[0] += int(bool(row.get("correct")))
        bucket[1] += 1
    return {label: f"{correct}/{total}" for label, (correct, total) in sorted(counts.items())}


def format_class_accuracy(values: Mapping[str, str]) -> str:
    return ", ".join(f"{label} {score}" for label, score in sorted(values.items()))


def violation_counts(rows: list[Mapping[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        for violation in row.get("generic_violations", []):
            counts[str(violation)] += 1
    return dict(sorted(counts.items()))


def gold_candidate(record: Mapping[str, Any]) -> dict[str, str]:
    action = str(record["scoring_key"]["gold_action"])
    candidate = {"action": action}
    if action in {"ground", "flag"}:
        candidate["nct"] = str(record["scoring_key"]["gold_nct"])
    return candidate


def candidate_matches(candidate: Mapping[str, Any], target: Mapping[str, str]) -> bool:
    if candidate.get("action") != target.get("action"):
        return False
    if target.get("action") in {"ground", "flag"}:
        return candidate.get("nct") == target.get("nct")
    return True


def gold_candidate_rank(
    row: Mapping[str, Any],
    record: Mapping[str, Any],
) -> dict[str, Any] | None:
    scores = row.get("candidate_scores")
    if not isinstance(scores, list):
        return None
    target = gold_candidate(record)
    sorted_scores = sorted(scores, key=lambda item: item["mean_nll"])
    winner = sorted_scores[0]
    for rank, score in enumerate(sorted_scores, start=1):
        candidate = score.get("candidate", {})
        if candidate_matches(candidate, target):
            return {
                "gold_candidate": target,
                "gold_rank": rank,
                "candidate_count": len(sorted_scores),
                "winner_candidate": winner["candidate"],
                "winner_mean_nll": winner["mean_nll"],
                "gold_mean_nll": score["mean_nll"],
                "gold_minus_winner_mean_nll": score["mean_nll"] - winner["mean_nll"],
            }
    return {
        "gold_candidate": target,
        "gold_rank": None,
        "candidate_count": len(sorted_scores),
        "winner_candidate": winner["candidate"],
        "winner_mean_nll": winner["mean_nll"],
        "gold_mean_nll": None,
        "gold_minus_winner_mean_nll": None,
    }


def failure_rows(
    rows: list[Mapping[str, Any]],
    *,
    task_index: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    failures = []
    for row in rows:
        if row.get("correct"):
            continue
        task_id = str(row["packet_id"])
        record = task_index[task_id]
        item = {
            "packet_id": task_id,
            "fold": row["fold"],
            "gold": row["class"],
            "pred": pred_action(row),
            "pred_detail": row.get("pred"),
            "violations": list(row.get("generic_violations", [])),
        }
        rank = gold_candidate_rank(row, record)
        if rank is not None:
            item["gold_candidate_rank"] = rank
        failures.append(item)
    return failures


def constrained_rank_summary(
    rows: list[Mapping[str, Any]],
    *,
    task_index: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    by_class: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"total": 0, "rank_counts": Counter(), "margins": []}
    )
    for row in rows:
        if not isinstance(row.get("candidate_scores"), list):
            continue
        record = task_index[str(row["packet_id"])]
        rank = gold_candidate_rank(row, record)
        if rank is None:
            continue
        bucket = by_class[str(row["class"])]
        bucket["total"] += 1
        rank_key = str(rank["gold_rank"]) if rank["gold_rank"] is not None else "missing"
        bucket["rank_counts"][rank_key] += 1
        margin = rank["gold_minus_winner_mean_nll"]
        if margin is not None:
            bucket["margins"].append(float(margin))

    out = {}
    for label, bucket in sorted(by_class.items()):
        margins = bucket["margins"]
        out[label] = {
            "total": bucket["total"],
            "rank_counts": dict(sorted(bucket["rank_counts"].items())),
            "mean_gold_minus_winner_mean_nll": round(sum(margins) / len(margins), 4) if margins else None,
        }
    return out


def summarize_condition(
    rows: list[Mapping[str, Any]],
    *,
    task_index: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    total = len(rows)
    correct = sum(1 for row in rows if row.get("correct"))
    parse_failures = sum(1 for row in rows if pred_action(row) in {None, "None"})
    failures = failure_rows(rows, task_index=task_index)
    return {
        "n": total,
        "correct": correct,
        "accuracy": round(correct / total, 3) if total else None,
        "parse_failures": parse_failures,
        "class_accuracy": class_accuracy(rows),
        "confusion_matrix": confusion_matrix(rows),
        "violation_counts": violation_counts(rows),
        "gold_candidate_rank_summary": constrained_rank_summary(rows, task_index=task_index),
        "failure_count": len(failures),
        "failures": failures,
    }


def recurrent_failures(condition_summaries: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for condition, summary in condition_summaries.items():
        for failure in summary["failures"]:
            grouped[failure["packet_id"]].append({
                "condition": condition,
                "fold": failure["fold"],
                "gold": failure["gold"],
                "pred": failure["pred"],
            })
    recurrent = [
        {"packet_id": packet_id, "failures": failures, "failure_conditions": len(failures)}
        for packet_id, failures in grouped.items()
        if len(failures) >= 3
    ]
    return sorted(recurrent, key=lambda item: (-item["failure_conditions"], item["packet_id"]))


def analyze(
    *,
    tasks: str | Path,
    specs: Mapping[str, Mapping[str, str]] = CONDITION_SPECS,
) -> dict[str, Any]:
    task_index = {record["packet_id"]: record for record in load_task_records(tasks)}
    condition_summaries = {}
    for condition, spec in specs.items():
        rows = load_condition_rows(
            condition=condition,
            root=spec["root"],
            pattern=spec["pattern"],
            mode=spec["mode"],
        )
        condition_summaries[condition] = summarize_condition(rows, task_index=task_index)
    return {
        "tasks": str(tasks),
        "conditions": condition_summaries,
        "recurrent_failures": recurrent_failures(condition_summaries),
    }


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    out = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        out.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(out)


def render_markdown(analysis: Mapping[str, Any]) -> str:
    condition_rows = []
    for condition, summary in analysis["conditions"].items():
        condition_rows.append([
            condition,
            summary["accuracy"],
            summary["failure_count"],
            summary["parse_failures"],
            format_class_accuracy(summary["class_accuracy"]),
        ])

    sections = [
        "# SFT Row-Level Failure Analysis: 2026-06-26",
        "",
        "Raw run artifacts are under `post_training/runs/` and ignored by git.",
        "This file records compact row-level diagnostics for the full SFT sweep.",
        "",
        "## Condition Summary",
        "",
        markdown_table(["condition", "accuracy", "failures", "parse failures", "class accuracy"], condition_rows),
        "",
        "## Confusion Matrices",
        "",
    ]
    for condition, summary in analysis["conditions"].items():
        sections.extend([
            f"### {condition}",
            "",
            "```json",
            json.dumps(summary["confusion_matrix"], indent=2, sort_keys=True),
            "```",
            "",
            "Violations:",
            "",
            "```json",
            json.dumps(summary["violation_counts"], indent=2, sort_keys=True),
            "```",
            "",
        ])
        if summary["gold_candidate_rank_summary"]:
            sections.extend([
                "Gold candidate ranks in constrained scoring:",
                "",
                "```json",
                json.dumps(summary["gold_candidate_rank_summary"], indent=2, sort_keys=True),
                "```",
                "",
            ])

    sections.extend([
        "## Recurrent Failures",
        "",
        markdown_table(
            ["packet_id", "failure conditions", "failures"],
            [
                [
                    item["packet_id"],
                    item["failure_conditions"],
                    "; ".join(
                        f"{failure['condition']}:{failure['gold']}->{failure['pred']}"
                        for failure in item["failures"]
                    ),
                ]
                for item in analysis["recurrent_failures"]
            ],
        ),
        "",
        "## Main Diagnosis",
        "",
        "- `verify` is consistently pulled toward `defer`; this is the most stable action-class failure.",
        "- `flag` is often treated as `ground`, especially after oracle-400 warm start.",
        "- Oracle-400 warm start collapses many `reject` examples to `ground`, suggesting the larger artifact over-emphasizes cited positive-looking failure rows without enough contrastive pressure for mixed-endpoint and invalid-value cases.",
        "- Parse stability is solved here; the next bottleneck is action discrimination under similar tool observations.",
        "",
        "## Next Action",
        "",
        "Build a balanced SFT variant that increases effective pressure on `verify` and `flag`, then rerun the same CV/oracle evaluation harness before DPO/RLVR.",
        "",
    ])
    return "\n".join(sections)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default="negbiodb_ct/tasks_pilot.jsonl")
    parser.add_argument("--out-json", default="post_training/sft_failure_analysis_2026-06-26.json")
    parser.add_argument("--out-md", default="post_training/SFT_FAILURE_ANALYSIS_2026-06-26.md")
    args = parser.parse_args()

    analysis = analyze(tasks=args.tasks)
    out_json = Path(args.out_json)
    out_json.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n")
    out_md = Path(args.out_md)
    out_md.write_text(render_markdown(analysis))
    print(json.dumps({
        "out_json": str(out_json),
        "out_md": str(out_md),
        "conditions": {
            name: {
                "accuracy": summary["accuracy"],
                "failure_count": summary["failure_count"],
                "parse_failures": summary["parse_failures"],
                "class_accuracy": summary["class_accuracy"],
            }
            for name, summary in analysis["conditions"].items()
        },
        "recurrent_failure_count": len(analysis["recurrent_failures"]),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

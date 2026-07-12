#!/usr/bin/env python3
"""Analyze row-level failures from the curriculum-SFT rerun."""

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
from post_training.analyze_sft_sweep_failures import (  # noqa: E402
    analyze,
    format_class_accuracy,
    markdown_table,
)


CURRICULUM_CONDITION_SPECS = {
    "curriculum_strict": {
        "root": "post_training/runs/qwen_sft_cv4_curriculum_schema_action_80",
        "pattern": "fold*/heldout_decision_eval.json",
        "mode": "strict",
    },
    "curriculum_constrained": {
        "root": "post_training/runs/qwen_sft_cv4_curriculum_schema_action_80",
        "pattern": "fold*/heldout_constrained_loaded.json",
        "mode": "constrained",
    },
}


def confusion_text(confusion: Mapping[str, Mapping[str, int]]) -> str:
    rows = []
    for gold, preds in sorted(confusion.items()):
        pred_text = ", ".join(f"{pred} {count}" for pred, count in sorted(preds.items()))
        rows.append([gold, pred_text])
    return markdown_table(["gold", "predictions"], rows)


def condition_summary_rows(analysis: Mapping[str, Any]) -> list[list[Any]]:
    rows = []
    for condition, summary in analysis["conditions"].items():
        rows.append([
            condition,
            summary["accuracy"],
            summary["failure_count"],
            summary["parse_failures"],
            format_class_accuracy(summary["class_accuracy"]),
        ])
    return rows


def task_notes(tasks: str | Path) -> dict[str, str]:
    notes = {}
    for record in load_task_records(tasks):
        scoring_key = record["scoring_key"]
        note = scoring_key.get("note") or scoring_key.get("gold_failure_category") or ""
        notes[str(record["packet_id"])] = str(note)
    return notes


def add_failure_diagnostics(analysis: dict[str, Any], tasks: str | Path) -> dict[str, Any]:
    notes = task_notes(tasks)
    pair_counts: dict[str, Counter[str]] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for condition, summary in analysis["conditions"].items():
        pair_counts[condition] = Counter()
        for failure in summary["failures"]:
            pair_key = f"{failure['gold']}->{failure['pred']}"
            pair_counts[condition][pair_key] += 1
            enriched = {
                "condition": condition,
                "fold": failure["fold"],
                "packet_id": failure["packet_id"],
                "gold": failure["gold"],
                "pred": failure["pred"],
                "note": notes.get(failure["packet_id"], ""),
            }
            rank = failure.get("gold_candidate_rank")
            if rank is not None:
                enriched["gold_rank"] = rank["gold_rank"]
                enriched["winner_candidate"] = rank["winner_candidate"]
                enriched["gold_minus_winner_mean_nll"] = rank["gold_minus_winner_mean_nll"]
            grouped[failure["packet_id"]].append(enriched)

    persistent = [
        {
            "packet_id": packet_id,
            "failure_conditions": len(failures),
            "failures": failures,
        }
        for packet_id, failures in grouped.items()
        if len(failures) >= 2
    ]
    persistent.sort(key=lambda item: (-item["failure_conditions"], item["packet_id"]))

    analysis["curriculum_failure_pair_counts"] = {
        condition: dict(sorted(counter.items()))
        for condition, counter in sorted(pair_counts.items())
    }
    analysis["curriculum_persistent_failures"] = persistent
    analysis["diagnosis"] = {
        "primary_boundary_failures": [
            "defer->verify",
            "verify->defer",
            "ground->flag",
            "reject->flag",
        ],
        "interpretation": "Curriculum improved flag but did not resolve defer/verify symmetry, clean-efficacy ground versus suspicious flag, or mixed-endpoint reject override.",
        "next": "Build row-level contrast packs or prompt targets around the persistent failures before DPO/RLVR.",
    }
    return analysis


def persistent_failure_rows(analysis: Mapping[str, Any]) -> list[list[Any]]:
    rows = []
    for item in analysis["curriculum_persistent_failures"][:30]:
        rows.append([
            item["packet_id"],
            item["failure_conditions"],
            "; ".join(
                f"{failure['condition']}:{failure['gold']}->{failure['pred']}"
                for failure in item["failures"]
            ),
            item["failures"][0].get("note", ""),
        ])
    return rows


def render_markdown(analysis: Mapping[str, Any]) -> str:
    sections = [
        "# SFT Curriculum Failure Analysis: 2026-06-26",
        "",
        "Raw run artifacts are under `post_training/runs/` and ignored by git.",
        "This file records row-level diagnostics for the full curriculum-SFT CV rerun.",
        "",
        "## Condition Summary",
        "",
        markdown_table(
            ["condition", "accuracy", "failures", "parse failures", "class accuracy"],
            condition_summary_rows(analysis),
        ),
        "",
        "## Failure Pair Counts",
        "",
        "```json",
        json.dumps(analysis["curriculum_failure_pair_counts"], indent=2, sort_keys=True),
        "```",
        "",
        "## Confusion Matrices",
        "",
    ]

    for condition, summary in analysis["conditions"].items():
        sections.extend([
            f"### {condition}",
            "",
            confusion_text(summary["confusion_matrix"]),
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
        "## Persistent Strict-And-Constrained Failures",
        "",
        markdown_table(
            ["packet_id", "failure conditions", "failures", "task note"],
            persistent_failure_rows(analysis),
        ),
        "",
        "## Diagnosis",
        "",
        "- `flag` improved relative to the original native CV run, but that came with persistent `ground -> flag` and `reject -> flag` errors.",
        "- `defer` and `verify` remain symmetric boundary failures: five true `defer` rows become `verify`, and five true `verify` rows become `defer` in both strict and constrained scoring.",
        "- Clean efficacy-failure support is still confused with impossible-value evidence: true `ground` rows mostly lose to `flag` candidates.",
        "- Mixed-endpoint `reject` override is not stable. In constrained scoring, six of eight true `reject` rows are predicted as `flag`.",
        "- Candidate-rank diagnostics show many `defer`/`verify` and `ground` misses are rank-2, while `reject` misses can be much deeper; `reject` likely needs the strongest next formulation work.",
        "",
        "## Next Action",
        "",
        "Create a curriculum-v2 or targeted prompt/SFT variant from the persistent failures: keep `flag` gains, add harder clean-ground negatives, and explicitly teach mixed-endpoint `reject` override before trying DPO/RLVR.",
        "",
    ])
    return "\n".join(sections)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default="negbiodb_ct/tasks_pilot.jsonl")
    parser.add_argument("--out-json", default="post_training/sft_curriculum_failure_analysis_2026-06-26.json")
    parser.add_argument("--out-md", default="post_training/SFT_CURRICULUM_FAILURE_ANALYSIS_2026-06-26.md")
    args = parser.parse_args()

    analysis = analyze(tasks=args.tasks, specs=CURRICULUM_CONDITION_SPECS)
    analysis = add_failure_diagnostics(analysis, args.tasks)
    Path(args.out_json).write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n")
    Path(args.out_md).write_text(render_markdown(analysis))
    print(json.dumps({
        "out_json": args.out_json,
        "out_md": args.out_md,
        "conditions": {
            name: {
                "accuracy": summary["accuracy"],
                "failure_count": summary["failure_count"],
                "parse_failures": summary["parse_failures"],
                "class_accuracy": summary["class_accuracy"],
            }
            for name, summary in analysis["conditions"].items()
        },
        "failure_pair_counts": analysis["curriculum_failure_pair_counts"],
        "persistent_failure_count": len(analysis["curriculum_persistent_failures"]),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

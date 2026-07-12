#!/usr/bin/env python3
"""Analyze row-level failures from the pressure-SFT rerun."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from post_training.analyze_sft_sweep_failures import (
    analyze,
    format_class_accuracy,
    markdown_table,
)


PRESSURE_CONDITION_SPECS = {
    "pressure_strict": {
        "root": "post_training/runs/qwen_sft_cv4_pressure_schema_action_80",
        "pattern": "fold*/heldout_decision_eval.json",
        "mode": "strict",
    },
    "pressure_constrained": {
        "root": "post_training/runs/qwen_sft_cv4_pressure_schema_action_80",
        "pattern": "fold*/heldout_constrained_loaded.json",
        "mode": "constrained",
    },
    "oracle_balanced_strict": {
        "root": "post_training/runs/qwen_oracle_balanced_warmstart_cvheldout",
        "pattern": "fold*_heldout/decision_eval.json",
        "mode": "strict",
    },
    "oracle_balanced_constrained": {
        "root": "post_training/runs/qwen_oracle_balanced_warmstart_cvheldout",
        "pattern": "fold*_heldout/constrained_loaded.json",
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


def recurrent_failure_rows(analysis: Mapping[str, Any]) -> list[list[Any]]:
    rows = []
    for item in analysis["recurrent_failures"][:20]:
        rows.append([
            item["packet_id"],
            item["failure_conditions"],
            "; ".join(
                f"{failure['condition']}:{failure['gold']}->{failure['pred']}"
                for failure in item["failures"]
            ),
        ])
    return rows


def render_markdown(analysis: Mapping[str, Any]) -> str:
    sections = [
        "# SFT Pressure Failure Analysis: 2026-06-26",
        "",
        "Raw run artifacts are under `post_training/runs/` and ignored by git.",
        "This file records row-level diagnostics for the pressure-SFT rerun.",
        "",
        "## Condition Summary",
        "",
        markdown_table(
            ["condition", "accuracy", "failures", "parse failures", "class accuracy"],
            condition_summary_rows(analysis),
        ),
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
        "## Recurrent Failures",
        "",
        markdown_table(
            ["packet_id", "failure conditions", "failures"],
            recurrent_failure_rows(analysis),
        ),
        "",
        "## Diagnosis",
        "",
        "- Native pressure fixed the previous `verify -> defer` failure, but it over-rotated: all true `defer` rows now become `verify`.",
        "- Native pressure improved `flag` under constrained scoring, but `ground` became fragile and is often treated as `flag` or `reject`.",
        "- `reject` is still not stable; pressure CV splits it across `reject`, `flag`, and `verify`.",
        "- Balanced-oracle warm start is not useful in this form. Strict generation has 12 parse failures and both strict/constrained scoring collapse toward `ground` or `reject` priors rather than evidence-sensitive actions.",
        "- The next SFT step should not be global class balancing. It should be contrastive/curriculum SFT organized around near-neighbor action confusions.",
        "",
        "## Curriculum Prescription",
        "",
        "1. Keep the native balanced CV held-out folds unchanged.",
        "2. Build train folds with small contrastive packs instead of broad class oversampling.",
        "3. Pair `ground` vs `flag` examples that share cited-NCT structure but differ by impossible-value evidence.",
        "4. Pair `verify` vs `defer` examples where the target indication lacks a valid efficacy failure, separating other-indication failures from no-failure cases.",
        "5. Pair `reject` against `ground`/`flag` examples so mixed-endpoint evidence overrides single-row positive-looking support.",
        "6. Keep `defer` and `ground` support in the curriculum; the pressure run showed that fixing `verify`/`flag` by oversampling alone can erase them.",
        "",
        "## Next Action",
        "",
        "Create a curriculum SFT artifact that tags contrast family and keeps per-fold held-out splits fixed. Then rerun the same CV harness before DPO/RLVR.",
        "",
    ])
    return "\n".join(sections)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default="negbiodb_ct/tasks_pilot.jsonl")
    parser.add_argument("--out-json", default="post_training/sft_pressure_failure_analysis_2026-06-26.json")
    parser.add_argument("--out-md", default="post_training/SFT_PRESSURE_FAILURE_ANALYSIS_2026-06-26.md")
    args = parser.parse_args()

    analysis = analyze(tasks=args.tasks, specs=PRESSURE_CONDITION_SPECS)
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
        "recurrent_failure_count": len(analysis["recurrent_failures"]),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

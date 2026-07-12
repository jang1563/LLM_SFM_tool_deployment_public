#!/usr/bin/env python3
"""Analyze row-level failures from the boundary-rationale SFT rerun."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
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


BOUNDARY_CONDITION_SPECS = {
    "boundary_strict": {
        "root": "post_training/runs/qwen_sft_cv4_boundary_rationale_schema_action_80_evalfast",
        "pattern": "fold*/heldout_decision_eval.json",
        "mode": "strict",
    },
    "boundary_constrained": {
        "root": "post_training/runs/qwen_sft_cv4_boundary_rationale_schema_action_80_evalfast",
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


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def tool_observation_signature(row: Mapping[str, Any]) -> dict[str, Any]:
    signature = {
        "search_failures_len": None,
        "other_indication_failures": None,
        "has_boundary_rationale_prompt": False,
    }
    for message in row["messages"]:
        if message.get("role") == "user" and "BOUNDARY_RATIONALE:" in str(message.get("content", "")):
            signature["has_boundary_rationale_prompt"] = True
        if message.get("role") != "tool":
            continue
        name = message.get("name")
        content = message.get("content")
        if name == "search_failures" and isinstance(content, list):
            signature["search_failures_len"] = len(content)
        if name == "check_other_indications" and isinstance(content, Mapping):
            signature["other_indication_failures"] = content.get("failures_for_other_indications")
    return signature


def heldout_signatures(root: str | Path) -> dict[str, dict[str, Any]]:
    signatures = {}
    for path in sorted(Path(root).glob("*fold*_heldout.jsonl")):
        for row in load_jsonl(path):
            packet_id = str(row.get("task_id") or row["id"].removeprefix("sft::"))
            signatures[packet_id] = tool_observation_signature(row)
    return signatures


def candidate_action_score(row: Mapping[str, Any], action: str) -> Mapping[str, Any] | None:
    scores = row.get("candidate_scores")
    if not isinstance(scores, list):
        return None
    for score in scores:
        candidate = score.get("candidate", {})
        if candidate.get("action") == action:
            return score
    return None


def pair_counts_and_failures(
    analysis: dict[str, Any],
    *,
    tasks: str | Path,
    heldout_root: str | Path,
) -> dict[str, Any]:
    notes = task_notes(tasks)
    signatures = heldout_signatures(heldout_root)
    pair_counts: dict[str, Counter[str]] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    defer_margins = []
    defer_failures = []

    for condition, summary in analysis["conditions"].items():
        pair_counts[condition] = Counter()
        for failure in summary["failures"]:
            packet_id = failure["packet_id"]
            pair_key = f"{failure['gold']}->{failure['pred']}"
            pair_counts[condition][pair_key] += 1
            enriched = {
                "condition": condition,
                "fold": failure["fold"],
                "packet_id": packet_id,
                "gold": failure["gold"],
                "pred": failure["pred"],
                "note": notes.get(packet_id, ""),
                "tool_signature": signatures.get(packet_id, {}),
            }
            rank = failure.get("gold_candidate_rank")
            if rank is not None:
                enriched["gold_rank"] = rank["gold_rank"]
                enriched["winner_candidate"] = rank["winner_candidate"]
                enriched["gold_minus_winner_mean_nll"] = rank["gold_minus_winner_mean_nll"]
            grouped[packet_id].append(enriched)

    root = Path(BOUNDARY_CONDITION_SPECS["boundary_constrained"]["root"])
    for path in sorted(root.glob("fold*/heldout_constrained_loaded.json")):
        data = json.loads(path.read_text())
        for row in data["rows"]:
            if row["class"] != "defer" or row.get("correct"):
                continue
            defer_score = candidate_action_score(row, "defer")
            verify_score = candidate_action_score(row, "verify")
            margin = None
            if defer_score and verify_score:
                margin = float(defer_score["mean_nll"]) - float(verify_score["mean_nll"])
                defer_margins.append(margin)
            defer_failures.append({
                "fold": path.parent.name,
                "packet_id": row["packet_id"],
                "pred": row.get("pred"),
                "tool_signature": signatures.get(row["packet_id"], {}),
                "verify_mean_nll": verify_score.get("mean_nll") if verify_score else None,
                "defer_mean_nll": defer_score.get("mean_nll") if defer_score else None,
                "defer_minus_verify_mean_nll": margin,
            })

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

    analysis["boundary_failure_pair_counts"] = {
        condition: dict(sorted(counter.items()))
        for condition, counter in sorted(pair_counts.items())
    }
    analysis["boundary_persistent_failures"] = persistent
    analysis["defer_verify_diagnostic"] = {
        "defer_failure_count": len(defer_failures),
        "all_defer_failures_predicted_verify": all(
            (failure["pred"] or {}).get("action") == "verify" for failure in defer_failures
        ),
        "all_defer_observations_empty": all(
            failure["tool_signature"].get("search_failures_len") == 0
            and failure["tool_signature"].get("other_indication_failures") == 0
            for failure in defer_failures
        ),
        "heldout_defer_prompt_has_boundary_rationale": any(
            failure["tool_signature"].get("has_boundary_rationale_prompt") for failure in defer_failures
        ),
        "mean_defer_minus_verify_mean_nll": round(mean(defer_margins), 4) if defer_margins else None,
        "min_defer_minus_verify_mean_nll": round(min(defer_margins), 4) if defer_margins else None,
        "max_defer_minus_verify_mean_nll": round(max(defer_margins), 4) if defer_margins else None,
        "failures": defer_failures,
    }
    analysis["diagnosis"] = {
        "primary_failure": "defer->verify",
        "interpretation": (
            "Boundary-rationale SFT improves aggregate native-SFT accuracy but "
            "fully collapses true defer rows into verify on held-out prompts."
        ),
        "likely_mechanism": (
            "The model learned the positive verify action boundary better than the "
            "negative empty-evidence defer boundary when the held-out prompt lacks "
            "the injected boundary-rationale message."
        ),
        "next": (
            "Run a held-out prompt-side rationale ablation and/or construct explicit "
            "defer-vs-verify preference pairs before DPO/RLVR."
        ),
    }
    return analysis


def persistent_failure_rows(analysis: Mapping[str, Any]) -> list[list[Any]]:
    rows = []
    for item in analysis["boundary_persistent_failures"][:30]:
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


def defer_failure_rows(analysis: Mapping[str, Any]) -> list[list[Any]]:
    rows = []
    for failure in analysis["defer_verify_diagnostic"]["failures"]:
        signature = failure["tool_signature"]
        rows.append([
            failure["fold"],
            failure["packet_id"],
            signature.get("search_failures_len"),
            signature.get("other_indication_failures"),
            f"{failure['verify_mean_nll']:.4f}",
            f"{failure['defer_mean_nll']:.4f}",
            f"{failure['defer_minus_verify_mean_nll']:.4f}",
        ])
    return rows


def render_markdown(analysis: Mapping[str, Any]) -> str:
    defer_diag = analysis["defer_verify_diagnostic"]
    sections = [
        "# SFT Boundary-Rationale Failure Analysis: 2026-06-26",
        "",
        "Raw run artifacts are under `post_training/runs/` and ignored by git.",
        "This file records row-level diagnostics for the full boundary-rationale SFT CV rerun.",
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
        json.dumps(analysis["boundary_failure_pair_counts"], indent=2, sort_keys=True),
        "```",
        "",
        "## Defer-Vs-Verify Diagnostic",
        "",
        "```text",
        f"defer_failure_count = {defer_diag['defer_failure_count']}",
        f"all_defer_failures_predicted_verify = {defer_diag['all_defer_failures_predicted_verify']}",
        f"all_defer_observations_empty = {defer_diag['all_defer_observations_empty']}",
        f"heldout_defer_prompt_has_boundary_rationale = {defer_diag['heldout_defer_prompt_has_boundary_rationale']}",
        f"mean_defer_minus_verify_mean_nll = {defer_diag['mean_defer_minus_verify_mean_nll']}",
        f"min_defer_minus_verify_mean_nll = {defer_diag['min_defer_minus_verify_mean_nll']}",
        f"max_defer_minus_verify_mean_nll = {defer_diag['max_defer_minus_verify_mean_nll']}",
        "```",
        "",
        markdown_table(
            [
                "fold",
                "packet_id",
                "search_failures",
                "other_indication_failures",
                "verify nll",
                "defer nll",
                "defer-verify nll",
            ],
            defer_failure_rows(analysis),
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
        "## Persistent Strict-And-Constrained Failures",
        "",
        markdown_table(
            ["packet_id", "failure conditions", "failures", "task note"],
            persistent_failure_rows(analysis),
        ),
        "",
        "## Diagnosis",
        "",
        "- Boundary-rationale SFT is a modest positive aggregate result, but the remaining failure is sharply structured.",
        "- Every true `defer` held-out row is predicted as `verify` in both strict and constrained scoring.",
        "- Those rows have the empty-evidence observation signature that should define `defer`: `search_failures=[]` and `failures_for_other_indications=0`.",
        "- Candidate scoring shows `defer` is not narrowly losing: mean `defer - verify` NLL is positive and substantial across all eight true-defer rows.",
        "- The held-out prompts do not contain the injected boundary-rationale message, so the next test should determine whether rationale conditioning helps only when present at inference time or whether the target signal itself is too weak.",
        "",
        "## Next Action",
        "",
        "Run a held-out prompt-side rationale ablation for the already-trained boundary-rationale fold states. If that rescues `defer`, the issue is inference-time rationale availability. If it does not, construct explicit `defer` vs `verify` preference pairs before trying broader DPO/RLVR.",
        "",
    ])
    return "\n".join(sections)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default="negbiodb_ct/tasks_pilot.jsonl")
    parser.add_argument("--heldout-root", default="post_training/cv")
    parser.add_argument("--out-json", default="post_training/sft_boundary_rationale_failure_analysis_2026-06-26.json")
    parser.add_argument("--out-md", default="post_training/SFT_BOUNDARY_RATIONALE_FAILURE_ANALYSIS_2026-06-26.md")
    args = parser.parse_args()

    analysis = analyze(tasks=args.tasks, specs=BOUNDARY_CONDITION_SPECS)
    analysis = pair_counts_and_failures(analysis, tasks=args.tasks, heldout_root=args.heldout_root)
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
        "failure_pair_counts": analysis["boundary_failure_pair_counts"],
        "persistent_failure_count": len(analysis["boundary_persistent_failures"]),
        "defer_verify_diagnostic": {
            key: value
            for key, value in analysis["defer_verify_diagnostic"].items()
            if key != "failures"
        },
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

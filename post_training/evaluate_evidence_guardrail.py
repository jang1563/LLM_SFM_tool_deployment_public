#!/usr/bin/env python3
"""Evaluate evidence-derived final-action override as a guardrail."""

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
from negbiodb_ct.baselines import score_decision  # noqa: E402
from post_training.evidence_rationale import evidence_decision  # noqa: E402
from post_training.run_sft_cv_sweep import load_json  # noqa: E402
from post_training.split_sft_data import load_jsonl  # noqa: E402
from post_training.summarize_sft_curriculum_run import markdown_table  # noqa: E402


def format_fraction(numerator: int, denominator: int) -> str:
    return f"{numerator}/{denominator}"


def accuracy_by_class(rows: list[Mapping[str, Any]], *, correct_key: str) -> dict[str, str]:
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for row in rows:
        counts[str(row["class"])][0] += int(bool(row[correct_key]))
        counts[str(row["class"])][1] += 1
    return {key: format_fraction(value[0], value[1]) for key, value in sorted(counts.items())}


def failure_pairs(rows: list[Mapping[str, Any]], *, pred_key: str, correct_key: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        if row[correct_key]:
            continue
        pred = row[pred_key] or {}
        counts[f"{row['gold']}->{pred.get('action')}"] += 1
    return dict(sorted(counts.items()))


def outcome_counts(rows: list[Mapping[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        model_correct = bool(row["model_correct"])
        guardrail_correct = bool(row["guardrail_correct"])
        if model_correct and guardrail_correct:
            counts["kept_correct"] += 1
        elif not model_correct and guardrail_correct:
            counts["rescued_error"] += 1
        elif model_correct and not guardrail_correct:
            counts["introduced_error"] += 1
        else:
            counts["still_wrong"] += 1
    return dict(sorted(counts.items()))


def normalize_model_pred(pred: Mapping[str, Any] | None) -> dict[str, str | None]:
    pred = pred or {}
    action = pred.get("action")
    cited_nct = pred.get("cited_nct") or pred.get("nct")
    return {
        "action": str(action) if action is not None else None,
        "cited_nct": str(cited_nct) if cited_nct is not None else None,
    }


def evaluate_rows(
    *,
    examples: list[Mapping[str, Any]],
    eval_rows: list[Mapping[str, Any]],
    records: Mapping[str, Mapping[str, Any]],
    source_eval_filename: str,
) -> dict[str, Any]:
    examples_by_task = {str(row["task_id"]): row for row in examples}
    eval_rows_by_task = {str(row["packet_id"]): row for row in eval_rows}
    if set(examples_by_task) != set(eval_rows_by_task):
        missing_eval = sorted(set(examples_by_task) - set(eval_rows_by_task))
        missing_example = sorted(set(eval_rows_by_task) - set(examples_by_task))
        raise ValueError(
            f"Example/eval task mismatch for {source_eval_filename}: "
            f"missing_eval={missing_eval}, missing_example={missing_example}"
        )

    rows = []
    for task_id, example in sorted(examples_by_task.items()):
        eval_row = eval_rows_by_task[task_id]
        record = records[task_id]
        guardrail_pred = evidence_decision(example)
        guardrail_score = score_decision(guardrail_pred, record["scoring_key"])
        model_pred = normalize_model_pred(eval_row.get("pred"))
        rows.append({
            "packet_id": task_id,
            "class": record["action_class"],
            "gold": record["scoring_key"]["gold_action"],
            "model_pred": model_pred,
            "guardrail_pred": guardrail_pred,
            "model_correct": bool(eval_row.get("correct")),
            "guardrail_correct": bool(guardrail_score["correct"]),
            "model_reward": float(eval_row.get("reward") or 0.0),
            "guardrail_reward": float(guardrail_score["reward"]),
            "action_changed": model_pred != guardrail_pred,
        })

    return {
        "source_eval_filename": source_eval_filename,
        "n": len(rows),
        "model_action_accuracy": round(sum(row["model_correct"] for row in rows) / len(rows), 3),
        "guardrail_action_accuracy": round(sum(row["guardrail_correct"] for row in rows) / len(rows), 3),
        "model_mean_reward": round(mean(row["model_reward"] for row in rows), 3),
        "guardrail_mean_reward": round(mean(row["guardrail_reward"] for row in rows), 3),
        "model_by_class": accuracy_by_class(rows, correct_key="model_correct"),
        "guardrail_by_class": accuracy_by_class(rows, correct_key="guardrail_correct"),
        "model_failure_pairs": failure_pairs(rows, pred_key="model_pred", correct_key="model_correct"),
        "guardrail_failure_pairs": failure_pairs(rows, pred_key="guardrail_pred", correct_key="guardrail_correct"),
        "outcome_counts": outcome_counts(rows),
        "action_changes": sum(row["action_changed"] for row in rows),
        "rows": rows,
    }


def evaluate_manifest(
    *,
    manifest: Mapping[str, Any],
    run_root: str | Path,
    tasks: str | Path,
    eval_filenames: list[str],
) -> dict[str, Any]:
    records = {str(record["packet_id"]): record for record in load_task_records(tasks)}
    run_root = Path(run_root)
    conditions = []
    for filename in eval_filenames:
        fold_summaries = []
        all_rows = []
        for fold in manifest["fold_manifests"]:
            fold_id = int(fold["fold"])
            examples = load_jsonl(fold["heldout"])
            eval_data = load_json(run_root / f"fold{fold_id}" / filename)
            fold_summary = evaluate_rows(
                examples=examples,
                eval_rows=eval_data["rows"],
                records=records,
                source_eval_filename=filename,
            )
            fold_summary["fold"] = fold_id
            fold_summaries.append(fold_summary)
            all_rows.extend(fold_summary["rows"])
        aggregate = evaluate_rows(
            examples=[
                row
                for fold in manifest["fold_manifests"]
                for row in load_jsonl(fold["heldout"])
            ],
            eval_rows=[
                row
                for fold in manifest["fold_manifests"]
                for row in load_json(run_root / f"fold{int(fold['fold'])}" / filename)["rows"]
            ],
            records=records,
            source_eval_filename=filename,
        )
        aggregate.pop("rows")
        conditions.append({
            "source_eval_filename": filename,
            "folds": [
                {key: value for key, value in fold.items() if key != "rows"}
                for fold in fold_summaries
            ],
            "aggregate": aggregate,
            "rows": all_rows,
        })
    return {
        "condition": "evidence_guardrail_override",
        "source_manifest": manifest.get("source_cv_manifest"),
        "source_strategy": manifest.get("strategy"),
        "run_root": str(run_root),
        "tasks": str(tasks),
        "eval_filenames": eval_filenames,
        "conditions": conditions,
    }


def render_markdown(summary: Mapping[str, Any]) -> str:
    sections = [
        "# Evidence-Rationale Guardrail Evaluation: 2026-06-27",
        "",
        "This file evaluates a deterministic evidence-derived final-action override on the normal boundary-rationale held-out runs.",
        "The guardrail reads only visible native CT tool observations and does not read the gold action label.",
        "",
        "## Commands",
        "",
        "```bash",
        "python3 post_training/evaluate_evidence_guardrail.py",
        "```",
        "",
        "## Aggregate",
        "",
        markdown_table(
            [
                "source eval",
                "model acc",
                "guardrail acc",
                "model defer",
                "guardrail defer",
                "rescued",
                "introduced",
                "action changes",
            ],
            [
                [
                    condition["source_eval_filename"],
                    f"{condition['aggregate']['model_action_accuracy']:.3f}",
                    f"{condition['aggregate']['guardrail_action_accuracy']:.3f}",
                    condition["aggregate"]["model_by_class"].get("defer"),
                    condition["aggregate"]["guardrail_by_class"].get("defer"),
                    condition["aggregate"]["outcome_counts"].get("rescued_error", 0),
                    condition["aggregate"]["outcome_counts"].get("introduced_error", 0),
                    condition["aggregate"]["action_changes"],
                ]
                for condition in summary["conditions"]
            ],
        ),
        "",
        "Failure pairs after guardrail:",
        "",
        "```json",
        json.dumps(
            {
                condition["source_eval_filename"]: condition["aggregate"]["guardrail_failure_pairs"]
                for condition in summary["conditions"]
            },
            indent=2,
            sort_keys=True,
        ),
        "```",
        "",
        "## Interpretation",
        "",
        "- The guardrail is deterministic and uses the same visible tool observations available to the model.",
        "- If guardrail accuracy is 1.000 with no introduced errors, the normal held-out failure is fully routable by an external evidence-boundary layer.",
        "- This supports evaluating the layer as a deployable routing/override component before spending effort on broader DPO or RLVR.",
        "",
    ]
    return "\n".join(sections)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="post_training/negbiodb_ct_native_sft_cv4_boundary_rationale_manifest.json")
    parser.add_argument("--run-root", default="post_training/runs/qwen_sft_cv4_boundary_rationale_schema_action_80_evalfast")
    parser.add_argument("--tasks", default="negbiodb_ct/tasks_pilot.jsonl")
    parser.add_argument(
        "--eval-filenames",
        nargs="+",
        default=["heldout_decision_eval.json", "heldout_constrained_loaded.json"],
    )
    parser.add_argument("--out-json", default="post_training/evidence_rationale_guardrail_eval_2026-06-27.json")
    parser.add_argument("--out-md", default="post_training/EVIDENCE_RATIONALE_GUARDRAIL_EVAL_2026-06-27.md")
    args = parser.parse_args()

    manifest = load_json(args.manifest)
    summary = evaluate_manifest(
        manifest=manifest,
        run_root=args.run_root,
        tasks=args.tasks,
        eval_filenames=args.eval_filenames,
    )
    Path(args.out_json).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    Path(args.out_md).write_text(render_markdown(summary))
    print(json.dumps({
        "out_json": args.out_json,
        "out_md": args.out_md,
        "conditions": [
            {
                "source_eval_filename": condition["source_eval_filename"],
                "aggregate": condition["aggregate"],
            }
            for condition in summary["conditions"]
        ],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

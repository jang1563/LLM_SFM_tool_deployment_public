#!/usr/bin/env python3
"""Summarize full curriculum-SFT rerun artifacts into tracked result files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from post_training.summarize_sft_pressure_runs import (
    constrained_by_class,
    format_class_accuracy,
    load_json,
    markdown_table,
    merge_class_accuracy,
    rounded_mean,
    rounded_range,
)


def summarize_curriculum(root: str | Path) -> dict[str, Any]:
    root_path = Path(root)
    summary = load_json(root_path / "summary.json")
    folds = []
    for fold in summary["folds"]:
        fold_id = int(fold["fold"])
        constrained = constrained_by_class(root_path / f"fold{fold_id}" / "heldout_constrained_loaded.json")
        folds.append({
            "fold": fold_id,
            "train_examples": fold["train_examples"],
            "heldout_examples": fold["heldout_examples"],
            "train_first_loss": round(float(fold["train_first_loss"]), 4),
            "train_last_loss": round(float(fold["train_last_loss"]), 4),
            "train_teacher_forced_loaded_loss": maybe_round(fold["train_teacher_forced_loaded_loss"]),
            "heldout_teacher_forced_loaded_loss": round(float(fold["heldout_teacher_forced_loaded_loss"]), 4),
            "strict_action_accuracy": float(fold["heldout_strict_action_accuracy"]),
            "strict_parse_failures": int(fold["heldout_strict_parse_failures"]),
            "strict_by_class": dict(fold["heldout_strict_by_class"]),
            "constrained_base_action_accuracy": maybe_float(fold["heldout_constrained_base_action_accuracy"]),
            "constrained_loaded_action_accuracy": float(fold["heldout_constrained_loaded_action_accuracy"]),
            "constrained_loaded_by_class": constrained,
        })

    strict = [fold["strict_action_accuracy"] for fold in folds]
    constrained_base = [
        fold["constrained_base_action_accuracy"]
        for fold in folds
        if fold["constrained_base_action_accuracy"] is not None
    ]
    constrained_loaded = [fold["constrained_loaded_action_accuracy"] for fold in folds]
    heldout_loss = [fold["heldout_teacher_forced_loaded_loss"] for fold in folds]
    return {
        "condition": "native_curriculum_cv",
        "root": str(root_path),
        "manifest": summary["manifest"],
        "config": summary["config"],
        "train_examples_per_fold": folds[0]["train_examples"] if folds else None,
        "folds": folds,
        "aggregate": {
            "heldout_loss_mean": round(mean(heldout_loss), 4),
            "strict_action_accuracy_mean": rounded_mean(strict),
            "strict_action_accuracy_range": rounded_range(strict),
            "strict_parse_failures_total": sum(fold["strict_parse_failures"] for fold in folds),
            "strict_class_accuracy": merge_class_accuracy([fold["strict_by_class"] for fold in folds]),
            "constrained_base_accuracy_mean": rounded_mean(constrained_base) if constrained_base else None,
            "constrained_loaded_accuracy_mean": rounded_mean(constrained_loaded),
            "constrained_loaded_accuracy_range": rounded_range(constrained_loaded),
            "constrained_loaded_class_accuracy": merge_class_accuracy(
                [fold["constrained_loaded_by_class"] for fold in folds]
            ),
        },
    }


def maybe_float(value: Any) -> float | None:
    return None if value is None else float(value)


def maybe_round(value: Any, digits: int = 4) -> float | None:
    return None if value is None else round(float(value), digits)


def format_optional_metric(value: Any, digits: int = 3) -> str:
    return "n/a" if value is None else f"{float(value):.{digits}f}"


def render_markdown(summary: Mapping[str, Any]) -> str:
    aggregate = summary["aggregate"]
    config = summary["config"]

    sections = [
        "# SFT Curriculum Run Results: 2026-06-26",
        "",
        "This file records the full curriculum CV rerun after creating the contrast-family SFT artifact.",
        "Raw run artifacts are under `post_training/runs/` and ignored by git.",
        "",
        "## Command",
        "",
        "```bash",
        "python3 post_training/run_sft_cv_sweep.py \\",
        "  --manifest post_training/negbiodb_ct_native_sft_cv4_curriculum_manifest.json \\",
        "  --out-dir post_training/runs/qwen_sft_cv4_curriculum_schema_action_80",
        "```",
        "",
        "Shared model/eval settings:",
        "",
        "```text",
        f"model = {config['model']}",
        f"batch_size = {config['batch_size']}",
        f"max_length = {config['max_length']}",
        f"max_steps = {config['max_steps']}",
        f"train_last_layers = {config['train_last_layers']}",
        f"lr = {config['lr']}",
        f"score_mode = {config['score_mode']}",
        "```",
        "",
        "## Aggregate Comparison",
        "",
        markdown_table(
            [
                "condition",
                "strict mean",
                "parse failures",
                "constrained loaded mean",
                "class takeaway",
            ],
            [
                [
                    "native CV baseline",
                    "0.475",
                    "0",
                    "0.400",
                    "ground/reject learned; verify/flag weak",
                ],
                [
                    "native pressure CV",
                    "0.400",
                    "0",
                    "0.450",
                    "verify fixed; ground/defer hurt",
                ],
                [
                    "native curriculum CV",
                    f"{aggregate['strict_action_accuracy_mean']:.3f}",
                    aggregate["strict_parse_failures_total"],
                    f"{aggregate['constrained_loaded_accuracy_mean']:.3f}",
                    "flag improved; balance still mixed",
                ],
                [
                    "oracle balanced warm-start",
                    "0.200",
                    "12",
                    "0.200",
                    "negative result; ground collapse",
                ],
            ],
        ),
        "",
        "## Curriculum CV Folds",
        "",
        markdown_table(
            [
                "fold",
                "heldout loss",
                "strict acc",
                "constrained base",
                "constrained loaded",
                "strict by class",
            ],
            [
                [
                    fold["fold"],
                    f"{fold['heldout_teacher_forced_loaded_loss']:.4f}",
                    f"{fold['strict_action_accuracy']:.3f}",
                    format_optional_metric(fold["constrained_base_action_accuracy"]),
                    f"{fold['constrained_loaded_action_accuracy']:.3f}",
                    format_class_accuracy(fold["strict_by_class"]),
                ]
                for fold in summary["folds"]
            ],
        ),
        "",
        "Aggregate:",
        "",
        "```text",
        f"heldout_loss_mean = {aggregate['heldout_loss_mean']:.4f}",
        f"strict_action_accuracy_mean = {aggregate['strict_action_accuracy_mean']:.3f}",
        f"strict_action_accuracy_range = {aggregate['strict_action_accuracy_range']}",
        f"strict_parse_failures_total = {aggregate['strict_parse_failures_total']}",
        f"strict_class_accuracy = {format_class_accuracy(aggregate['strict_class_accuracy'])}",
        f"constrained_base_accuracy_mean = {format_optional_metric(aggregate['constrained_base_accuracy_mean'])}",
        f"constrained_loaded_accuracy_mean = {aggregate['constrained_loaded_accuracy_mean']:.3f}",
        f"constrained_loaded_accuracy_range = {aggregate['constrained_loaded_accuracy_range']}",
        f"constrained_loaded_class_accuracy = {format_class_accuracy(aggregate['constrained_loaded_class_accuracy'])}",
        "```",
        "",
        "## Interpretation",
        "",
        "- Curriculum SFT restores the native baseline strict mean of 0.475 while improving strict `flag` from 3/8 in the original native CV run to 6/8.",
        "- It avoids the pressure run's all-`verify` over-rotation and the balanced-oracle `ground` collapse, but it does not improve the overall held-out strict mean beyond the original native CV baseline.",
        "- Constrained loaded accuracy is 0.425: above the original native baseline of 0.400, below the pressure run's 0.450, and far above the balanced-oracle negative result of 0.200.",
        "- This is still an SFT-formulation bottleneck. DPO/RLVR should wait until row-level curriculum failures show a cleaner action boundary.",
        "",
        "## Next Action",
        "",
        "Row-level curriculum-failure analysis is now recorded in",
        "`post_training/SFT_CURRICULUM_FAILURE_ANALYSIS_2026-06-26.md`.",
        "",
        "Follow-up complete: targeted curriculum-v2 is recorded in",
        "`post_training/SFT_CURRICULUM_V2_RUN_RESULTS_2026-06-26.md`; it is a negative",
        "oversampling result.",
        "",
        "The current next implementation step is boundary-rationale or paired contrast",
        "SFT, not another row-duplication pass and not DPO/RLVR yet.",
        "",
    ]
    return "\n".join(sections)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="post_training/runs/qwen_sft_cv4_curriculum_schema_action_80")
    parser.add_argument("--out-json", default="post_training/sft_curriculum_run_summary_2026-06-26.json")
    parser.add_argument("--out-md", default="post_training/SFT_CURRICULUM_RUN_RESULTS_2026-06-26.md")
    args = parser.parse_args()

    summary = summarize_curriculum(args.root)
    Path(args.out_json).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    Path(args.out_md).write_text(render_markdown(summary))
    print(json.dumps({
        "out_json": args.out_json,
        "out_md": args.out_md,
        "native_curriculum": summary["aggregate"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

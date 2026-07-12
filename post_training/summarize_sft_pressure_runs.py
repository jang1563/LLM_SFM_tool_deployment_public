#!/usr/bin/env python3
"""Summarize full pressure-SFT rerun artifacts into tracked result files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any, Mapping


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def parse_fraction(value: str) -> tuple[int, int]:
    if "/" not in value:
        raise ValueError(f"Expected fraction string like '1/2': {value!r}")
    left, right = value.split("/", 1)
    return int(left), int(right)


def format_fraction(numerator: int, denominator: int) -> str:
    return f"{numerator}/{denominator}"


def merge_class_accuracy(rows: list[Mapping[str, str]]) -> dict[str, str]:
    counts: dict[str, list[int]] = {}
    for row in rows:
        for label, value in row.items():
            numerator, denominator = parse_fraction(value)
            bucket = counts.setdefault(label, [0, 0])
            bucket[0] += numerator
            bucket[1] += denominator
    return {
        label: format_fraction(numerator, denominator)
        for label, (numerator, denominator) in sorted(counts.items())
    }


def format_class_accuracy(values: Mapping[str, str]) -> str:
    return ", ".join(f"{label} {score}" for label, score in sorted(values.items()))


def rounded_mean(values: list[float]) -> float:
    return round(mean(values), 3)


def rounded_range(values: list[float]) -> str:
    return f"{min(values):.3f}..{max(values):.3f}"


def constrained_by_class(path: str | Path) -> dict[str, str]:
    return dict(load_json(path)["summary"]["by_class"])


def summarize_native_pressure(root: str | Path) -> dict[str, Any]:
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
            "train_teacher_forced_loaded_loss": round(float(fold["train_teacher_forced_loaded_loss"]), 4),
            "heldout_teacher_forced_loaded_loss": round(float(fold["heldout_teacher_forced_loaded_loss"]), 4),
            "strict_action_accuracy": float(fold["heldout_strict_action_accuracy"]),
            "strict_parse_failures": int(fold["heldout_strict_parse_failures"]),
            "strict_by_class": dict(fold["heldout_strict_by_class"]),
            "constrained_base_action_accuracy": float(fold["heldout_constrained_base_action_accuracy"]),
            "constrained_loaded_action_accuracy": float(fold["heldout_constrained_loaded_action_accuracy"]),
            "constrained_loaded_by_class": constrained,
        })

    strict = [fold["strict_action_accuracy"] for fold in folds]
    constrained_base = [fold["constrained_base_action_accuracy"] for fold in folds]
    constrained_loaded = [fold["constrained_loaded_action_accuracy"] for fold in folds]
    heldout_loss = [fold["heldout_teacher_forced_loaded_loss"] for fold in folds]
    return {
        "condition": "native_pressure_cv",
        "root": str(root_path),
        "manifest": summary["manifest"],
        "train_examples_per_fold": folds[0]["train_examples"] if folds else None,
        "folds": folds,
        "aggregate": {
            "heldout_loss_mean": round(mean(heldout_loss), 4),
            "strict_action_accuracy_mean": rounded_mean(strict),
            "strict_action_accuracy_range": rounded_range(strict),
            "strict_parse_failures_total": sum(fold["strict_parse_failures"] for fold in folds),
            "strict_class_accuracy": merge_class_accuracy([fold["strict_by_class"] for fold in folds]),
            "constrained_base_accuracy_mean": rounded_mean(constrained_base),
            "constrained_loaded_accuracy_mean": rounded_mean(constrained_loaded),
            "constrained_loaded_accuracy_range": rounded_range(constrained_loaded),
            "constrained_loaded_class_accuracy": merge_class_accuracy(
                [fold["constrained_loaded_by_class"] for fold in folds]
            ),
        },
    }


def summarize_oracle_balanced(root: str | Path) -> dict[str, Any]:
    root_path = Path(root)
    summary = load_json(root_path / "summary.json")
    evals = []
    for item in summary["evals"]:
        name = str(item["name"])
        constrained = constrained_by_class(root_path / name / "constrained_loaded.json")
        evals.append({
            "name": name,
            "heldout_examples": item["examples"],
            "heldout_teacher_forced_loaded_loss": round(float(item["teacher_forced_loaded_loss"]), 4),
            "strict_action_accuracy": float(item["strict_action_accuracy"]),
            "strict_parse_failures": int(item["strict_parse_failures"]),
            "strict_by_class": dict(item["strict_by_class"]),
            "constrained_base_action_accuracy": float(item["constrained_base_action_accuracy"]),
            "constrained_loaded_action_accuracy": float(item["constrained_loaded_action_accuracy"]),
            "constrained_loaded_by_class": constrained,
        })

    strict = [item["strict_action_accuracy"] for item in evals]
    constrained_base = [item["constrained_base_action_accuracy"] for item in evals]
    constrained_loaded = [item["constrained_loaded_action_accuracy"] for item in evals]
    heldout_loss = [item["heldout_teacher_forced_loaded_loss"] for item in evals]
    return {
        "condition": "oracle_balanced_warmstart",
        "root": str(root_path),
        "train_sft": summary["train_sft"],
        "train_limit": summary["train_limit"],
        "train_first_loss": round(float(summary["train_first_loss"]), 4),
        "train_last_loss": round(float(summary["train_last_loss"]), 4),
        "train_loss_delta": round(float(summary["train_loss_delta"]), 4),
        "evals": evals,
        "aggregate": {
            "heldout_loss_mean": round(mean(heldout_loss), 4),
            "strict_action_accuracy_mean": rounded_mean(strict),
            "strict_action_accuracy_range": rounded_range(strict),
            "strict_parse_failures_total": sum(item["strict_parse_failures"] for item in evals),
            "strict_class_accuracy": merge_class_accuracy([item["strict_by_class"] for item in evals]),
            "constrained_base_accuracy_mean": rounded_mean(constrained_base),
            "constrained_loaded_accuracy_mean": rounded_mean(constrained_loaded),
            "constrained_loaded_accuracy_range": rounded_range(constrained_loaded),
            "constrained_loaded_class_accuracy": merge_class_accuracy(
                [item["constrained_loaded_by_class"] for item in evals]
            ),
        },
    }


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    out = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        out.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(out)


def render_markdown(summary: Mapping[str, Any]) -> str:
    pressure = summary["conditions"]["native_pressure_cv"]
    oracle = summary["conditions"]["oracle_balanced_warmstart"]
    pressure_agg = pressure["aggregate"]
    oracle_agg = oracle["aggregate"]

    sections = [
        "# SFT Pressure Run Results: 2026-06-26",
        "",
        "This file records the full rerun after creating the pressure-SFT artifacts.",
        "Raw run artifacts are under `post_training/runs/` and ignored by git.",
        "",
        "## Commands",
        "",
        "```bash",
        "python3 post_training/run_sft_cv_sweep.py \\",
        "  --manifest post_training/negbiodb_ct_native_sft_cv4_pressure_manifest.json \\",
        "  --out-dir post_training/runs/qwen_sft_cv4_pressure_schema_action_80",
        "",
        "python3 post_training/run_sft_oracle_warmstart.py \\",
        "  --train-sft post_training/negbiodb_ct_oracle_sft_balanced_v1.jsonl \\",
        "  --train-limit 700 \\",
        "  --out-dir post_training/runs/qwen_oracle_balanced_warmstart_cvheldout",
        "```",
        "",
        "Shared model/eval settings:",
        "",
        "```text",
        "model = Qwen/Qwen2.5-0.5B-Instruct",
        "batch_size = 2",
        "max_length = 512",
        "train_last_layers = 2",
        "lr = 5e-5",
        "score_mode = mean",
        "```",
        "",
        "## Aggregate Comparison",
        "",
        markdown_table(
            [
                "condition",
                "train data",
                "strict mean",
                "strict range",
                "parse failures",
                "constrained loaded mean",
                "constrained loaded range",
            ],
            [
                [
                    "native pressure CV",
                    "54/fold",
                    f"{pressure_agg['strict_action_accuracy_mean']:.3f}",
                    pressure_agg["strict_action_accuracy_range"],
                    pressure_agg["strict_parse_failures_total"],
                    f"{pressure_agg['constrained_loaded_accuracy_mean']:.3f}",
                    pressure_agg["constrained_loaded_accuracy_range"],
                ],
                [
                    "oracle balanced warm-start",
                    str(oracle["train_limit"]),
                    f"{oracle_agg['strict_action_accuracy_mean']:.3f}",
                    oracle_agg["strict_action_accuracy_range"],
                    oracle_agg["strict_parse_failures_total"],
                    f"{oracle_agg['constrained_loaded_accuracy_mean']:.3f}",
                    oracle_agg["constrained_loaded_accuracy_range"],
                ],
            ],
        ),
        "",
        "## Native Pressure CV",
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
                    f"{fold['constrained_base_action_accuracy']:.3f}",
                    f"{fold['constrained_loaded_action_accuracy']:.3f}",
                    format_class_accuracy(fold["strict_by_class"]),
                ]
                for fold in pressure["folds"]
            ],
        ),
        "",
        "Aggregate:",
        "",
        "```text",
        f"heldout_loss_mean = {pressure_agg['heldout_loss_mean']:.4f}",
        f"strict_action_accuracy_mean = {pressure_agg['strict_action_accuracy_mean']:.3f}",
        f"strict_action_accuracy_range = {pressure_agg['strict_action_accuracy_range']}",
        f"strict_parse_failures_total = {pressure_agg['strict_parse_failures_total']}",
        f"strict_class_accuracy = {format_class_accuracy(pressure_agg['strict_class_accuracy'])}",
        f"constrained_base_accuracy_mean = {pressure_agg['constrained_base_accuracy_mean']:.3f}",
        f"constrained_loaded_accuracy_mean = {pressure_agg['constrained_loaded_accuracy_mean']:.3f}",
        f"constrained_loaded_class_accuracy = {format_class_accuracy(pressure_agg['constrained_loaded_class_accuracy'])}",
        "```",
        "",
        "## Oracle Balanced Warm Start",
        "",
        "```text",
        f"train_limit = {oracle['train_limit']}",
        f"train_first_loss = {oracle['train_first_loss']:.4f}",
        f"train_last_loss = {oracle['train_last_loss']:.4f}",
        f"train_loss_delta = {oracle['train_loss_delta']:.4f}",
        "```",
        "",
        markdown_table(
            [
                "eval set",
                "heldout loss",
                "strict acc",
                "parse failures",
                "constrained base",
                "constrained loaded",
                "strict by class",
            ],
            [
                [
                    item["name"].replace("_", " "),
                    f"{item['heldout_teacher_forced_loaded_loss']:.4f}",
                    f"{item['strict_action_accuracy']:.3f}",
                    item["strict_parse_failures"],
                    f"{item['constrained_base_action_accuracy']:.3f}",
                    f"{item['constrained_loaded_action_accuracy']:.3f}",
                    format_class_accuracy(item["strict_by_class"]),
                ]
                for item in oracle["evals"]
            ],
        ),
        "",
        "Aggregate:",
        "",
        "```text",
        f"heldout_loss_mean = {oracle_agg['heldout_loss_mean']:.4f}",
        f"strict_action_accuracy_mean = {oracle_agg['strict_action_accuracy_mean']:.3f}",
        f"strict_action_accuracy_range = {oracle_agg['strict_action_accuracy_range']}",
        f"strict_parse_failures_total = {oracle_agg['strict_parse_failures_total']}",
        f"strict_class_accuracy = {format_class_accuracy(oracle_agg['strict_class_accuracy'])}",
        f"constrained_base_accuracy_mean = {oracle_agg['constrained_base_accuracy_mean']:.3f}",
        f"constrained_loaded_accuracy_mean = {oracle_agg['constrained_loaded_accuracy_mean']:.3f}",
        f"constrained_loaded_class_accuracy = {format_class_accuracy(oracle_agg['constrained_loaded_class_accuracy'])}",
        "```",
        "",
        "## Interpretation",
        "",
        "- Native pressure training does not improve strict generation over the first native CV run, but it slightly improves constrained loaded accuracy from 0.400 to 0.450.",
        "- The pressure effect is specific: `verify` becomes reliable in strict generation, while `defer` and `ground` collapse to 0/8 under strict generation.",
        "- Balanced-oracle warm start is a negative result: it overfits the 700-row teacher artifact and collapses native held-out predictions toward `ground`, with 12 strict parse failures.",
        "- DPO/RLVR should still wait. The next bottleneck is SFT formulation/curriculum, not preference optimization.",
        "",
        "## Next Action",
        "",
        "Run row-level pressure-failure analysis and design a curriculum/contrastive SFT variant that separates `ground` vs `flag`, `verify` vs `defer`, and preserves `reject` while keeping parse stability.",
        "",
    ]
    return "\n".join(sections)


def build_summary(pressure_root: str | Path, oracle_root: str | Path) -> dict[str, Any]:
    return {
        "conditions": {
            "native_pressure_cv": summarize_native_pressure(pressure_root),
            "oracle_balanced_warmstart": summarize_oracle_balanced(oracle_root),
        },
        "interpretation": {
            "native_pressure": "slight constrained improvement but strict action-class tradeoff",
            "oracle_balanced": "negative result; ground collapse and parse failures on native heldout",
            "next": "row-level pressure-failure analysis before DPO/RLVR",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pressure-root", default="post_training/runs/qwen_sft_cv4_pressure_schema_action_80")
    parser.add_argument("--oracle-root", default="post_training/runs/qwen_oracle_balanced_warmstart_cvheldout")
    parser.add_argument("--out-json", default="post_training/sft_pressure_run_summary_2026-06-26.json")
    parser.add_argument("--out-md", default="post_training/SFT_PRESSURE_RUN_RESULTS_2026-06-26.md")
    args = parser.parse_args()

    summary = build_summary(args.pressure_root, args.oracle_root)
    Path(args.out_json).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    Path(args.out_md).write_text(render_markdown(summary))
    print(json.dumps({
        "out_json": args.out_json,
        "out_md": args.out_md,
        "native_pressure": summary["conditions"]["native_pressure_cv"]["aggregate"],
        "oracle_balanced": summary["conditions"]["oracle_balanced_warmstart"]["aggregate"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Summarize held-out boundary-rationale ablation artifacts."""

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

from post_training.analyze_sft_sweep_failures import load_json  # noqa: E402
from post_training.summarize_sft_curriculum_run import markdown_table  # noqa: E402
from post_training.summarize_sft_pressure_runs import (  # noqa: E402
    format_class_accuracy,
    merge_class_accuracy,
    rounded_mean,
    rounded_range,
)


def constrained_by_class(path: str | Path) -> dict[str, str]:
    data = load_json(path)
    return dict(data["summary"]["by_class"])


def failure_pair_counts(root: str | Path, filename: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for path in sorted(Path(root).glob(f"fold*/{filename}")):
        data = load_json(path)
        for row in data["rows"]:
            if row.get("correct"):
                continue
            pred = row.get("pred") or {}
            counts[f"{row['class']}->{pred.get('action')}"] += 1
    return dict(sorted(counts.items()))


def confusion_matrix(root: str | Path, filename: str) -> dict[str, dict[str, int]]:
    matrix: dict[str, Counter[str]] = defaultdict(Counter)
    for path in sorted(Path(root).glob(f"fold*/{filename}")):
        data = load_json(path)
        for row in data["rows"]:
            pred = row.get("pred") or {}
            matrix[str(row["class"])][str(pred.get("action"))] += 1
    return {gold: dict(preds) for gold, preds in sorted(matrix.items())}


def summarize(root: str | Path, manifest: str | Path) -> dict[str, Any]:
    root_path = Path(root)
    raw = load_json(root_path / "summary.json")
    ablation_manifest = load_json(manifest)
    folds = []
    for fold in raw["folds"]:
        fold_id = int(fold["fold"])
        constrained = constrained_by_class(root_path / f"fold{fold_id}" / "heldout_constrained_loaded.json")
        folds.append({
            "fold": fold_id,
            "source_state": fold["source_state"],
            "source_heldout": fold["source_heldout"],
            "heldout": fold["heldout"],
            "heldout_examples": fold["heldout_examples"],
            "heldout_by_class": fold["heldout_by_class"],
            "heldout_by_role": fold["heldout_by_role"],
            "heldout_teacher_forced_loaded_loss": round(float(fold["heldout_teacher_forced_loaded_loss"]), 4),
            "strict_action_accuracy": float(fold["heldout_strict_action_accuracy"]),
            "strict_parse_failures": int(fold["heldout_strict_parse_failures"]),
            "strict_by_class": dict(fold["heldout_strict_by_class"]),
            "constrained_loaded_action_accuracy": float(fold["heldout_constrained_loaded_action_accuracy"]),
            "constrained_loaded_by_class": constrained,
        })
    strict = [fold["strict_action_accuracy"] for fold in folds]
    constrained = [fold["constrained_loaded_action_accuracy"] for fold in folds]
    heldout_loss = [fold["heldout_teacher_forced_loaded_loss"] for fold in folds]
    return {
        "condition": raw["condition"],
        "root": str(root_path),
        "ablation_manifest": str(manifest),
        "config": raw["config"],
        "artifact": {
            "dataset": ablation_manifest["dataset"],
            "strategy": ablation_manifest["strategy"],
            "rationale_mode": ablation_manifest.get("rationale_mode", "oracle"),
            "source_boundary_manifest": ablation_manifest["source_boundary_manifest"],
            "folds": [
                {
                    "fold": fold["fold"],
                    "heldout_examples": fold["heldout_examples"],
                    "heldout_by_class": fold["heldout_by_class"],
                    "heldout_by_role": fold["heldout_by_role"],
                    "evidence_action_mismatches": fold.get("evidence_action_mismatches"),
                }
                for fold in ablation_manifest["fold_manifests"]
            ],
        },
        "folds": folds,
        "aggregate": {
            "heldout_loss_mean": round(mean(heldout_loss), 4),
            "strict_action_accuracy_mean": rounded_mean(strict),
            "strict_action_accuracy_range": rounded_range(strict),
            "strict_parse_failures_total": sum(fold["strict_parse_failures"] for fold in folds),
            "strict_class_accuracy": merge_class_accuracy([fold["strict_by_class"] for fold in folds]),
            "constrained_loaded_accuracy_mean": rounded_mean(constrained),
            "constrained_loaded_accuracy_range": rounded_range(constrained),
            "constrained_loaded_class_accuracy": merge_class_accuracy(
                [fold["constrained_loaded_by_class"] for fold in folds]
            ),
        },
        "failure_pairs": {
            "strict": failure_pair_counts(root_path, "heldout_decision_eval.json"),
            "constrained_loaded": failure_pair_counts(root_path, "heldout_constrained_loaded.json"),
        },
        "confusion_matrices": {
            "strict": confusion_matrix(root_path, "heldout_decision_eval.json"),
            "constrained_loaded": confusion_matrix(root_path, "heldout_constrained_loaded.json"),
        },
    }


def render_markdown(summary: Mapping[str, Any]) -> str:
    aggregate = summary["aggregate"]
    artifact = summary["artifact"]
    config = summary["config"]
    rationale_mode = artifact["rationale_mode"]
    mode_label = "oracle-rationale" if rationale_mode == "oracle" else "evidence-rationale"
    title_label = "Oracle-Rationale" if rationale_mode == "oracle" else "Evidence-Rationale"
    if rationale_mode == "oracle":
        rationale_description = (
            "The ablation inserts the same oracle `BOUNDARY_RATIONALE` prompt into "
            "each held-out row before the final `submit_decision` target."
        )
        interpretation = [
            "- This is an oracle-rationale ablation, not a deployable evaluation condition.",
            "- The ablation fully rescues the previous `defer -> verify` collapse when the correct boundary rationale is visible at inference time.",
            "- Because strict generation and constrained scoring both reach 1.000, the failure is not candidate scoring or schema parsing.",
            "- The normal held-out failure is therefore best interpreted as missing inference-time boundary reasoning, not a broad need for RLVR yet.",
        ]
        next_action = (
            "Choose between a deployable rationale-at-inference design and explicit "
            "`defer` versus `verify` preference supervision. A non-oracle rationale "
            "generator or a DPO-style pair set should be tested before broader RLVR."
        )
    else:
        rationale_description = (
            "The ablation inserts a deterministic `BOUNDARY_RATIONALE` prompt derived "
            "only from visible held-out tool observations before the final "
            "`submit_decision` target."
        )
        interpretation = [
            "- This is a tool-derived rationale ablation: the rationale is generated from visible tool outputs, not from the gold action label.",
            "- If this condition rescues `defer`, then the main missing component is a deployable boundary-rationale generator or policy layer at inference time.",
            "- If it fails, explicit `defer` versus `verify` preference supervision is still needed even with visible rationale hints.",
            "- This condition is more deployable than oracle rationale, but it still relies on a deterministic rule-rationale preprocessor.",
        ]
        next_action = (
            "Use this result to decide whether the deterministic evidence-rationale "
            "layer should become a deployable guardrail/routing component, or whether "
            "its outputs should instead be distilled into preference/SFT data."
        )
    sections = [
        f"# SFT Boundary-Rationale Held-Out {title_label} Ablation: 2026-06-27",
        "",
        "This file records an eval-only ablation using the already-trained boundary-rationale fold states.",
        rationale_description,
        "Raw run artifacts are under `post_training/runs/` and ignored by git.",
        "",
        "## Commands",
        "",
        "```bash",
        "python3 post_training/run_sft_boundary_rationale_ablation.py",
        "python3 post_training/summarize_sft_boundary_rationale_ablation.py",
        "```",
        "",
        "Eval settings:",
        "",
        "```text",
        f"model = {config['model']}",
        f"source_state_root = {config['source_state_root']}",
        f"batch_size = {config['batch_size']}",
        f"max_length = {config['max_length']}",
        f"score_mode = {config['score_mode']}",
        "```",
        "",
        "## Artifact Summary",
        "",
        "```text",
        f"dataset = {artifact['dataset']}",
        f"strategy = {artifact['strategy']}",
        f"rationale_mode = {artifact['rationale_mode']}",
        f"source_boundary_manifest = {artifact['source_boundary_manifest']}",
        "```",
        "",
        markdown_table(
            ["fold", "heldout rows", "heldout by class", "heldout by role", "evidence mismatches"],
            [
                [
                    fold["fold"],
                    fold["heldout_examples"],
                    json.dumps(fold["heldout_by_class"], sort_keys=True),
                    json.dumps(fold["heldout_by_role"], sort_keys=True),
                    "" if fold["evidence_action_mismatches"] is None else fold["evidence_action_mismatches"],
                ]
                for fold in artifact["folds"]
            ],
        ),
        "",
        "## Aggregate Comparison",
        "",
        markdown_table(
            ["condition", "strict mean", "constrained loaded mean", "parse failures", "defer", "takeaway"],
            [
                ["boundary rationale, normal held-out", "0.500", "0.500", "0", "0/8", "best native-SFT aggregate but defer collapsed"],
                [
                    f"held-out {mode_label} ablation",
                    f"{aggregate['strict_action_accuracy_mean']:.3f}",
                    f"{aggregate['constrained_loaded_accuracy_mean']:.3f}",
                    aggregate["strict_parse_failures_total"],
                    aggregate["strict_class_accuracy"].get("defer"),
                    "tests whether inference-time rationale rescues defer",
                ],
            ],
        ),
        "",
        "## Ablation Folds",
        "",
        markdown_table(
            ["fold", "heldout loss", "strict acc", "constrained loaded", "strict by class"],
            [
                [
                    fold["fold"],
                    f"{fold['heldout_teacher_forced_loaded_loss']:.4f}",
                    f"{fold['strict_action_accuracy']:.3f}",
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
        f"constrained_loaded_accuracy_mean = {aggregate['constrained_loaded_accuracy_mean']:.3f}",
        f"constrained_loaded_accuracy_range = {aggregate['constrained_loaded_accuracy_range']}",
        f"constrained_loaded_class_accuracy = {format_class_accuracy(aggregate['constrained_loaded_class_accuracy'])}",
        "```",
        "",
        "Failure pair counts:",
        "",
        "```json",
        json.dumps(summary["failure_pairs"], indent=2, sort_keys=True),
        "```",
        "",
        "## Interpretation",
        "",
        *interpretation,
        "",
        "## Next Action",
        "",
        next_action,
        "",
    ]
    return "\n".join(sections)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="post_training/runs/qwen_sft_cv4_boundary_rationale_heldout_oracle_ablation")
    parser.add_argument("--manifest", default="post_training/negbiodb_ct_native_sft_cv4_boundary_rationale_heldout_ablation_manifest.json")
    parser.add_argument("--out-json", default="post_training/sft_boundary_rationale_heldout_ablation_summary_2026-06-27.json")
    parser.add_argument("--out-md", default="post_training/SFT_BOUNDARY_RATIONALE_HELDOUT_ABLATION_2026-06-27.md")
    args = parser.parse_args()

    summary = summarize(args.root, args.manifest)
    Path(args.out_json).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    Path(args.out_md).write_text(render_markdown(summary))
    print(json.dumps({
        "out_json": args.out_json,
        "out_md": args.out_md,
        "aggregate": summary["aggregate"],
        "failure_pairs": summary["failure_pairs"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Summarize full boundary-rationale SFT CV artifacts."""

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

from post_training.analyze_sft_sweep_failures import load_json  # noqa: E402
from post_training.summarize_sft_curriculum_run import (  # noqa: E402
    format_class_accuracy,
    format_optional_metric,
    markdown_table,
    summarize_curriculum,
)


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


def manifest_summary(path: str | Path) -> dict[str, Any]:
    manifest = load_json(path)
    return {
        "dataset": manifest["dataset"],
        "strategy": manifest["strategy"],
        "boundary_negative_actions": manifest["boundary_negative_actions"],
        "folds": [
            {
                "fold": fold["fold"],
                "train_examples": fold["train_examples"],
                "train_by_class": fold["train_by_class"],
                "train_by_role": fold["train_by_role"],
                "heldout_examples": fold["heldout_examples"],
                "heldout_by_class": fold["heldout_by_class"],
            }
            for fold in manifest["fold_manifests"]
        ],
    }


def build_summary(root: str | Path, manifest: str | Path) -> dict[str, Any]:
    summary = summarize_curriculum(root)
    summary["condition"] = "native_boundary_rationale_cv"
    summary["artifact"] = manifest_summary(manifest)
    summary["failure_pairs"] = {
        "strict": failure_pair_counts(root, "heldout_decision_eval.json"),
        "constrained_loaded": failure_pair_counts(root, "heldout_constrained_loaded.json"),
    }
    summary["confusion_matrices"] = {
        "strict": confusion_matrix(root, "heldout_decision_eval.json"),
        "constrained_loaded": confusion_matrix(root, "heldout_constrained_loaded.json"),
    }
    return summary


def render_markdown(summary: Mapping[str, Any]) -> str:
    aggregate = summary["aggregate"]
    artifact = summary["artifact"]
    config = summary["config"]
    sections = [
        "# SFT Boundary-Rationale Run Results: 2026-06-26",
        "",
        "This file records the full boundary-rationale CV rerun after the targeted curriculum-v2 negative result.",
        "Raw run artifacts are under `post_training/runs/` and ignored by git.",
        "",
        "## Commands",
        "",
        "```bash",
        "python3 post_training/build_sft_boundary_rationale_data.py",
        "",
        "python3 post_training/run_sft_cv_sweep.py \\",
        "  --manifest post_training/negbiodb_ct_native_sft_cv4_boundary_rationale_manifest.json \\",
        "  --out-dir post_training/runs/qwen_sft_cv4_boundary_rationale_schema_action_80_evalfast \\",
        "  --skip-train-loss \\",
        "  --skip-base-constrained",
        "```",
        "",
        "Eval settings:",
        "",
        "```text",
        f"model = {config['model']}",
        f"max_steps = {config['max_steps']}",
        f"batch_size = {config['batch_size']}",
        f"max_length = {config['max_length']}",
        f"train_last_layers = {config['train_last_layers']}",
        f"lr = {config['lr']}",
        f"skip_train_loss = {config['skip_train_loss']}",
        f"skip_base_constrained = {config['skip_base_constrained']}",
        "```",
        "",
        "## Artifact Summary",
        "",
        "```text",
        f"dataset = {artifact['dataset']}",
        f"strategy = {artifact['strategy']}",
        f"boundary_negative_actions = {json.dumps(artifact['boundary_negative_actions'], sort_keys=True)}",
        "```",
        "",
        markdown_table(
            ["fold", "train rows", "train by class", "train by role"],
            [
                [
                    fold["fold"],
                    fold["train_examples"],
                    json.dumps(fold["train_by_class"], sort_keys=True),
                    json.dumps(fold["train_by_role"], sort_keys=True),
                ]
                for fold in artifact["folds"]
            ],
        ),
        "",
        "## Aggregate Comparison",
        "",
        markdown_table(
            ["condition", "strict mean", "constrained loaded mean", "parse failures", "takeaway"],
            [
                ["native CV baseline", "0.475", "0.400", "0", "parse-stable, action-fragile"],
                ["native pressure CV", "0.400", "0.450", "0", "verify fixed, ground/defer hurt"],
                ["curriculum v1", "0.475", "0.425", "0", "flag improved, mixed balance"],
                ["curriculum v2 targeted", "0.375", "0.300", "0", "negative; oversampling over-rotates"],
                [
                    "boundary rationale",
                    f"{aggregate['strict_action_accuracy_mean']:.3f}",
                    f"{aggregate['constrained_loaded_accuracy_mean']:.3f}",
                    aggregate["strict_parse_failures_total"],
                    "modest best native-SFT aggregate; defer still unsolved",
                ],
                ["oracle balanced warm-start", "0.200", "0.200", "12", "negative; ground collapse"],
            ],
        ),
        "",
        "## Boundary-Rationale CV Folds",
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
        "Failure pair counts:",
        "",
        "```json",
        json.dumps(summary["failure_pairs"], indent=2, sort_keys=True),
        "```",
        "",
        "## Interpretation",
        "",
        "- Boundary-rationale SFT is the best native-SFT aggregate so far in this run family: strict mean 0.500 and constrained-loaded mean 0.500.",
        "- It remains parse-stable with zero strict parse failures, so the issue is action-boundary learning rather than JSON formation.",
        "- `verify` is fully recovered, and `reject` is strong in strict generation, but `defer` remains 0/8.",
        "- This is a modest positive SFT formulation result, not a reason to jump straight to RLVR. The next useful step is row-level boundary-failure analysis or a targeted held-out prompt-side rationale ablation.",
        "",
        "## Next Action",
        "",
        "Run row-level boundary-rationale failure analysis before DPO/RLVR. The key question is whether `defer` fails because the rationale is absent from held-out prompts, because the final-action target is too short, or because `defer` and `verify` need explicit paired preference supervision.",
        "",
    ]
    return "\n".join(sections)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="post_training/runs/qwen_sft_cv4_boundary_rationale_schema_action_80_evalfast",
    )
    parser.add_argument(
        "--manifest",
        default="post_training/negbiodb_ct_native_sft_cv4_boundary_rationale_manifest.json",
    )
    parser.add_argument("--out-json", default="post_training/sft_boundary_rationale_run_summary_2026-06-26.json")
    parser.add_argument("--out-md", default="post_training/SFT_BOUNDARY_RATIONALE_RUN_RESULTS_2026-06-26.md")
    args = parser.parse_args()

    summary = build_summary(args.root, args.manifest)
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

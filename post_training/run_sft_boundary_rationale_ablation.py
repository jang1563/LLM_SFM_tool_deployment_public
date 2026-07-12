#!/usr/bin/env python3
"""Run held-out rationale ablations for boundary-rationale SFT folds."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from post_training.build_sft_boundary_rationale_data import (  # noqa: E402
    BOUNDARY_NEGATIVES,
    class_counts,
    rationale_copy,
    role_counts,
)
from post_training.evidence_rationale import evidence_rationale_copy  # noqa: E402
from post_training.run_sft_cv_sweep import (  # noqa: E402
    command_with_download,
    load_json,
    run_command,
    summary_value,
)
from post_training.split_sft_data import load_jsonl, write_jsonl  # noqa: E402


@dataclass(frozen=True)
class AblationConfig:
    model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    tasks: str = "negbiodb_ct/tasks_pilot.jsonl"
    source_state_root: str = "post_training/runs/qwen_sft_cv4_boundary_rationale_schema_action_80_evalfast"
    batch_size: int = 2
    max_length: int = 512
    device: str = "auto"
    score_mode: str = "mean"
    max_new_tokens: int = 64
    allow_download: bool = False


def build_heldout_rationale_rows(
    rows: list[Mapping[str, Any]],
    *,
    dataset: str,
    rationale_mode: str,
) -> list[dict[str, Any]]:
    if rationale_mode == "oracle":
        return [
            rationale_copy(row, dataset=dataset, pair_index=pair_index)
            for pair_index, row in enumerate(rows)
        ]
    if rationale_mode == "evidence":
        return [
            evidence_rationale_copy(
                row,
                dataset=dataset,
                pair_index=pair_index,
                strategy="heldout_evidence_boundary_rationale_ablation_v1",
                action_hint_label="Correct final action",
            )
            for pair_index, row in enumerate(rows)
        ]
    raise ValueError(f"Unsupported rationale_mode: {rationale_mode}")


def build_ablation_manifest(
    manifest: Mapping[str, Any],
    *,
    out_dir: str | Path,
    prefix: str,
    dataset: str,
    rationale_mode: str,
) -> dict[str, Any]:
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    folds = []
    for fold in manifest["fold_manifests"]:
        heldout_rows = load_jsonl(fold["heldout"])
        ablation_rows = build_heldout_rationale_rows(
            heldout_rows,
            dataset=dataset,
            rationale_mode=rationale_mode,
        )
        heldout_path = out_root / f"{prefix}_fold{fold['fold']}_heldout.jsonl"
        write_jsonl(heldout_path, ablation_rows)
        folds.append({
            "fold": fold["fold"],
            "source_heldout": fold["heldout"],
            "heldout": str(heldout_path),
            "heldout_examples": len(ablation_rows),
            "heldout_by_class": class_counts(ablation_rows),
            "heldout_by_role": role_counts(ablation_rows),
            "evidence_action_mismatches": sum(
                1 for row in ablation_rows if row.get("evidence_matches_action_class") is False
            ),
        })
    return {
        "dataset": dataset,
        "strategy": f"heldout_{rationale_mode}_boundary_rationale_ablation_v1",
        "rationale_mode": rationale_mode,
        "source_cv_manifest": manifest.get("source_cv_manifest"),
        "source_boundary_manifest_strategy": manifest.get("strategy"),
        "boundary_negative_actions": BOUNDARY_NEGATIVES,
        "fold_manifests": folds,
    }


def fold_commands(
    fold: Mapping[str, Any],
    *,
    fold_dir: str | Path,
    config: AblationConfig,
) -> dict[str, list[str]]:
    fold_root = Path(fold_dir)
    state_path = (
        Path(config.source_state_root)
        / f"fold{fold['fold']}"
        / "train"
        / "trainable_state.pt"
    )
    heldout_sft = str(fold["heldout"])
    heldout_limit = str(fold["heldout_examples"])
    commands: dict[str, list[str]] = {
        "heldout_loss": command_with_download([
            sys.executable,
            "post_training/evaluate_sft_loss.py",
            "--model",
            config.model,
            "--state",
            str(state_path),
            "--sft",
            heldout_sft,
            "--limit",
            heldout_limit,
            "--batch-size",
            str(config.batch_size),
            "--max-length",
            str(config.max_length),
            "--device",
            config.device,
            "--out",
            str(fold_root / "heldout_loss.json"),
        ], config.allow_download),
        "heldout_decision_eval": command_with_download([
            sys.executable,
            "post_training/run_sft_decision_eval.py",
            "--model",
            config.model,
            "--state",
            str(state_path),
            "--sft",
            heldout_sft,
            "--tasks",
            config.tasks,
            "--limit",
            heldout_limit,
            "--device",
            config.device,
            "--max-new-tokens",
            str(config.max_new_tokens),
            "--out",
            str(fold_root / "heldout_decision_eval.json"),
        ], config.allow_download),
        "heldout_constrained_loaded": command_with_download([
            sys.executable,
            "post_training/run_sft_constrained_eval.py",
            "--model",
            config.model,
            "--state",
            str(state_path),
            "--sft",
            heldout_sft,
            "--tasks",
            config.tasks,
            "--limit",
            heldout_limit,
            "--max-length",
            str(config.max_length),
            "--device",
            config.device,
            "--score-mode",
            config.score_mode,
            "--out",
            str(fold_root / "heldout_constrained_loaded.json"),
        ], config.allow_download),
    }
    return commands


def fold_summary(
    fold: Mapping[str, Any],
    *,
    fold_dir: str | Path,
    command_records: list[Mapping[str, Any]],
    source_state_root: str | Path,
) -> dict[str, Any]:
    fold_root = Path(fold_dir)
    state_path = Path(source_state_root) / f"fold{fold['fold']}" / "train" / "trainable_state.pt"
    return {
        "fold": fold["fold"],
        "source_state": str(state_path),
        "heldout": fold["heldout"],
        "source_heldout": fold["source_heldout"],
        "heldout_examples": fold["heldout_examples"],
        "heldout_by_class": fold["heldout_by_class"],
        "heldout_by_role": fold["heldout_by_role"],
        "commands": list(command_records),
        "heldout_teacher_forced_loaded_loss": summary_value(fold_root / "heldout_loss.json", "loaded", "loss"),
        "heldout_strict_action_accuracy": summary_value(
            fold_root / "heldout_decision_eval.json",
            "summary",
            "action_accuracy",
        ),
        "heldout_strict_parse_failures": summary_value(
            fold_root / "heldout_decision_eval.json",
            "summary",
            "parse_failures",
        ),
        "heldout_strict_by_class": summary_value(
            fold_root / "heldout_decision_eval.json",
            "summary",
            "by_class",
        ),
        "heldout_constrained_loaded_action_accuracy": summary_value(
            fold_root / "heldout_constrained_loaded.json",
            "summary",
            "action_accuracy",
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--boundary-manifest", default="post_training/negbiodb_ct_native_sft_cv4_boundary_rationale_manifest.json")
    parser.add_argument("--ablation-out-dir", default="post_training/boundary_rationale_heldout_ablation")
    parser.add_argument("--ablation-prefix", default="negbiodb_ct_native_sft_cv4_boundary_rationale_heldout_oracle_v1")
    parser.add_argument("--ablation-manifest-out", default="post_training/negbiodb_ct_native_sft_cv4_boundary_rationale_heldout_ablation_manifest.json")
    parser.add_argument("--dataset", default="negbiodb_ct_native_sft_boundary_rationale_heldout_oracle_v1")
    parser.add_argument("--rationale-mode", choices=("oracle", "evidence"), default="oracle")
    parser.add_argument("--out-dir", default="post_training/runs/qwen_sft_cv4_boundary_rationale_heldout_oracle_ablation")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--model", default=AblationConfig.model)
    parser.add_argument("--tasks", default=AblationConfig.tasks)
    parser.add_argument("--source-state-root", default=AblationConfig.source_state_root)
    parser.add_argument("--batch-size", type=int, default=AblationConfig.batch_size)
    parser.add_argument("--max-length", type=int, default=AblationConfig.max_length)
    parser.add_argument("--device", default=AblationConfig.device)
    parser.add_argument("--score-mode", choices=("mean", "sum"), default=AblationConfig.score_mode)
    parser.add_argument("--max-new-tokens", type=int, default=AblationConfig.max_new_tokens)
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()

    source_manifest = load_json(args.boundary_manifest)
    ablation_manifest = build_ablation_manifest(
        source_manifest,
        out_dir=args.ablation_out_dir,
        prefix=args.ablation_prefix,
        dataset=args.dataset,
        rationale_mode=args.rationale_mode,
    )
    ablation_manifest["source_boundary_manifest"] = args.boundary_manifest
    ablation_manifest["out_dir"] = args.ablation_out_dir
    ablation_manifest["prefix"] = args.ablation_prefix
    Path(args.ablation_manifest_out).write_text(json.dumps(ablation_manifest, indent=2, sort_keys=True) + "\n")

    config = AblationConfig(
        model=args.model,
        tasks=args.tasks,
        source_state_root=args.source_state_root,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device,
        score_mode=args.score_mode,
        max_new_tokens=args.max_new_tokens,
        allow_download=args.allow_download,
    )
    out_root = Path(args.out_dir)
    fold_summaries = []
    for fold in ablation_manifest["fold_manifests"]:
        fold_id = int(fold["fold"])
        fold_root = out_root / f"fold{fold_id}"
        commands = fold_commands(fold, fold_dir=fold_root, config=config)
        records = []
        for label, command in commands.items():
            records.append(
                run_command(
                    label,
                    command,
                    log_path=fold_root / "logs" / f"{label}.log",
                    dry_run=args.dry_run,
                )
            )
        fold_summaries.append(
            fold_summary(
                fold,
                fold_dir=fold_root,
                command_records=records,
                source_state_root=config.source_state_root,
            )
        )

    run_summary = {
        "condition": f"boundary_rationale_heldout_{args.rationale_mode}_ablation",
        "dry_run": args.dry_run,
        "config": asdict(config),
        "source_boundary_manifest": args.boundary_manifest,
        "ablation_manifest": args.ablation_manifest_out,
        "out_dir": args.out_dir,
        "folds": fold_summaries,
    }
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "summary.json").write_text(json.dumps(run_summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(run_summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

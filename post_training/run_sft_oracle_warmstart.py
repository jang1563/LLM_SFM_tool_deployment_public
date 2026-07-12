#!/usr/bin/env python3
"""Train on the larger oracle SFT artifact and evaluate native held-out folds."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from post_training.run_sft_cv_sweep import (  # noqa: E402
    SweepConfig,
    command_with_download,
    load_json,
    run_command,
    summary_value,
)


def eval_sets_from_cv_manifest(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "name": f"fold{fold['fold']}_heldout",
            "sft": fold["heldout"],
            "examples": fold["heldout_examples"],
            "by_class": fold["heldout_by_class"],
        }
        for fold in manifest["fold_manifests"]
    ]


def train_command(
    *,
    train_sft: str,
    train_limit: int,
    out_dir: str | Path,
    config: SweepConfig,
) -> list[str]:
    return command_with_download([
        sys.executable,
        "post_training/run_sft_smoke.py",
        "--model",
        config.model,
        "--sft",
        train_sft,
        "--limit",
        str(train_limit),
        "--max-steps",
        str(config.max_steps),
        "--batch-size",
        str(config.batch_size),
        "--max-length",
        str(config.max_length),
        "--train-last-layers",
        str(config.train_last_layers),
        "--lr",
        str(config.lr),
        "--device",
        config.device,
        "--out-dir",
        str(out_dir),
    ], config.allow_download)


def eval_commands(
    eval_set: Mapping[str, Any],
    *,
    eval_dir: str | Path,
    state_path: str | Path,
    config: SweepConfig,
) -> dict[str, list[str]]:
    out_root = Path(eval_dir)
    sft = str(eval_set["sft"])
    limit = str(eval_set["examples"])
    commands: dict[str, list[str]] = {
        "loss": command_with_download([
            sys.executable,
            "post_training/evaluate_sft_loss.py",
            "--model",
            config.model,
            "--state",
            str(state_path),
            "--sft",
            sft,
            "--limit",
            limit,
            "--batch-size",
            str(config.batch_size),
            "--max-length",
            str(config.max_length),
            "--device",
            config.device,
            "--out",
            str(out_root / "loss.json"),
        ], config.allow_download),
    }

    if not config.skip_strict_generation:
        commands["decision_eval"] = command_with_download([
            sys.executable,
            "post_training/run_sft_decision_eval.py",
            "--model",
            config.model,
            "--state",
            str(state_path),
            "--sft",
            sft,
            "--tasks",
            config.tasks,
            "--limit",
            limit,
            "--device",
            config.device,
            "--max-new-tokens",
            str(config.max_new_tokens),
            "--out",
            str(out_root / "decision_eval.json"),
        ], config.allow_download)

    if not config.skip_constrained:
        if not config.skip_base_constrained:
            commands["constrained_base"] = command_with_download([
                sys.executable,
                "post_training/run_sft_constrained_eval.py",
                "--model",
                config.model,
                "--sft",
                sft,
                "--tasks",
                config.tasks,
                "--limit",
                limit,
                "--max-length",
                str(config.max_length),
                "--device",
                config.device,
                "--score-mode",
                config.score_mode,
                "--out",
                str(out_root / "constrained_base.json"),
            ], config.allow_download)
        commands["constrained_loaded"] = command_with_download([
            sys.executable,
            "post_training/run_sft_constrained_eval.py",
            "--model",
            config.model,
            "--state",
            str(state_path),
            "--sft",
            sft,
            "--tasks",
            config.tasks,
            "--limit",
            limit,
            "--max-length",
            str(config.max_length),
            "--device",
            config.device,
            "--score-mode",
            config.score_mode,
            "--out",
            str(out_root / "constrained_loaded.json"),
        ], config.allow_download)

    return commands


def eval_summary(
    eval_set: Mapping[str, Any],
    *,
    eval_dir: str | Path,
    command_records: list[Mapping[str, Any]],
) -> dict[str, Any]:
    out_root = Path(eval_dir)
    return {
        "name": eval_set["name"],
        "sft": eval_set["sft"],
        "examples": eval_set["examples"],
        "by_class": eval_set["by_class"],
        "commands": list(command_records),
        "teacher_forced_loaded_loss": summary_value(out_root / "loss.json", "loaded", "loss"),
        "strict_action_accuracy": summary_value(out_root / "decision_eval.json", "summary", "action_accuracy"),
        "strict_parse_failures": summary_value(out_root / "decision_eval.json", "summary", "parse_failures"),
        "strict_by_class": summary_value(out_root / "decision_eval.json", "summary", "by_class"),
        "constrained_base_action_accuracy": summary_value(
            out_root / "constrained_base.json",
            "summary",
            "action_accuracy",
        ),
        "constrained_loaded_action_accuracy": summary_value(
            out_root / "constrained_loaded.json",
            "summary",
            "action_accuracy",
        ),
    }


def parse_only_eval(raw: str | None) -> set[str] | None:
    if raw is None:
        return None
    return {part.strip() for part in raw.split(",") if part.strip()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-sft", default="post_training/negbiodb_ct_oracle_sft_v1.jsonl")
    parser.add_argument("--train-limit", type=int, default=400)
    parser.add_argument("--eval-cv-manifest", default="post_training/negbiodb_ct_native_sft_cv4_manifest.json")
    parser.add_argument("--only-eval", help="Comma-separated eval names, e.g. fold0_heldout,fold1_heldout")
    parser.add_argument("--out-dir", default="post_training/runs/qwen_oracle400_warmstart_cvheldout")
    parser.add_argument("--model", default=SweepConfig.model)
    parser.add_argument("--tasks", default=SweepConfig.tasks)
    parser.add_argument("--max-steps", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=SweepConfig.batch_size)
    parser.add_argument("--max-length", type=int, default=SweepConfig.max_length)
    parser.add_argument("--train-last-layers", type=int, default=SweepConfig.train_last_layers)
    parser.add_argument("--lr", type=float, default=SweepConfig.lr)
    parser.add_argument("--device", default=SweepConfig.device)
    parser.add_argument("--score-mode", choices=("mean", "sum"), default=SweepConfig.score_mode)
    parser.add_argument("--max-new-tokens", type=int, default=SweepConfig.max_new_tokens)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--skip-strict-generation", action="store_true")
    parser.add_argument("--skip-constrained", action="store_true")
    parser.add_argument("--skip-base-constrained", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = SweepConfig(
        model=args.model,
        tasks=args.tasks,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        max_length=args.max_length,
        train_last_layers=args.train_last_layers,
        lr=args.lr,
        device=args.device,
        score_mode=args.score_mode,
        max_new_tokens=args.max_new_tokens,
        allow_download=args.allow_download,
        skip_train_loss=True,
        skip_strict_generation=args.skip_strict_generation,
        skip_constrained=args.skip_constrained,
        skip_base_constrained=args.skip_base_constrained,
    )
    out_root = Path(args.out_dir)
    train_dir = out_root / "train"
    state_path = train_dir / "trainable_state.pt"
    out_root.mkdir(parents=True, exist_ok=True)

    train_record = run_command(
        "train_oracle_sft",
        train_command(
            train_sft=args.train_sft,
            train_limit=args.train_limit,
            out_dir=train_dir,
            config=config,
        ),
        log_path=out_root / "logs" / "train_oracle_sft.log",
        dry_run=args.dry_run,
    )

    eval_sets = eval_sets_from_cv_manifest(load_json(args.eval_cv_manifest))
    only_eval = parse_only_eval(args.only_eval)
    if only_eval is not None:
        eval_sets = [item for item in eval_sets if item["name"] in only_eval]

    eval_summaries = []
    for eval_set in eval_sets:
        eval_dir = out_root / eval_set["name"]
        records = []
        for label, command in eval_commands(
            eval_set,
            eval_dir=eval_dir,
            state_path=state_path,
            config=config,
        ).items():
            records.append(
                run_command(
                    label,
                    command,
                    log_path=eval_dir / "logs" / f"{label}.log",
                    dry_run=args.dry_run,
                )
            )
        eval_summaries.append(eval_summary(eval_set, eval_dir=eval_dir, command_records=records))

    train_report = load_json(train_dir / "report.json") if (train_dir / "report.json").exists() else None
    summary = {
        "condition": "oracle_sft_warmstart_native_cvheldout",
        "dry_run": args.dry_run,
        "train_sft": args.train_sft,
        "train_limit": args.train_limit,
        "eval_cv_manifest": args.eval_cv_manifest,
        "out_dir": args.out_dir,
        "config": asdict(config),
        "train_command": train_record,
        "train_first_loss": train_report["losses"][0] if train_report and train_report.get("losses") else None,
        "train_last_loss": train_report["losses"][-1] if train_report and train_report.get("losses") else None,
        "train_loss_delta": train_report.get("loss_delta") if train_report else None,
        "evals": eval_summaries,
    }
    summary_path = out_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

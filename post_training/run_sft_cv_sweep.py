#!/usr/bin/env python3
"""Run or plan a repeated SFT train/eval sweep across CV folds."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class SweepConfig:
    model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    tasks: str = "negbiodb_ct/tasks_pilot.jsonl"
    max_steps: int = 80
    batch_size: int = 2
    max_length: int = 512
    train_last_layers: int = 2
    lr: float = 5e-5
    device: str = "auto"
    score_mode: str = "mean"
    max_new_tokens: int = 64
    allow_download: bool = False
    skip_train_loss: bool = False
    skip_strict_generation: bool = False
    skip_constrained: bool = False
    skip_base_constrained: bool = False


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def maybe_load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return load_json(path)


def command_with_download(command: list[str], allow_download: bool) -> list[str]:
    return command + (["--allow-download"] if allow_download else [])


def fold_commands(
    fold: Mapping[str, Any],
    *,
    fold_dir: str | Path,
    config: SweepConfig,
) -> dict[str, list[str]]:
    fold_root = Path(fold_dir)
    train_dir = fold_root / "train"
    state_path = train_dir / "trainable_state.pt"
    train_sft = str(fold["train"])
    heldout_sft = str(fold["heldout"])
    train_limit = str(fold["train_examples"])
    heldout_limit = str(fold["heldout_examples"])

    commands: dict[str, list[str]] = {
        "train": command_with_download([
            sys.executable,
            "post_training/run_sft_smoke.py",
            "--model",
            config.model,
            "--sft",
            train_sft,
            "--limit",
            train_limit,
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
            str(train_dir),
        ], config.allow_download),
    }

    if not config.skip_train_loss:
        commands["train_loss"] = command_with_download([
            sys.executable,
            "post_training/evaluate_sft_loss.py",
            "--model",
            config.model,
            "--state",
            str(state_path),
            "--sft",
            train_sft,
            "--limit",
            train_limit,
            "--batch-size",
            str(config.batch_size),
            "--max-length",
            str(config.max_length),
            "--device",
            config.device,
            "--out",
            str(fold_root / "train_loss.json"),
        ], config.allow_download)

    commands["heldout_loss"] = command_with_download([
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
    ], config.allow_download)

    if not config.skip_strict_generation:
        commands["heldout_decision_eval"] = command_with_download([
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
        ], config.allow_download)

    if not config.skip_constrained:
        if not config.skip_base_constrained:
            commands["heldout_constrained_base"] = command_with_download([
                sys.executable,
                "post_training/run_sft_constrained_eval.py",
                "--model",
                config.model,
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
                str(fold_root / "heldout_constrained_base.json"),
            ], config.allow_download)
        commands["heldout_constrained_loaded"] = command_with_download([
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
        ], config.allow_download)

    return commands


def run_command(label: str, command: list[str], *, log_path: Path, dry_run: bool) -> dict[str, Any]:
    record = {
        "label": label,
        "command": command,
        "log": str(log_path),
        "dry_run": dry_run,
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        log_path.write_text("$ " + " ".join(command) + "\n")
        return record | {"returncode": None}

    completed = subprocess.run(command, text=True, capture_output=True)
    log_path.write_text(
        "$ "
        + " ".join(command)
        + "\n\n"
        + "STDOUT\n"
        + completed.stdout
        + "\nSTDERR\n"
        + completed.stderr
    )
    record["returncode"] = completed.returncode
    if completed.returncode != 0:
        raise RuntimeError(f"{label} failed with exit code {completed.returncode}; see {log_path}")
    return record


def summary_value(path: Path, *keys: str) -> Any:
    data = maybe_load_json(path)
    if data is None:
        return None
    value: Any = data
    for key in keys:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def fold_summary(
    fold: Mapping[str, Any],
    *,
    fold_dir: str | Path,
    command_records: list[Mapping[str, Any]],
) -> dict[str, Any]:
    fold_root = Path(fold_dir)
    train_report = maybe_load_json(fold_root / "train" / "report.json")
    losses = train_report.get("losses", []) if train_report else []
    return {
        "fold": fold["fold"],
        "train": fold["train"],
        "heldout": fold["heldout"],
        "train_examples": fold["train_examples"],
        "heldout_examples": fold["heldout_examples"],
        "heldout_by_class": fold["heldout_by_class"],
        "commands": list(command_records),
        "train_first_loss": losses[0] if losses else None,
        "train_last_loss": losses[-1] if losses else None,
        "train_loss_delta": train_report.get("loss_delta") if train_report else None,
        "train_teacher_forced_loaded_loss": summary_value(fold_root / "train_loss.json", "loaded", "loss"),
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
        "heldout_constrained_base_action_accuracy": summary_value(
            fold_root / "heldout_constrained_base.json",
            "summary",
            "action_accuracy",
        ),
        "heldout_constrained_loaded_action_accuracy": summary_value(
            fold_root / "heldout_constrained_loaded.json",
            "summary",
            "action_accuracy",
        ),
        "heldout_strict_by_class": summary_value(
            fold_root / "heldout_decision_eval.json",
            "summary",
            "by_class",
        ),
    }


def select_folds(manifest: Mapping[str, Any], only_fold: set[int] | None) -> list[Mapping[str, Any]]:
    folds = list(manifest["fold_manifests"])
    if only_fold is None:
        return folds
    return [fold for fold in folds if int(fold["fold"]) in only_fold]


def parse_fold_set(raw: str | None) -> set[int] | None:
    if raw is None:
        return None
    return {int(part) for part in raw.split(",") if part.strip()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="post_training/negbiodb_ct_native_sft_cv4_manifest.json")
    parser.add_argument("--out-dir", default="post_training/runs/qwen_sft_cv4_schema_action_80")
    parser.add_argument("--only-fold", help="Comma-separated fold IDs to run, e.g. 1,2")
    parser.add_argument("--model", default=SweepConfig.model)
    parser.add_argument("--tasks", default=SweepConfig.tasks)
    parser.add_argument("--max-steps", type=int, default=SweepConfig.max_steps)
    parser.add_argument("--batch-size", type=int, default=SweepConfig.batch_size)
    parser.add_argument("--max-length", type=int, default=SweepConfig.max_length)
    parser.add_argument("--train-last-layers", type=int, default=SweepConfig.train_last_layers)
    parser.add_argument("--lr", type=float, default=SweepConfig.lr)
    parser.add_argument("--device", default=SweepConfig.device)
    parser.add_argument("--score-mode", choices=("mean", "sum"), default=SweepConfig.score_mode)
    parser.add_argument("--max-new-tokens", type=int, default=SweepConfig.max_new_tokens)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--skip-train-loss", action="store_true")
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
        skip_train_loss=args.skip_train_loss,
        skip_strict_generation=args.skip_strict_generation,
        skip_constrained=args.skip_constrained,
        skip_base_constrained=args.skip_base_constrained,
    )
    manifest = load_json(args.manifest)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    fold_summaries = []
    for fold in select_folds(manifest, parse_fold_set(args.only_fold)):
        fold_id = int(fold["fold"])
        fold_root = out_root / f"fold{fold_id}"
        commands = fold_commands(fold, fold_dir=fold_root, config=config)
        command_records = []
        for label, command in commands.items():
            command_records.append(
                run_command(
                    label,
                    command,
                    log_path=fold_root / "logs" / f"{label}.log",
                    dry_run=args.dry_run,
                )
            )
        fold_summaries.append(fold_summary(fold, fold_dir=fold_root, command_records=command_records))

    summary = {
        "condition": "sft_cv_fold_sweep",
        "dry_run": args.dry_run,
        "manifest": args.manifest,
        "out_dir": args.out_dir,
        "config": asdict(config),
        "folds": fold_summaries,
    }
    summary_path = out_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

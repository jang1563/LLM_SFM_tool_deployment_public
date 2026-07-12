#!/usr/bin/env python3
"""Build class-pressure SFT artifacts for the next SFT formulation pass."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from post_training.split_sft_data import by_action_class, load_jsonl, write_jsonl  # noqa: E402


def parse_multipliers(raw: str) -> dict[str, int]:
    multipliers: dict[str, int] = {}
    if not raw.strip():
        return multipliers
    for item in raw.split(","):
        if "=" not in item:
            raise ValueError(f"Multiplier must use class=count syntax: {item!r}")
        label, value = item.split("=", 1)
        label = label.strip()
        if not label:
            raise ValueError(f"Missing class label in multiplier: {item!r}")
        count = int(value)
        if count < 1:
            raise ValueError(f"Multiplier must be >= 1 for {label!r}")
        multipliers[label] = count
    return multipliers


def pressure_copy(
    row: Mapping[str, Any],
    *,
    dataset: str,
    replicate_index: int,
    multiplier: int,
    strategy: str,
) -> dict[str, Any]:
    out = dict(row)
    source_id = str(row["id"])
    out["id"] = f"{source_id}::pressure::{replicate_index}"
    out["dataset"] = dataset
    out["source_example_id"] = source_id
    out["pressure_strategy"] = strategy
    out["pressure_replicate_index"] = replicate_index
    out["pressure_multiplier"] = multiplier
    return out


def apply_class_pressure(
    rows: list[Mapping[str, Any]],
    *,
    multipliers: Mapping[str, int],
    dataset: str,
    strategy: str = "class_multiplier",
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        action_class = str(row["action_class"])
        multiplier = multipliers.get(action_class, 1)
        for replicate in range(multiplier):
            out.append(
                pressure_copy(
                    row,
                    dataset=dataset,
                    replicate_index=replicate,
                    multiplier=multiplier,
                    strategy=strategy,
                )
            )
    return out


def balance_to_max_class_count(
    rows: list[Mapping[str, Any]],
    *,
    dataset: str,
    seed: int,
) -> list[dict[str, Any]]:
    groups = by_action_class(rows)
    target = max(len(class_rows) for class_rows in groups.values())
    out: list[dict[str, Any]] = []
    for action_class, class_rows in sorted(groups.items()):
        shuffled = list(class_rows)
        random.Random(f"{seed}:{action_class}").shuffle(shuffled)
        for i in range(target):
            row = shuffled[i % len(shuffled)]
            out.append(
                pressure_copy(
                    row,
                    dataset=dataset,
                    replicate_index=i // len(shuffled),
                    multiplier=target,
                    strategy="class_balance_to_max",
                )
            )
    return out


def class_counts(rows: list[Mapping[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(row["action_class"] for row in rows).items()))


def build_native_cv_pressure(
    manifest: Mapping[str, Any],
    *,
    multipliers: Mapping[str, int],
    out_dir: str | Path,
    prefix: str,
    dataset: str,
) -> dict[str, Any]:
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    fold_manifests = []
    for fold in manifest["fold_manifests"]:
        train_rows = load_jsonl(fold["train"])
        pressure_rows = apply_class_pressure(
            train_rows,
            multipliers=multipliers,
            dataset=dataset,
            strategy="native_cv_verify_flag_pressure",
        )
        train_path = out_root / f"{prefix}_fold{fold['fold']}_train.jsonl"
        write_jsonl(train_path, pressure_rows)
        fold_manifests.append({
            "fold": fold["fold"],
            "source_train": fold["train"],
            "train": str(train_path),
            "heldout": fold["heldout"],
            "source_train_examples": len(train_rows),
            "train_examples": len(pressure_rows),
            "source_train_by_class": class_counts(train_rows),
            "train_by_class": class_counts(pressure_rows),
            "heldout_examples": fold["heldout_examples"],
            "heldout_by_class": fold["heldout_by_class"],
        })
    return {
        "dataset": dataset,
        "source_manifest": manifest.get("source"),
        "source_cv_manifest_folds": manifest.get("folds"),
        "strategy": "native_cv_verify_flag_pressure",
        "multipliers": dict(sorted(multipliers.items())),
        "fold_manifests": fold_manifests,
    }


def build_oracle_balanced(
    rows: list[Mapping[str, Any]],
    *,
    dataset: str,
    seed: int,
    source: str | Path,
    out: str | Path,
) -> dict[str, Any]:
    balanced = balance_to_max_class_count(rows, dataset=dataset, seed=seed)
    write_jsonl(out, balanced)
    return {
        "dataset": dataset,
        "source": str(source),
        "sft": str(out),
        "strategy": "class_balance_to_max",
        "seed": seed,
        "source_examples": len(rows),
        "sft_examples": len(balanced),
        "source_by_class": class_counts(rows),
        "by_class": class_counts(balanced),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv-manifest", default="post_training/negbiodb_ct_native_sft_cv4_manifest.json")
    parser.add_argument("--native-out-dir", default="post_training/pressure")
    parser.add_argument("--native-prefix", default="negbiodb_ct_native_sft_cv4_pressure_v1")
    parser.add_argument("--native-manifest-out", default="post_training/negbiodb_ct_native_sft_cv4_pressure_manifest.json")
    parser.add_argument("--native-dataset", default="negbiodb_ct_native_sft_pressure_v1")
    parser.add_argument("--native-multipliers", default="flag=3,verify=3")
    parser.add_argument("--oracle-sft", default="post_training/negbiodb_ct_oracle_sft_v1.jsonl")
    parser.add_argument("--oracle-out", default="post_training/negbiodb_ct_oracle_sft_balanced_v1.jsonl")
    parser.add_argument("--oracle-manifest-out", default="post_training/negbiodb_ct_oracle_sft_balanced_manifest.json")
    parser.add_argument("--oracle-dataset", default="negbiodb_ct_oracle_sft_balanced_v1")
    parser.add_argument("--seed", type=int, default=20260626)
    args = parser.parse_args()

    cv_manifest = json.loads(Path(args.cv_manifest).read_text())
    multipliers = parse_multipliers(args.native_multipliers)
    native_manifest = build_native_cv_pressure(
        cv_manifest,
        multipliers=multipliers,
        out_dir=args.native_out_dir,
        prefix=args.native_prefix,
        dataset=args.native_dataset,
    )
    native_manifest["source_cv_manifest"] = args.cv_manifest
    native_manifest["out_dir"] = args.native_out_dir
    native_manifest["prefix"] = args.native_prefix
    Path(args.native_manifest_out).write_text(json.dumps(native_manifest, indent=2, sort_keys=True) + "\n")

    oracle_rows = load_jsonl(args.oracle_sft)
    oracle_manifest = build_oracle_balanced(
        oracle_rows,
        dataset=args.oracle_dataset,
        seed=args.seed,
        source=args.oracle_sft,
        out=args.oracle_out,
    )
    Path(args.oracle_manifest_out).write_text(json.dumps(oracle_manifest, indent=2, sort_keys=True) + "\n")

    print(json.dumps({
        "native": native_manifest,
        "oracle": oracle_manifest,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

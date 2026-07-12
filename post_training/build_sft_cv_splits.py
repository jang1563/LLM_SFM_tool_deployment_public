#!/usr/bin/env python3
"""Create deterministic stratified cross-validation splits for SFT examples."""

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

from post_training.split_sft_data import by_action_class, load_jsonl, write_jsonl


def fold_chunks(
    rows: list[Mapping[str, Any]],
    *,
    folds: int,
    seed: int,
) -> dict[int, set[str]]:
    if folds <= 1:
        raise ValueError("folds must be greater than 1")

    heldout_by_fold = {fold: set() for fold in range(folds)}
    for action_class, class_rows in sorted(by_action_class(rows).items()):
        if len(class_rows) % folds != 0:
            raise ValueError(
                f"Class {action_class!r} has {len(class_rows)} rows, not divisible by {folds} folds"
            )
        shuffled = list(class_rows)
        random.Random(f"{seed}:{action_class}").shuffle(shuffled)
        chunk_size = len(shuffled) // folds
        for fold in range(folds):
            heldout_by_fold[fold].update(
                str(row["id"]) for row in shuffled[fold * chunk_size : (fold + 1) * chunk_size]
            )
    return heldout_by_fold


def split_for_fold(
    rows: list[Mapping[str, Any]],
    heldout_ids: set[str],
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    train = [row for row in rows if row["id"] not in heldout_ids]
    heldout = [row for row in rows if row["id"] in heldout_ids]
    return train, heldout


def class_counts(rows: list[Mapping[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(row["action_class"] for row in rows).items()))


def build_folds(
    rows: list[Mapping[str, Any]],
    *,
    folds: int,
    seed: int,
    out_dir: str | Path,
    prefix: str,
) -> dict[str, Any]:
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    heldout_by_fold = fold_chunks(rows, folds=folds, seed=seed)

    fold_manifests = []
    all_heldout_ids: list[str] = []
    for fold in range(folds):
        train, heldout = split_for_fold(rows, heldout_by_fold[fold])
        train_path = out_root / f"{prefix}_fold{fold}_train.jsonl"
        heldout_path = out_root / f"{prefix}_fold{fold}_heldout.jsonl"
        write_jsonl(train_path, train)
        write_jsonl(heldout_path, heldout)
        all_heldout_ids.extend(str(row["id"]) for row in heldout)
        fold_manifests.append({
            "fold": fold,
            "train": str(train_path),
            "heldout": str(heldout_path),
            "train_examples": len(train),
            "heldout_examples": len(heldout),
            "train_by_class": class_counts(train),
            "heldout_by_class": class_counts(heldout),
            "heldout_task_ids": [str(row["task_id"]) for row in heldout],
        })

    coverage = Counter(all_heldout_ids)
    return {
        "folds": folds,
        "seed": seed,
        "source_examples": len(rows),
        "source_by_class": class_counts(rows),
        "fold_manifests": fold_manifests,
        "heldout_coverage_unique_examples": len(coverage),
        "heldout_coverage_min_count": min(coverage.values()) if coverage else 0,
        "heldout_coverage_max_count": max(coverage.values()) if coverage else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft", default="post_training/negbiodb_ct_native_sft_v1.jsonl")
    parser.add_argument("--out-dir", default="post_training/cv")
    parser.add_argument("--prefix", default="negbiodb_ct_native_sft_cv4_v1")
    parser.add_argument("--manifest-out", default="post_training/negbiodb_ct_native_sft_cv4_manifest.json")
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260626)
    args = parser.parse_args()

    rows = load_jsonl(args.sft)
    manifest = build_folds(
        rows,
        folds=args.folds,
        seed=args.seed,
        out_dir=args.out_dir,
        prefix=args.prefix,
    )
    manifest["source"] = args.sft
    manifest["out_dir"] = args.out_dir
    manifest["prefix"] = args.prefix
    Path(args.manifest_out).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

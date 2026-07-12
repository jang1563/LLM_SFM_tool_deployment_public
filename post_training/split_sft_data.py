#!/usr/bin/env python3
"""Create deterministic stratified train/held-out SFT splits."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open() as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: list[Mapping[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def by_action_class(rows: list[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        action_class = row.get("action_class")
        if not isinstance(action_class, str) or not action_class:
            raise ValueError(f"Missing action_class in row: {row.get('id')}")
        groups[action_class].append(row)
    return dict(groups)


def split_rows(
    rows: list[Mapping[str, Any]],
    *,
    heldout_per_class: int,
    seed: int,
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    index_by_id = {row["id"]: i for i, row in enumerate(rows)}
    train: list[Mapping[str, Any]] = []
    heldout: list[Mapping[str, Any]] = []

    for action_class, class_rows in sorted(by_action_class(rows).items()):
        if len(class_rows) <= heldout_per_class:
            raise ValueError(
                f"Class {action_class!r} has {len(class_rows)} rows, cannot hold out {heldout_per_class}"
            )
        rng = random.Random(f"{seed}:{action_class}")
        shuffled = list(class_rows)
        rng.shuffle(shuffled)
        heldout_ids = {row["id"] for row in shuffled[:heldout_per_class]}
        for row in class_rows:
            if row["id"] in heldout_ids:
                heldout.append(row)
            else:
                train.append(row)

    train.sort(key=lambda row: index_by_id[row["id"]])
    heldout.sort(key=lambda row: index_by_id[row["id"]])
    return train, heldout


def manifest_for_split(
    source: str | Path,
    train_path: str | Path,
    heldout_path: str | Path,
    train: list[Mapping[str, Any]],
    heldout: list[Mapping[str, Any]],
    *,
    seed: int,
    heldout_per_class: int,
) -> dict[str, Any]:
    return {
        "source": str(source),
        "train": str(train_path),
        "heldout": str(heldout_path),
        "seed": seed,
        "heldout_per_class": heldout_per_class,
        "train_examples": len(train),
        "heldout_examples": len(heldout),
        "train_by_class": dict(sorted(Counter(row["action_class"] for row in train).items())),
        "heldout_by_class": dict(sorted(Counter(row["action_class"] for row in heldout).items())),
        "train_task_ids": [str(row["task_id"]) for row in train],
        "heldout_task_ids": [str(row["task_id"]) for row in heldout],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft", default="post_training/negbiodb_ct_native_sft_v1.jsonl")
    parser.add_argument("--train-out", default="post_training/negbiodb_ct_native_sft_train_v1.jsonl")
    parser.add_argument("--heldout-out", default="post_training/negbiodb_ct_native_sft_heldout_v1.jsonl")
    parser.add_argument("--manifest-out", default="post_training/negbiodb_ct_native_sft_split_manifest.json")
    parser.add_argument("--heldout-per-class", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260626)
    args = parser.parse_args()

    rows = load_jsonl(args.sft)
    train, heldout = split_rows(rows, heldout_per_class=args.heldout_per_class, seed=args.seed)
    write_jsonl(args.train_out, train)
    write_jsonl(args.heldout_out, heldout)
    manifest = manifest_for_split(
        args.sft,
        args.train_out,
        args.heldout_out,
        train,
        heldout,
        seed=args.seed,
        heldout_per_class=args.heldout_per_class,
    )
    Path(args.manifest_out).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

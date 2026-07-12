#!/usr/bin/env python3
"""Create deterministic train/held-out splits for hard boundary preferences."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from post_training.split_sft_data import load_jsonl, write_jsonl  # noqa: E402


DEFAULT_TRAIN_DATASET = "negbiodb_ct_oracle_boundary_preferences_hard_train_v1"
DEFAULT_HELDOUT_DATASET = "negbiodb_ct_oracle_boundary_preferences_hard_heldout_v1"


def grouped_by_mode(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        mode = row.get("failure_mode")
        if not isinstance(mode, str) or not mode:
            raise ValueError(f"Missing failure_mode in row: {row.get('id')}")
        groups[mode].append(row)
    return dict(groups)


def split_by_failure_mode(
    rows: Sequence[Mapping[str, Any]],
    *,
    heldout_per_mode: int,
    seed: int,
    train_dataset: str = DEFAULT_TRAIN_DATASET,
    heldout_dataset: str = DEFAULT_HELDOUT_DATASET,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if heldout_per_mode <= 0:
        raise ValueError("heldout_per_mode must be positive.")

    source_order = {str(row["id"]): index for index, row in enumerate(rows)}
    heldout_ids: set[str] = set()
    for mode, mode_rows in sorted(grouped_by_mode(rows).items()):
        if len(mode_rows) <= heldout_per_mode:
            raise ValueError(
                f"Failure mode {mode!r} has {len(mode_rows)} rows, cannot hold out {heldout_per_mode}."
            )
        rng = random.Random(f"{seed}:{mode}")
        shuffled = list(mode_rows)
        rng.shuffle(shuffled)
        heldout_ids.update(str(row["id"]) for row in shuffled[:heldout_per_mode])

    train: list[dict[str, Any]] = []
    heldout: list[dict[str, Any]] = []
    for row in rows:
        out = dict(row)
        out["source_hard_preference_id"] = row.get("id")
        if str(row["id"]) in heldout_ids:
            out["id"] = f"{row['id']}::heldout"
            out["dataset"] = heldout_dataset
            out["split"] = "heldout"
            heldout.append(out)
        else:
            out["id"] = f"{row['id']}::train"
            out["dataset"] = train_dataset
            out["split"] = "train"
            train.append(out)

    train.sort(key=lambda row: source_order[str(row["source_hard_preference_id"])])
    heldout.sort(key=lambda row: source_order[str(row["source_hard_preference_id"])])
    return train, heldout


def counts(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(key)) for row in rows).items()))


def manifest_for_split(
    *,
    source: str | Path,
    train_out: str | Path,
    heldout_out: str | Path,
    seed: int,
    heldout_per_mode: int,
    train: Sequence[Mapping[str, Any]],
    heldout: Sequence[Mapping[str, Any]],
    train_dataset: str,
    heldout_dataset: str,
) -> dict[str, Any]:
    train_source_ids = {str(row["source_hard_preference_id"]) for row in train}
    heldout_source_ids = {str(row["source_hard_preference_id"]) for row in heldout}
    return {
        "source": str(source),
        "train": str(train_out),
        "heldout": str(heldout_out),
        "seed": seed,
        "heldout_per_mode": heldout_per_mode,
        "train_dataset": train_dataset,
        "heldout_dataset": heldout_dataset,
        "train_pairs": len(train),
        "heldout_pairs": len(heldout),
        "train_by_failure_mode": counts(train, "failure_mode"),
        "heldout_by_failure_mode": counts(heldout, "failure_mode"),
        "train_by_chosen_action": counts(train, "evidence_derived_action"),
        "heldout_by_chosen_action": counts(heldout, "evidence_derived_action"),
        "train_by_rejected_action": counts(train, "rejected_action"),
        "heldout_by_rejected_action": counts(heldout, "rejected_action"),
        "train_source_ids": sorted(train_source_ids),
        "heldout_source_ids": sorted(heldout_source_ids),
        "overlap_source_ids": sorted(train_source_ids & heldout_source_ids),
        "boundary": (
            "Deterministic stratified split of the hard boundary preference pairs. "
            "Held-out pairs are reserved for margin diagnostics, not for training."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        default="post_training/negbiodb_ct_oracle_boundary_preferences_hard_v1.jsonl",
    )
    parser.add_argument(
        "--train-out",
        default="post_training/negbiodb_ct_oracle_boundary_preferences_hard_train_v1.jsonl",
    )
    parser.add_argument(
        "--heldout-out",
        default="post_training/negbiodb_ct_oracle_boundary_preferences_hard_heldout_v1.jsonl",
    )
    parser.add_argument(
        "--manifest-out",
        default="post_training/negbiodb_ct_oracle_boundary_preferences_hard_split_manifest.json",
    )
    parser.add_argument("--heldout-per-mode", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260627)
    parser.add_argument("--train-dataset", default=DEFAULT_TRAIN_DATASET)
    parser.add_argument("--heldout-dataset", default=DEFAULT_HELDOUT_DATASET)
    args = parser.parse_args()

    rows = load_jsonl(args.source)
    train, heldout = split_by_failure_mode(
        rows,
        heldout_per_mode=args.heldout_per_mode,
        seed=args.seed,
        train_dataset=args.train_dataset,
        heldout_dataset=args.heldout_dataset,
    )
    manifest = manifest_for_split(
        source=args.source,
        train_out=args.train_out,
        heldout_out=args.heldout_out,
        seed=args.seed,
        heldout_per_mode=args.heldout_per_mode,
        train=train,
        heldout=heldout,
        train_dataset=args.train_dataset,
        heldout_dataset=args.heldout_dataset,
    )
    if manifest["overlap_source_ids"]:
        raise ValueError(f"Train/heldout source overlap: {manifest['overlap_source_ids'][:5]}")
    write_jsonl(args.train_out, train)
    write_jsonl(args.heldout_out, heldout)
    Path(args.manifest_out).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

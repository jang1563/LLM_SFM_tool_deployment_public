#!/usr/bin/env python3
"""Create deterministic train/held-out splits for Stage A artifacts."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

STAGE_A_SPLIT_DATASET = "negbiodb_ct_stage_a_split_v1"
DEFAULT_SEED = 20260704


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open() as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def split_case_ids(
    sft_rows: Sequence[Mapping[str, Any]],
    *,
    heldout_per_family: int,
    seed: int,
) -> tuple[list[str], list[str]]:
    if heldout_per_family <= 0:
        raise ValueError("heldout_per_family must be positive")

    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    seen_case_ids: set[str] = set()
    for row in sft_rows:
        case_id = str(row.get("source_manifest_case_id") or "")
        case_family = str(row.get("case_family") or "")
        if not case_id:
            raise ValueError(f"Missing source_manifest_case_id in row: {row.get('id')}")
        if not case_family:
            raise ValueError(f"Missing case_family in row: {row.get('id')}")
        if case_id in seen_case_ids:
            raise ValueError(f"Duplicate Stage A SFT case ID: {case_id}")
        seen_case_ids.add(case_id)
        groups[case_family].append(row)

    heldout_ids: set[str] = set()
    for case_family, rows in sorted(groups.items()):
        if len(rows) <= heldout_per_family:
            raise ValueError(
                f"Case family {case_family!r} has {len(rows)} rows, cannot hold out {heldout_per_family}"
            )
        shuffled = list(rows)
        random.Random(f"{seed}:{case_family}").shuffle(shuffled)
        heldout_ids.update(str(row["source_manifest_case_id"]) for row in shuffled[:heldout_per_family])

    train_ids = [str(row["source_manifest_case_id"]) for row in sft_rows if row["source_manifest_case_id"] not in heldout_ids]
    heldout_ordered = [str(row["source_manifest_case_id"]) for row in sft_rows if row["source_manifest_case_id"] in heldout_ids]
    return train_ids, heldout_ordered


def subset_with_split(
    rows: Sequence[Mapping[str, Any]],
    case_ids: set[str],
    *,
    split: str,
) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        if str(row.get("source_manifest_case_id")) in case_ids:
            copied = dict(row)
            copied["split"] = split
            out.append(copied)
    return out


def build_stage_a_split(
    sft_rows: Sequence[Mapping[str, Any]],
    preference_rows: Sequence[Mapping[str, Any]],
    process_rows: Sequence[Mapping[str, Any]],
    *,
    heldout_per_family: int,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    train_case_ids, heldout_case_ids = split_case_ids(
        sft_rows,
        heldout_per_family=heldout_per_family,
        seed=seed,
    )
    train_cases = set(train_case_ids)
    heldout_cases = set(heldout_case_ids)
    return {
        "train_sft": subset_with_split(sft_rows, train_cases, split="train"),
        "heldout_sft": subset_with_split(sft_rows, heldout_cases, split="heldout"),
        "train_preferences": subset_with_split(preference_rows, train_cases, split="train"),
        "heldout_preferences": subset_with_split(preference_rows, heldout_cases, split="heldout"),
        "train_process": subset_with_split(process_rows, train_cases, split="train"),
        "heldout_process": subset_with_split(process_rows, heldout_cases, split="heldout"),
    }


def manifest_for_stage_a_split(
    *,
    source_export_manifest: str | Path,
    source_sft: str | Path,
    source_preferences: str | Path,
    source_process: str | Path,
    train_sft_path: str | Path,
    heldout_sft_path: str | Path,
    train_preferences_path: str | Path,
    heldout_preferences_path: str | Path,
    train_process_path: str | Path,
    heldout_process_path: str | Path,
    splits: Mapping[str, Sequence[Mapping[str, Any]]],
    seed: int,
    heldout_per_family: int,
) -> dict[str, Any]:
    train_sft = list(splits["train_sft"])
    heldout_sft = list(splits["heldout_sft"])
    train_preferences = list(splits["train_preferences"])
    heldout_preferences = list(splits["heldout_preferences"])
    train_process = list(splits["train_process"])
    heldout_process = list(splits["heldout_process"])

    train_case_ids = sorted(case_ids(train_sft))
    heldout_case_ids = sorted(case_ids(heldout_sft))
    train_split_groups = sorted(split_groups(train_sft))
    heldout_split_groups = sorted(split_groups(heldout_sft))
    train_source_task_ids = sorted(source_task_ids(train_sft))
    heldout_source_task_ids = sorted(source_task_ids(heldout_sft))

    return {
        "dataset": STAGE_A_SPLIT_DATASET,
        "source_export_manifest": str(source_export_manifest),
        "source_sft": str(source_sft),
        "source_preferences": str(source_preferences),
        "source_process": str(source_process),
        "train_sft": str(train_sft_path),
        "heldout_sft": str(heldout_sft_path),
        "train_preferences": str(train_preferences_path),
        "heldout_preferences": str(heldout_preferences_path),
        "train_process": str(train_process_path),
        "heldout_process": str(heldout_process_path),
        "split_unit": "source_manifest_case_id",
        "seed": seed,
        "heldout_per_family": heldout_per_family,
        "train_cases": len(train_case_ids),
        "heldout_cases": len(heldout_case_ids),
        "train_sft_examples": len(train_sft),
        "heldout_sft_examples": len(heldout_sft),
        "train_preference_pairs": len(train_preferences),
        "heldout_preference_pairs": len(heldout_preferences),
        "train_process_examples": len(train_process),
        "heldout_process_examples": len(heldout_process),
        "train_by_case_family": count_by(train_sft, "case_family"),
        "heldout_by_case_family": count_by(heldout_sft, "case_family"),
        "train_by_evidence_status": count_by(train_sft, "gold_evidence_status"),
        "heldout_by_evidence_status": count_by(heldout_sft, "gold_evidence_status"),
        "train_preference_failure_modes": count_by(train_preferences, "failure_mode"),
        "heldout_preference_failure_modes": count_by(heldout_preferences, "failure_mode"),
        "train_case_ids": train_case_ids,
        "heldout_case_ids": heldout_case_ids,
        "train_split_groups": train_split_groups,
        "heldout_split_groups": heldout_split_groups,
        "overlap_case_ids": sorted(set(train_case_ids) & set(heldout_case_ids)),
        "overlap_split_groups": sorted(set(train_split_groups) & set(heldout_split_groups)),
        "train_source_task_ids": train_source_task_ids,
        "heldout_source_task_ids": heldout_source_task_ids,
        "overlap_source_task_ids": sorted(set(train_source_task_ids) & set(heldout_source_task_ids)),
        "boundary": (
            "Deterministic Stage A split by source_manifest_case_id; no train/eval "
            "split_group or source_task_id overlap is allowed before training."
        ),
    }


def case_ids(rows: Sequence[Mapping[str, Any]]) -> set[str]:
    return {str(row.get("source_manifest_case_id")) for row in rows if row.get("source_manifest_case_id")}


def split_groups(rows: Sequence[Mapping[str, Any]]) -> set[str]:
    return {str(row.get("split_group")) for row in rows if row.get("split_group")}


def source_task_ids(rows: Sequence[Mapping[str, Any]]) -> set[str]:
    return {str(row.get("source_task_id")) for row in rows if row.get("source_task_id")}


def count_by(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(key)) for row in rows).items()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft", default="post_training/stage_a_sft_v1.jsonl")
    parser.add_argument("--preferences", default="post_training/stage_a_preferences_v1.jsonl")
    parser.add_argument("--process", default="post_training/stage_a_process_supervision_v1.jsonl")
    parser.add_argument("--source-export-manifest", default="post_training/stage_a_export_manifest.json")
    parser.add_argument("--train-sft-out", default="post_training/stage_a_sft_train_v1.jsonl")
    parser.add_argument("--heldout-sft-out", default="post_training/stage_a_sft_heldout_v1.jsonl")
    parser.add_argument("--train-preferences-out", default="post_training/stage_a_preferences_train_v1.jsonl")
    parser.add_argument("--heldout-preferences-out", default="post_training/stage_a_preferences_heldout_v1.jsonl")
    parser.add_argument("--train-process-out", default="post_training/stage_a_process_train_v1.jsonl")
    parser.add_argument("--heldout-process-out", default="post_training/stage_a_process_heldout_v1.jsonl")
    parser.add_argument("--manifest-out", default="post_training/stage_a_split_manifest.json")
    parser.add_argument("--heldout-per-family", type=int, default=1)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    sft_rows = load_jsonl(args.sft)
    preference_rows = load_jsonl(args.preferences)
    process_rows = load_jsonl(args.process)
    splits = build_stage_a_split(
        sft_rows,
        preference_rows,
        process_rows,
        heldout_per_family=args.heldout_per_family,
        seed=args.seed,
    )

    write_jsonl(args.train_sft_out, splits["train_sft"])
    write_jsonl(args.heldout_sft_out, splits["heldout_sft"])
    write_jsonl(args.train_preferences_out, splits["train_preferences"])
    write_jsonl(args.heldout_preferences_out, splits["heldout_preferences"])
    write_jsonl(args.train_process_out, splits["train_process"])
    write_jsonl(args.heldout_process_out, splits["heldout_process"])

    manifest = manifest_for_stage_a_split(
        source_export_manifest=args.source_export_manifest,
        source_sft=args.sft,
        source_preferences=args.preferences,
        source_process=args.process,
        train_sft_path=args.train_sft_out,
        heldout_sft_path=args.heldout_sft_out,
        train_preferences_path=args.train_preferences_out,
        heldout_preferences_path=args.heldout_preferences_out,
        train_process_path=args.train_process_out,
        heldout_process_path=args.heldout_process_out,
        splits=splits,
        seed=args.seed,
        heldout_per_family=args.heldout_per_family,
    )
    write_json(args.manifest_out, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))

    if manifest["overlap_case_ids"] or manifest["overlap_split_groups"] or manifest["overlap_source_task_ids"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

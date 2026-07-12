#!/usr/bin/env python3
"""Build contrast-family curriculum SFT artifacts after pressure-run analysis."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from post_training.split_sft_data import by_action_class, load_jsonl, write_jsonl  # noqa: E402


FAMILY_CLASS_ORDERS = {
    "ground_flag": ["ground", "flag"],
    "verify_defer": ["defer", "verify"],
    "reject_override": ["ground", "flag", "reject"],
}


def curriculum_copy(
    row: Mapping[str, Any],
    *,
    dataset: str,
    family: str,
    block_index: int,
    replicate_index: int,
    family_classes: list[str],
) -> dict[str, Any]:
    out = dict(row)
    source_id = str(row["id"])
    out["id"] = f"{source_id}::curriculum::{family}::{replicate_index}"
    out["dataset"] = dataset
    out["source_example_id"] = source_id
    out["curriculum_strategy"] = "contrast_family_interleave"
    out["curriculum_family"] = family
    out["curriculum_family_classes"] = list(family_classes)
    out["curriculum_block_index"] = block_index
    out["curriculum_replicate_index"] = replicate_index
    return out


def class_counts(rows: list[Mapping[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(row["action_class"]) for row in rows).items()))


def family_counts(rows: list[Mapping[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get("curriculum_family", "source")) for row in rows).items()))


def interleave_family(
    groups: Mapping[str, list[Mapping[str, Any]]],
    *,
    class_order: list[str],
    family: str,
    dataset: str,
    block_index: int,
) -> list[dict[str, Any]]:
    missing = [label for label in class_order if label not in groups]
    if missing:
        raise ValueError(f"Missing classes for {family}: {missing}")
    target = min(len(groups[label]) for label in class_order)
    out = []
    for i in range(target):
        for label in class_order:
            out.append(
                curriculum_copy(
                    groups[label][i],
                    dataset=dataset,
                    family=family,
                    block_index=block_index,
                    replicate_index=i,
                    family_classes=class_order,
                )
            )
    return out


def build_curriculum_rows(
    rows: list[Mapping[str, Any]],
    *,
    dataset: str,
    family_orders: Mapping[str, list[str]] = FAMILY_CLASS_ORDERS,
) -> list[dict[str, Any]]:
    groups = by_action_class(rows)
    out = [
        curriculum_copy(
            row,
            dataset=dataset,
            family="base",
            block_index=0,
            replicate_index=i,
            family_classes=["defer", "flag", "ground", "reject", "verify"],
        )
        for i, row in enumerate(rows)
    ]
    for block_index, (family, class_order) in enumerate(family_orders.items(), start=1):
        out.extend(
            interleave_family(
                groups,
                class_order=class_order,
                family=family,
                dataset=dataset,
                block_index=block_index,
            )
        )
    return out


def build_native_cv_curriculum(
    manifest: Mapping[str, Any],
    *,
    out_dir: str | Path,
    prefix: str,
    dataset: str,
) -> dict[str, Any]:
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    fold_manifests = []
    for fold in manifest["fold_manifests"]:
        train_rows = load_jsonl(fold["train"])
        curriculum_rows = build_curriculum_rows(train_rows, dataset=dataset)
        train_path = out_root / f"{prefix}_fold{fold['fold']}_train.jsonl"
        write_jsonl(train_path, curriculum_rows)
        fold_manifests.append({
            "fold": fold["fold"],
            "source_train": fold["train"],
            "train": str(train_path),
            "heldout": fold["heldout"],
            "source_train_examples": len(train_rows),
            "train_examples": len(curriculum_rows),
            "source_train_by_class": class_counts(train_rows),
            "train_by_class": class_counts(curriculum_rows),
            "train_by_family": family_counts(curriculum_rows),
            "heldout_examples": fold["heldout_examples"],
            "heldout_by_class": fold["heldout_by_class"],
        })
    return {
        "dataset": dataset,
        "source_manifest": manifest.get("source"),
        "source_cv_manifest_folds": manifest.get("folds"),
        "strategy": "contrast_family_interleave",
        "family_class_orders": FAMILY_CLASS_ORDERS,
        "fold_manifests": fold_manifests,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv-manifest", default="post_training/negbiodb_ct_native_sft_cv4_manifest.json")
    parser.add_argument("--out-dir", default="post_training/curriculum")
    parser.add_argument("--prefix", default="negbiodb_ct_native_sft_cv4_curriculum_v1")
    parser.add_argument("--manifest-out", default="post_training/negbiodb_ct_native_sft_cv4_curriculum_manifest.json")
    parser.add_argument("--dataset", default="negbiodb_ct_native_sft_curriculum_v1")
    args = parser.parse_args()

    cv_manifest = json.loads(Path(args.cv_manifest).read_text())
    curriculum_manifest = build_native_cv_curriculum(
        cv_manifest,
        out_dir=args.out_dir,
        prefix=args.prefix,
        dataset=args.dataset,
    )
    curriculum_manifest["source_cv_manifest"] = args.cv_manifest
    curriculum_manifest["out_dir"] = args.out_dir
    curriculum_manifest["prefix"] = args.prefix
    Path(args.manifest_out).write_text(json.dumps(curriculum_manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(curriculum_manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

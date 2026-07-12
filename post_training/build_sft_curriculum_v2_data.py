#!/usr/bin/env python3
"""Build targeted curriculum-v2 SFT artifacts from persistent failure rows."""

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

from post_training.build_sft_curriculum_data import (  # noqa: E402
    build_curriculum_rows,
    class_counts,
    curriculum_copy,
    family_counts,
)
from post_training.split_sft_data import load_jsonl, write_jsonl  # noqa: E402


TARGET_WEIGHTS = {
    "defer": 2,
    "flag": 1,
    "ground": 2,
    "reject": 3,
    "verify": 2,
}


def load_persistent_failures(path: str | Path) -> dict[str, dict[str, Any]]:
    data = json.loads(Path(path).read_text())
    failures = {}
    for item in data["curriculum_persistent_failures"]:
        packet_id = str(item["packet_id"])
        first = item["failures"][0]
        failures[packet_id] = {
            "packet_id": packet_id,
            "gold": str(first["gold"]),
            "pred": str(first["pred"]),
            "failure_pair": f"{first['gold']}->{first['pred']}",
            "note": first.get("note", ""),
            "failure_conditions": int(item["failure_conditions"]),
            "failures": item["failures"],
        }
    return failures


def target_family(gold: str, pred: str) -> str:
    if gold in {"defer", "verify"} and pred in {"defer", "verify"}:
        return "target_defer_verify"
    if gold == "ground":
        return "target_clean_ground"
    if gold == "reject":
        return "target_reject_override"
    if gold == "flag":
        return "target_flag_preserve"
    return "target_other"


def targeted_copy(
    row: Mapping[str, Any],
    *,
    dataset: str,
    failure: Mapping[str, Any],
    replicate_index: int,
) -> dict[str, Any]:
    gold = str(failure["gold"])
    pred = str(failure["pred"])
    family = target_family(gold, pred)
    out = curriculum_copy(
        row,
        dataset=dataset,
        family=family,
        block_index=10,
        replicate_index=replicate_index,
        family_classes=[gold, pred],
    )
    out["id"] = f"{row['id']}::curriculum_v2::{family}::{replicate_index}"
    out["curriculum_strategy"] = "persistent_failure_targeted_v2"
    out["curriculum_source"] = "sft_curriculum_failure_analysis_2026-06-26"
    out["curriculum_failure_pair"] = str(failure["failure_pair"])
    out["curriculum_failure_conditions"] = int(failure["failure_conditions"])
    out["curriculum_failure_note"] = str(failure.get("note", ""))
    return out


def build_targeted_rows(
    train_rows: list[Mapping[str, Any]],
    *,
    dataset: str,
    persistent_failures: Mapping[str, Mapping[str, Any]],
    target_weights: Mapping[str, int] = TARGET_WEIGHTS,
) -> list[dict[str, Any]]:
    out = []
    for row in train_rows:
        task_id = str(row["task_id"])
        failure = persistent_failures.get(task_id)
        if failure is None:
            continue
        weight = int(target_weights[str(failure["gold"])])
        for replicate_index in range(weight):
            out.append(
                targeted_copy(
                    row,
                    dataset=dataset,
                    failure=failure,
                    replicate_index=replicate_index,
                )
            )
    return out


def task_ids(rows: list[Mapping[str, Any]]) -> set[str]:
    return {str(row["task_id"]) for row in rows}


def target_failure_counts(rows: list[Mapping[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get("curriculum_failure_pair", "")) for row in rows).items()))


def build_native_cv_curriculum_v2(
    manifest: Mapping[str, Any],
    *,
    failure_analysis: str | Path,
    out_dir: str | Path,
    prefix: str,
    dataset: str,
) -> dict[str, Any]:
    persistent_failures = load_persistent_failures(failure_analysis)
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    fold_manifests = []
    for fold in manifest["fold_manifests"]:
        train_rows = load_jsonl(fold["train"])
        heldout_rows = load_jsonl(fold["heldout"])
        heldout_ids = task_ids(heldout_rows)
        v1_rows = build_curriculum_rows(train_rows, dataset=dataset)
        targeted_rows = build_targeted_rows(
            train_rows,
            dataset=dataset,
            persistent_failures=persistent_failures,
        )
        leaked = sorted(str(row["task_id"]) for row in targeted_rows if str(row["task_id"]) in heldout_ids)
        if leaked:
            raise ValueError(f"Targeted rows leak held-out task IDs in fold {fold['fold']}: {leaked}")
        curriculum_rows = v1_rows + targeted_rows
        train_path = out_root / f"{prefix}_fold{fold['fold']}_train.jsonl"
        write_jsonl(train_path, curriculum_rows)
        train_ids = task_ids(train_rows)
        fold_manifests.append({
            "fold": fold["fold"],
            "source_train": fold["train"],
            "train": str(train_path),
            "heldout": fold["heldout"],
            "source_train_examples": len(train_rows),
            "v1_curriculum_examples": len(v1_rows),
            "targeted_examples": len(targeted_rows),
            "train_examples": len(curriculum_rows),
            "source_train_by_class": class_counts(train_rows),
            "train_by_class": class_counts(curriculum_rows),
            "train_by_family": family_counts(curriculum_rows),
            "targeted_by_class": class_counts(targeted_rows),
            "targeted_by_family": family_counts(targeted_rows),
            "targeted_by_failure_pair": target_failure_counts(targeted_rows),
            "targeted_task_ids": sorted(task_ids(targeted_rows)),
            "persistent_failures_in_train": sorted(set(persistent_failures) & train_ids),
            "persistent_failures_heldout_excluded": sorted(set(persistent_failures) & heldout_ids),
            "heldout_examples": fold["heldout_examples"],
            "heldout_by_class": fold["heldout_by_class"],
        })
    return {
        "dataset": dataset,
        "source_manifest": manifest.get("source"),
        "source_cv_manifest_folds": manifest.get("folds"),
        "source_failure_analysis": str(failure_analysis),
        "strategy": "contrast_family_interleave_plus_persistent_failure_targeted_v2",
        "target_weights": TARGET_WEIGHTS,
        "persistent_failure_count": len(persistent_failures),
        "fold_manifests": fold_manifests,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv-manifest", default="post_training/negbiodb_ct_native_sft_cv4_manifest.json")
    parser.add_argument(
        "--failure-analysis",
        default="post_training/sft_curriculum_failure_analysis_2026-06-26.json",
    )
    parser.add_argument("--out-dir", default="post_training/curriculum_v2")
    parser.add_argument("--prefix", default="negbiodb_ct_native_sft_cv4_curriculum_v2_targeted")
    parser.add_argument(
        "--manifest-out",
        default="post_training/negbiodb_ct_native_sft_cv4_curriculum_v2_manifest.json",
    )
    parser.add_argument("--dataset", default="negbiodb_ct_native_sft_curriculum_v2_targeted")
    args = parser.parse_args()

    cv_manifest = json.loads(Path(args.cv_manifest).read_text())
    curriculum_manifest = build_native_cv_curriculum_v2(
        cv_manifest,
        failure_analysis=args.failure_analysis,
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

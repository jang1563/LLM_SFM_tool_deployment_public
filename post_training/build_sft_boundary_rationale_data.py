#!/usr/bin/env python3
"""Build paired boundary-rationale SFT artifacts without class oversampling."""

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

from post_training.split_sft_data import load_jsonl, write_jsonl  # noqa: E402


BOUNDARY_NEGATIVES = {
    "defer": ["verify"],
    "verify": ["defer"],
    "ground": ["flag", "reject"],
    "flag": ["ground", "reject"],
    "reject": ["ground", "flag"],
}


BOUNDARY_RATIONALES = {
    "defer": (
        "Choose defer when the returned target-indication rows do not contain a "
        "recorded efficacy failure and the other-indication check is also empty. "
        "Do not choose verify unless other-indication failures exist."
    ),
    "verify": (
        "Choose verify when the target indication lacks a valid efficacy failure "
        "but the other-indication check shows related failures. Do not choose defer "
        "when related failures exist."
    ),
    "ground": (
        "Choose ground when a clean target-indication efficacy failure is cited. "
        "Do not choose flag unless the cited row has an impossible value, and do "
        "not choose reject unless endpoint evidence is mixed."
    ),
    "flag": (
        "Choose flag when the decisive returned record has an impossible value. "
        "Do not choose ground for impossible evidence, and do not choose reject "
        "unless endpoint evidence is mixed."
    ),
    "reject": (
        "Choose reject when this drug-indication pair has mixed endpoint evidence. "
        "Mixed endpoint evidence overrides single-row ground or flag support."
    ),
}


def class_counts(rows: list[Mapping[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(row["action_class"]) for row in rows).items()))


def role_counts(rows: list[Mapping[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get("boundary_pair_role", "source")) for row in rows).items()))


def rationale_message(action: str) -> dict[str, str]:
    return {
        "role": "user",
        "content": (
            "BOUNDARY_RATIONALE: "
            + BOUNDARY_RATIONALES[action]
            + f" Correct final action: {action}."
        ),
    }


def base_copy(row: Mapping[str, Any], *, dataset: str, pair_index: int) -> dict[str, Any]:
    out = dict(row)
    out["id"] = f"{row['id']}::boundary_rationale::base::{pair_index}"
    out["dataset"] = dataset
    out["source_example_id"] = row["id"]
    out["boundary_strategy"] = "paired_boundary_rationale_v1"
    out["boundary_pair_role"] = "base"
    out["boundary_pair_index"] = pair_index
    return out


def rationale_copy(row: Mapping[str, Any], *, dataset: str, pair_index: int) -> dict[str, Any]:
    action = str(row["action_class"])
    messages = list(row["messages"])
    if messages[-1].get("tool_call", {}).get("name") != "submit_decision":
        raise ValueError(f"Expected final submit_decision for {row['id']}")
    out = dict(row)
    out["id"] = f"{row['id']}::boundary_rationale::rationale::{pair_index}"
    out["dataset"] = dataset
    out["source_example_id"] = row["id"]
    out["boundary_strategy"] = "paired_boundary_rationale_v1"
    out["boundary_pair_role"] = "rationale"
    out["boundary_pair_index"] = pair_index
    out["boundary_negative_actions"] = list(BOUNDARY_NEGATIVES[action])
    out["boundary_rationale"] = BOUNDARY_RATIONALES[action]
    out["messages"] = messages[:-1] + [rationale_message(action), messages[-1]]
    return out


def build_boundary_rows(rows: list[Mapping[str, Any]], *, dataset: str) -> list[dict[str, Any]]:
    out = []
    for pair_index, row in enumerate(rows):
        out.append(base_copy(row, dataset=dataset, pair_index=pair_index))
        out.append(rationale_copy(row, dataset=dataset, pair_index=pair_index))
    return out


def build_native_cv_boundary_rationale(
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
        boundary_rows = build_boundary_rows(train_rows, dataset=dataset)
        train_path = out_root / f"{prefix}_fold{fold['fold']}_train.jsonl"
        write_jsonl(train_path, boundary_rows)
        fold_manifests.append({
            "fold": fold["fold"],
            "source_train": fold["train"],
            "train": str(train_path),
            "heldout": fold["heldout"],
            "source_train_examples": len(train_rows),
            "train_examples": len(boundary_rows),
            "source_train_by_class": class_counts(train_rows),
            "train_by_class": class_counts(boundary_rows),
            "train_by_role": role_counts(boundary_rows),
            "heldout_examples": fold["heldout_examples"],
            "heldout_by_class": fold["heldout_by_class"],
        })
    return {
        "dataset": dataset,
        "source_manifest": manifest.get("source"),
        "source_cv_manifest_folds": manifest.get("folds"),
        "strategy": "paired_boundary_rationale_v1",
        "boundary_negative_actions": BOUNDARY_NEGATIVES,
        "fold_manifests": fold_manifests,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv-manifest", default="post_training/negbiodb_ct_native_sft_cv4_manifest.json")
    parser.add_argument("--out-dir", default="post_training/boundary_rationale")
    parser.add_argument("--prefix", default="negbiodb_ct_native_sft_cv4_boundary_rationale_v1")
    parser.add_argument(
        "--manifest-out",
        default="post_training/negbiodb_ct_native_sft_cv4_boundary_rationale_manifest.json",
    )
    parser.add_argument("--dataset", default="negbiodb_ct_native_sft_boundary_rationale_v1")
    args = parser.parse_args()

    cv_manifest = json.loads(Path(args.cv_manifest).read_text())
    boundary_manifest = build_native_cv_boundary_rationale(
        cv_manifest,
        out_dir=args.out_dir,
        prefix=args.prefix,
        dataset=args.dataset,
    )
    boundary_manifest["source_cv_manifest"] = args.cv_manifest
    boundary_manifest["out_dir"] = args.out_dir
    boundary_manifest["prefix"] = args.prefix
    Path(args.manifest_out).write_text(json.dumps(boundary_manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(boundary_manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build the negative-margin hard-mode subset of boundary preference pairs."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from post_training.split_sft_data import load_jsonl, write_jsonl  # noqa: E402


DEFAULT_HARD_MODES = (
    "boundary_defer_over_verify",
    "boundary_flag_over_ground",
    "boundary_reject_over_ground",
    "boundary_reject_over_flag",
)
DEFAULT_DATASET = "negbiodb_ct_oracle_boundary_preferences_hard_v1"
DEFAULT_STRATEGY = "base_negative_margin_hard_modes_v1"


def parse_modes(value: str | None) -> tuple[str, ...]:
    if value is None or not value.strip():
        return DEFAULT_HARD_MODES
    modes = tuple(part.strip() for part in value.split(",") if part.strip())
    if not modes:
        raise ValueError("At least one failure mode is required.")
    return modes


def select_hard_pairs(
    rows: Sequence[Mapping[str, Any]],
    *,
    modes: Sequence[str] = DEFAULT_HARD_MODES,
    dataset: str = DEFAULT_DATASET,
    strategy: str = DEFAULT_STRATEGY,
) -> list[dict[str, Any]]:
    mode_set = set(modes)
    selected: list[dict[str, Any]] = []
    for row in rows:
        failure_mode = str(row.get("failure_mode"))
        if failure_mode not in mode_set:
            continue
        out = dict(row)
        out["source_preference_id"] = row.get("id")
        out["source_dataset"] = row.get("dataset")
        out["id"] = f"prefhard::{row['task_id']}::{failure_mode}::{len(selected)}"
        out["dataset"] = dataset
        out["strategy"] = strategy
        out["hard_mode_selection"] = "base_margin_negative_or_near_zero_v1"
        selected.append(out)
    return selected


def counts(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(key)) for row in rows).items()))


def manifest_for_hard_pairs(
    *,
    source: str | Path,
    out: str | Path,
    dataset: str,
    strategy: str,
    modes: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
    selected: Sequence[Mapping[str, Any]],
    selection_source: str | Path,
) -> dict[str, Any]:
    missing_modes = [mode for mode in modes if mode not in counts(selected, "failure_mode")]
    return {
        "source": str(source),
        "out": str(out),
        "dataset": dataset,
        "strategy": strategy,
        "selection_source": str(selection_source),
        "selection_rule": (
            "Failure modes with negative or near-zero base-model chosen margins "
            "in the 2026-06-27 boundary preference base-margin diagnostic."
        ),
        "selected_failure_modes": list(modes),
        "missing_selected_failure_modes": missing_modes,
        "source_preference_pairs": len(rows),
        "preference_pairs": len(selected),
        "pairs_by_failure_mode": counts(selected, "failure_mode"),
        "pairs_by_chosen_action": counts(selected, "evidence_derived_action"),
        "pairs_by_rejected_action": counts(selected, "rejected_action"),
        "chosen_passed": sum(bool(row.get("chosen_score", {}).get("passed")) for row in selected),
        "rejected_passed": sum(bool(row.get("rejected_score", {}).get("passed")) for row in selected),
        "boundary": (
            "Hard-mode subset of evidence-derived boundary preference pairs. "
            "Chosen and rejected responses share the same visible native CT tool "
            "observations and differ only in terminal submit_decision action."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        default="post_training/negbiodb_ct_oracle_boundary_preferences_v1.jsonl",
    )
    parser.add_argument(
        "--out",
        default="post_training/negbiodb_ct_oracle_boundary_preferences_hard_v1.jsonl",
    )
    parser.add_argument(
        "--manifest-out",
        default="post_training/negbiodb_ct_oracle_boundary_preferences_hard_manifest.json",
    )
    parser.add_argument(
        "--selection-source",
        default="post_training/boundary_preference_margin_base_summary_2026-06-27.json",
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--strategy", default=DEFAULT_STRATEGY)
    parser.add_argument(
        "--modes",
        default=",".join(DEFAULT_HARD_MODES),
        help="Comma-separated failure modes to keep.",
    )
    args = parser.parse_args()

    modes = parse_modes(args.modes)
    rows = load_jsonl(args.source)
    selected = select_hard_pairs(
        rows,
        modes=modes,
        dataset=args.dataset,
        strategy=args.strategy,
    )
    manifest = manifest_for_hard_pairs(
        source=args.source,
        out=args.out,
        dataset=args.dataset,
        strategy=args.strategy,
        modes=modes,
        rows=rows,
        selected=selected,
        selection_source=args.selection_source,
    )
    if manifest["missing_selected_failure_modes"]:
        raise ValueError(f"Missing selected failure modes: {manifest['missing_selected_failure_modes']}")

    write_jsonl(args.out, selected)
    Path(args.manifest_out).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

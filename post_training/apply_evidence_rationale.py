#!/usr/bin/env python3
"""Apply the evidence-derived boundary rationale layer to native CT SFT JSONL."""

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

from post_training.evidence_rationale import evidence_rationale_copy  # noqa: E402
from post_training.split_sft_data import load_jsonl, write_jsonl  # noqa: E402


DEFAULT_STRATEGY = "deployable_evidence_boundary_rationale_v1"


def counts(rows: list[Mapping[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(key)) for row in rows).items()))


def build_evidence_rationale_rows(
    rows: list[Mapping[str, Any]],
    *,
    dataset: str,
    strategy: str = DEFAULT_STRATEGY,
    action_hint_label: str = "Evidence-derived final action",
) -> list[dict[str, Any]]:
    return [
        evidence_rationale_copy(
            row,
            dataset=dataset,
            pair_index=pair_index,
            strategy=strategy,
            action_hint_label=action_hint_label,
        )
        for pair_index, row in enumerate(rows)
    ]


def manifest_for_rows(
    *,
    source: str | Path,
    out: str | Path,
    dataset: str,
    strategy: str,
    rows: list[Mapping[str, Any]],
) -> dict[str, Any]:
    matches = [row.get("evidence_matches_action_class") for row in rows]
    return {
        "source": str(source),
        "out": str(out),
        "dataset": dataset,
        "strategy": strategy,
        "examples": len(rows),
        "by_action_class": counts(rows, "action_class"),
        "by_evidence_action": counts(rows, "evidence_derived_action"),
        "by_role": counts(rows, "boundary_pair_role"),
        "evidence_action_matches": sum(match is True for match in matches),
        "evidence_action_mismatches": sum(match is False for match in matches),
        "evidence_action_unlabeled": sum(match is None for match in matches),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft", default="post_training/negbiodb_ct_native_sft_v1.jsonl")
    parser.add_argument("--out", default="post_training/negbiodb_ct_native_sft_evidence_rationale_v1.jsonl")
    parser.add_argument(
        "--manifest-out",
        default="post_training/negbiodb_ct_native_sft_evidence_rationale_manifest.json",
    )
    parser.add_argument("--dataset", default="negbiodb_ct_native_sft_evidence_rationale_v1")
    parser.add_argument("--strategy", default=DEFAULT_STRATEGY)
    parser.add_argument("--action-hint-label", default="Evidence-derived final action")
    args = parser.parse_args()

    source_rows = load_jsonl(args.sft)
    rows = build_evidence_rationale_rows(
        source_rows,
        dataset=args.dataset,
        strategy=args.strategy,
        action_hint_label=args.action_hint_label,
    )
    write_jsonl(args.out, rows)
    manifest = manifest_for_rows(
        source=args.sft,
        out=args.out,
        dataset=args.dataset,
        strategy=args.strategy,
        rows=rows,
    )
    Path(args.manifest_out).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Export enum-action corrective contrast pairs for Stage A.

The Cayuga enum-action diagnostics show a persistent `ground` / `supported`
collapse after schema validity and candidate-space pruning are fixed. This
exporter creates a small component-specific paired artifact:

- chosen: the strict enum/action oracle target;
- rejected: the observed collapse target, `ground` / `supported`.

Rows are derived only from the existing strict component target split. Grounded
cases where chosen and rejected would be identical are skipped. This is a
corrective data substrate, not a DPO/RLVR result.
"""

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

from post_training.export_stage_a_strict_component_targets import (  # noqa: E402
    DATASET as COMPONENT_TARGET_DATASET,
)
from post_training.run_stage_a_strict_component_sft_smoke import (  # noqa: E402
    build_component_eval_report,
    component_case_id,
    filter_component,
    load_jsonl,
    target_output_from_row,
    write_json,
    write_jsonl,
)


DATASET = "negbiodb_ct_stage_a_enum_corrective_pairs_v1"
MANIFEST_DATASET = "negbiodb_ct_stage_a_enum_corrective_pairs_manifest_v1"
PROMPT_CONTRACT = "stage_a_v2_strict"
COMPONENT = "enum_action"
FAILURE_MODE = "ground_supported_collapse"
COLLAPSE_OUTPUT = {"action": "ground", "evidence_status": "supported"}


def pair_key(output: Mapping[str, Any]) -> str:
    return f"{output.get('action')}/{output.get('evidence_status')}"


def final_json_message(output: Mapping[str, Any]) -> list[dict[str, str]]:
    return [{"role": "assistant", "content": json.dumps(dict(output), sort_keys=True)}]


def score_output(row: Mapping[str, Any], output: Mapping[str, Any], *, run_id: str) -> dict[str, Any]:
    report = build_component_eval_report(
        expected_rows=[row],
        prediction_rows=[
            {
                "id": f"{run_id}::{row['id']}",
                "source_component_target_id": row["id"],
                "split": row.get("split"),
                "prediction": dict(output),
            }
        ],
        component=COMPONENT,
        run_id=run_id,
    )
    return dict(report["rows"][0])


def corrective_pair(row: Mapping[str, Any]) -> dict[str, Any] | None:
    target = target_output_from_row(row)
    if target == COLLAPSE_OUTPUT:
        return None
    row_id = str(row["id"])
    run_prefix = f"stage_a_enum_corrective::{row_id}"
    chosen_score = score_output(row, target, run_id=f"{run_prefix}::chosen")
    rejected_score = score_output(row, COLLAPSE_OUTPUT, run_id=f"{run_prefix}::rejected")
    return {
        "id": f"{run_prefix}::{FAILURE_MODE}",
        "dataset": DATASET,
        "component": COMPONENT,
        "failure_mode": FAILURE_MODE,
        "candidate_policy": "train_observed_pairs",
        "prompt_contract": PROMPT_CONTRACT,
        "source_component_dataset": COMPONENT_TARGET_DATASET,
        "source_component_target_id": row_id,
        "source_manifest_case_id": row.get("source_manifest_case_id"),
        "source_task_id": row.get("source_task_id"),
        "split_group": row.get("split_group"),
        "case_family": row.get("case_family"),
        "gold_evidence_status": row.get("gold_evidence_status"),
        "expected_terminal_action": row.get("expected_terminal_action"),
        "target_keys": ["action", "evidence_status"],
        "prompt_messages": row.get("prompt_messages"),
        "chosen_output": target,
        "rejected_output": dict(COLLAPSE_OUTPUT),
        "chosen_pair": pair_key(target),
        "rejected_pair": pair_key(COLLAPSE_OUTPUT),
        "chosen_messages": final_json_message(target),
        "rejected_messages": final_json_message(COLLAPSE_OUTPUT),
        "chosen_score": chosen_score,
        "rejected_score": rejected_score,
        "split": row.get("split"),
        "oracle_target": True,
    }


def build_pairs(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in filter_component(rows, COMPONENT):
        pair = corrective_pair(row)
        if pair is not None:
            out.append(pair)
    return out


def case_ids(rows: Sequence[Mapping[str, Any]]) -> set[str]:
    return {str(row.get("source_manifest_case_id")) for row in rows if row.get("source_manifest_case_id")}


def value_set(rows: Sequence[Mapping[str, Any]], key: str) -> set[str]:
    return {str(row.get(key)) for row in rows if row.get(key)}


def count_by(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(key)) for row in rows).items()))


def display_path(path: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def manifest_for_pairs(
    *,
    source_targets: str | Path,
    source_train_targets: str | Path,
    source_heldout_targets: str | Path,
    pairs_out: str | Path,
    train_out: str | Path,
    heldout_out: str | Path,
    rows: Sequence[Mapping[str, Any]],
    train_rows: Sequence[Mapping[str, Any]],
    heldout_rows: Sequence[Mapping[str, Any]],
    skipped_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    train_case_ids = sorted(case_ids(train_rows))
    heldout_case_ids = sorted(case_ids(heldout_rows))
    train_split_groups = sorted(value_set(train_rows, "split_group"))
    heldout_split_groups = sorted(value_set(heldout_rows, "split_group"))
    train_source_task_ids = sorted(value_set(train_rows, "source_task_id"))
    heldout_source_task_ids = sorted(value_set(heldout_rows, "source_task_id"))
    return {
        "dataset": MANIFEST_DATASET,
        "pair_dataset": DATASET,
        "source_component_dataset": COMPONENT_TARGET_DATASET,
        "prompt_contract": PROMPT_CONTRACT,
        "component": COMPONENT,
        "failure_mode": FAILURE_MODE,
        "candidate_policy": "train_observed_pairs",
        "rejected_output": dict(COLLAPSE_OUTPUT),
        "source_targets": display_path(source_targets),
        "source_train_targets": display_path(source_train_targets),
        "source_heldout_targets": display_path(source_heldout_targets),
        "pairs": display_path(pairs_out),
        "train_pairs_path": display_path(train_out),
        "heldout_pairs_path": display_path(heldout_out),
        "pair_examples": len(rows),
        "train_pairs": len(train_rows),
        "heldout_pairs": len(heldout_rows),
        "skipped_examples": len(skipped_rows),
        "skipped_reason": "target_already_ground_supported",
        "skipped_case_ids": sorted(component_case_id(row) for row in skipped_rows),
        "by_chosen_pair": count_by(rows, "chosen_pair"),
        "train_by_chosen_pair": count_by(train_rows, "chosen_pair"),
        "heldout_by_chosen_pair": count_by(heldout_rows, "chosen_pair"),
        "by_case_family": count_by(rows, "case_family"),
        "train_case_ids": train_case_ids,
        "heldout_case_ids": heldout_case_ids,
        "overlap_case_ids": sorted(set(train_case_ids) & set(heldout_case_ids)),
        "train_split_groups": train_split_groups,
        "heldout_split_groups": heldout_split_groups,
        "overlap_split_groups": sorted(set(train_split_groups) & set(heldout_split_groups)),
        "train_source_task_ids": train_source_task_ids,
        "heldout_source_task_ids": heldout_source_task_ids,
        "overlap_source_task_ids": sorted(set(train_source_task_ids) & set(heldout_source_task_ids)),
        "boundary": (
            "Enum corrective pairs are component-specific contrast data against "
            "the observed ground/supported collapse. They are not DPO/RLVR "
            "results and do not use hidden held-out labels for training."
        ),
        "next_decision": (
            "Use train pairs for a small corrective enum-action experiment, "
            "then score against held-out pairs before moving to tool_query, "
            "DPO, or RLVR."
        ),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-targets", default="post_training/stage_a_strict_component_targets_v1.jsonl")
    parser.add_argument(
        "--source-train-targets",
        default="post_training/stage_a_strict_component_targets_train_v1.jsonl",
    )
    parser.add_argument(
        "--source-heldout-targets",
        default="post_training/stage_a_strict_component_targets_heldout_v1.jsonl",
    )
    parser.add_argument("--pairs-out", default="post_training/stage_a_enum_corrective_pairs_v1.jsonl")
    parser.add_argument("--train-out", default="post_training/stage_a_enum_corrective_pairs_train_v1.jsonl")
    parser.add_argument("--heldout-out", default="post_training/stage_a_enum_corrective_pairs_heldout_v1.jsonl")
    parser.add_argument("--manifest-out", default="post_training/stage_a_enum_corrective_pairs_manifest.json")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    source_rows = load_jsonl(args.source_targets)
    train_source_rows = load_jsonl(args.source_train_targets)
    heldout_source_rows = load_jsonl(args.source_heldout_targets)
    rows = build_pairs(source_rows)
    train_rows = build_pairs(train_source_rows)
    heldout_rows = build_pairs(heldout_source_rows)
    skipped_rows = [
        row
        for row in filter_component(source_rows, COMPONENT)
        if target_output_from_row(row) == COLLAPSE_OUTPUT
    ]
    manifest = manifest_for_pairs(
        source_targets=args.source_targets,
        source_train_targets=args.source_train_targets,
        source_heldout_targets=args.source_heldout_targets,
        pairs_out=args.pairs_out,
        train_out=args.train_out,
        heldout_out=args.heldout_out,
        rows=rows,
        train_rows=train_rows,
        heldout_rows=heldout_rows,
        skipped_rows=skipped_rows,
    )
    write_jsonl(args.pairs_out, rows)
    write_jsonl(args.train_out, train_rows)
    write_jsonl(args.heldout_out, heldout_rows)
    write_json(args.manifest_out, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    if (
        manifest["overlap_case_ids"]
        or manifest["overlap_split_groups"]
        or manifest["overlap_source_task_ids"]
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

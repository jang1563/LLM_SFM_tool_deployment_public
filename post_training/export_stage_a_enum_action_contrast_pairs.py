#!/usr/bin/env python3
"""Export Stage A enum-action same-status action contrast pairs.

The field-rank diagnostic shows the invalid-value held-out case has
`invalid_value` at evidence-status rank 2, while the `flag` action ranks 6.
This exporter creates a sharper enum-only contrast substrate:

- chosen: the strict enum/action oracle target;
- rejected: the same evidence_status with action forced to `ground`.

Rows are derived only from the existing strict component target split. Grounded
cases where the action is already `ground` are skipped. This is an action-field
diagnostic substrate, not DPO/RLVR and not a broad retraining claim.
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

from post_training.export_stage_a_enum_corrective_pairs import (  # noqa: E402
    COMPONENT,
    PROMPT_CONTRACT,
    count_by,
    display_path,
    final_json_message,
    pair_key,
    score_output,
    value_set,
)
from post_training.export_stage_a_strict_component_targets import (  # noqa: E402
    DATASET as COMPONENT_TARGET_DATASET,
)
from post_training.run_stage_a_strict_component_sft_smoke import (  # noqa: E402
    component_case_id,
    filter_component,
    load_jsonl,
    target_output_from_row,
    write_json,
    write_jsonl,
)


DATASET = "negbiodb_ct_stage_a_enum_action_contrast_pairs_v1"
MANIFEST_DATASET = "negbiodb_ct_stage_a_enum_action_contrast_pairs_manifest_v1"
FAILURE_MODE = "same_status_wrong_action_contrast"
CANDIDATE_POLICY = "same_status_action_contrast"
CONTRAST_AXIS = "action"
REJECTED_ACTION = "ground"
FIELD_RANK_SOURCE = "post_training/STAGE_A_COMPONENT_ENUM_ACTION_FIELD_RANK_CAYUGA_2026-07-05.md"


def case_ids(rows: Sequence[Mapping[str, Any]]) -> set[str]:
    return {str(row.get("source_manifest_case_id")) for row in rows if row.get("source_manifest_case_id")}


def action_contrast_rejected_output(target: Mapping[str, Any]) -> dict[str, str] | None:
    action = target.get("action")
    evidence_status = target.get("evidence_status")
    if action == REJECTED_ACTION or not isinstance(evidence_status, str):
        return None
    return {"action": REJECTED_ACTION, "evidence_status": evidence_status}


def action_contrast_pair(row: Mapping[str, Any]) -> dict[str, Any] | None:
    target = target_output_from_row(row)
    rejected = action_contrast_rejected_output(target)
    if rejected is None or rejected == target:
        return None

    row_id = str(row["id"])
    run_prefix = f"stage_a_enum_action_contrast::{row_id}"
    chosen_score = score_output(row, target, run_id=f"{run_prefix}::chosen")
    rejected_score = score_output(row, rejected, run_id=f"{run_prefix}::rejected")
    return {
        "id": f"{run_prefix}::{FAILURE_MODE}",
        "dataset": DATASET,
        "component": COMPONENT,
        "failure_mode": FAILURE_MODE,
        "contrast_axis": CONTRAST_AXIS,
        "candidate_policy": CANDIDATE_POLICY,
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
        "rejected_output": rejected,
        "chosen_pair": pair_key(target),
        "rejected_pair": pair_key(rejected),
        "chosen_messages": final_json_message(target),
        "rejected_messages": final_json_message(rejected),
        "chosen_score": chosen_score,
        "rejected_score": rejected_score,
        "split": row.get("split"),
        "oracle_target": True,
    }


def build_pairs(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in filter_component(rows, COMPONENT):
        pair = action_contrast_pair(row)
        if pair is not None:
            out.append(pair)
    return out


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
        "contrast_axis": CONTRAST_AXIS,
        "candidate_policy": CANDIDATE_POLICY,
        "rejected_action": REJECTED_ACTION,
        "rejected_evidence_status_policy": "same_as_chosen",
        "field_rank_source": FIELD_RANK_SOURCE,
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
        "skipped_reason": "target_action_already_ground",
        "skipped_case_ids": sorted(component_case_id(row) for row in skipped_rows),
        "by_chosen_pair": count_by(rows, "chosen_pair"),
        "by_rejected_pair": count_by(rows, "rejected_pair"),
        "by_case_family": count_by(rows, "case_family"),
        "train_by_chosen_pair": count_by(train_rows, "chosen_pair"),
        "heldout_by_chosen_pair": count_by(heldout_rows, "chosen_pair"),
        "train_case_ids": train_case_ids,
        "heldout_case_ids": heldout_case_ids,
        "overlap_case_ids": sorted(set(train_case_ids) & set(heldout_case_ids)),
        "train_split_groups": train_split_groups,
        "heldout_split_groups": heldout_split_groups,
        "overlap_split_groups": sorted(set(train_split_groups) & set(heldout_split_groups)),
        "train_source_task_ids": train_source_task_ids,
        "heldout_source_task_ids": heldout_source_task_ids,
        "overlap_source_task_ids": sorted(set(train_source_task_ids) & set(heldout_source_task_ids)),
        "priority": {
            "chosen_pair": "flag/invalid_value",
            "reason": "field-rank diagnostic found action rank 6 while status rank is 2",
        },
        "boundary": (
            "Enum action-contrast pairs preserve evidence_status and force only "
            "the rejected action to ground. They diagnose action-field learning, "
            "not explanation quality or DPO/RLVR reward quality."
        ),
        "next_decision": (
            "Run a Cayuga action-contrast margin smoke focused on "
            "flag/invalid_value before moving to tool_query, DPO, or RLVR."
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
    parser.add_argument("--pairs-out", default="post_training/stage_a_enum_action_contrast_pairs_v1.jsonl")
    parser.add_argument("--train-out", default="post_training/stage_a_enum_action_contrast_pairs_train_v1.jsonl")
    parser.add_argument(
        "--heldout-out",
        default="post_training/stage_a_enum_action_contrast_pairs_heldout_v1.jsonl",
    )
    parser.add_argument("--manifest-out", default="post_training/stage_a_enum_action_contrast_pairs_manifest.json")
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
        if action_contrast_rejected_output(target_output_from_row(row)) is None
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

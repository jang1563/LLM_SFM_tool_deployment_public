#!/usr/bin/env python3
"""Export Stage A routing action/status contrast pairs.

The evidence-conditioned Cayuga routing run fixed schema and citation shape but
still confused three held-out action/status families:

- `defer` / `insufficient` was routed as `reject` / `contradicted`;
- `verify` / `insufficient` was routed as `reject` / `contradicted`;
- `flag` / `invalid_value` was routed as `ground` / `supported`.

This exporter turns those observed confusions into a small public-safe contrast
substrate. It is diagnostic data for the routing slice, not a DPO/RLVR result.
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

from post_training.export_stage_a_evidence_conditioned_component_targets import (  # noqa: E402
    DATASET as COMPONENT_TARGET_DATASET,
    PROMPT_CONTRACT,
)
from post_training.export_stage_a_enum_corrective_pairs import (  # noqa: E402
    display_path,
    value_set,
)
from post_training.run_stage_a_strict_component_sft_smoke import (  # noqa: E402
    build_component_eval_report,
    component_case_id,
    filter_component,
    load_jsonl,
    target_output_from_row,
    visible_citation_ids,
    write_json,
    write_jsonl,
)


DATASET = "negbiodb_ct_stage_a_routing_action_status_contrast_pairs_v1"
MANIFEST_DATASET = "negbiodb_ct_stage_a_routing_action_status_contrast_pairs_manifest_v1"
COMPONENT = "routing_after_loop"
FAILURE_MODE = "routing_action_status_confusion"
CONTRAST_AXIS = "action_status"
CANDIDATE_POLICY = "observed_constrained_routing_confusion"
RESULT_SOURCE = "post_training/STAGE_A_EVIDENCE_ROUTING_OBSERVED_PAIR_CAYUGA_2026-07-08.md"
TARGET_KEYS = ["action", "evidence_status", "cited_source_ids"]

REJECTED_BY_CHOSEN_PAIR = {
    "defer/insufficient": {"action": "reject", "evidence_status": "contradicted"},
    "verify/insufficient": {"action": "reject", "evidence_status": "contradicted"},
    "flag/invalid_value": {"action": "ground", "evidence_status": "supported"},
}


def pair_key(output: Mapping[str, Any]) -> str:
    return f"{output.get('action')}/{output.get('evidence_status')}"


def count_by(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(key)) for row in rows).items()))


def case_ids(rows: Sequence[Mapping[str, Any]]) -> set[str]:
    return {str(row.get("source_manifest_case_id")) for row in rows if row.get("source_manifest_case_id")}


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


def rejected_output_for_row(row: Mapping[str, Any], target: Mapping[str, Any]) -> dict[str, Any] | None:
    rejected_base = REJECTED_BY_CHOSEN_PAIR.get(pair_key(target))
    if rejected_base is None:
        return None

    rejected = dict(rejected_base)
    if pair_key(rejected) == "ground/supported":
        rejected["cited_source_ids"] = visible_citation_ids(row)[:1]
    else:
        rejected["cited_source_ids"] = []
    return rejected


def routing_action_status_pair(row: Mapping[str, Any]) -> dict[str, Any] | None:
    target = target_output_from_row(row)
    rejected = rejected_output_for_row(row, target)
    if rejected is None or rejected == target:
        return None

    row_id = str(row["id"])
    run_prefix = f"stage_a_routing_action_status_contrast::{row_id}"
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
        "target_keys": list(TARGET_KEYS),
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
        pair = routing_action_status_pair(row)
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
        "result_source": RESULT_SOURCE,
        "rejected_by_chosen_pair": REJECTED_BY_CHOSEN_PAIR,
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
        "skipped_reason": "chosen_pair_not_failed_in_constrained_routing_readout",
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
        "boundary": (
            "Routing action/status contrast pairs are a diagnostic repair substrate "
            "for the evidence-conditioned routing slice. They are not DPO/RLVR "
            "training results, do not score explanation quality, and do not expose "
            "hidden labels in the prompt."
        ),
        "next_decision": (
            "Dry-run these pairs, then run a Cayuga routing margin smoke before "
            "moving to tool_query, DPO, or audited RLVR."
        ),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-targets",
        default="post_training/stage_a_evidence_conditioned_component_targets_v1.jsonl",
    )
    parser.add_argument(
        "--source-train-targets",
        default="post_training/stage_a_evidence_conditioned_component_targets_train_v1.jsonl",
    )
    parser.add_argument(
        "--source-heldout-targets",
        default="post_training/stage_a_evidence_conditioned_component_targets_heldout_v1.jsonl",
    )
    parser.add_argument(
        "--pairs-out",
        default="post_training/stage_a_routing_action_status_contrast_pairs_v1.jsonl",
    )
    parser.add_argument(
        "--train-out",
        default="post_training/stage_a_routing_action_status_contrast_pairs_train_v1.jsonl",
    )
    parser.add_argument(
        "--heldout-out",
        default="post_training/stage_a_routing_action_status_contrast_pairs_heldout_v1.jsonl",
    )
    parser.add_argument(
        "--manifest-out",
        default="post_training/stage_a_routing_action_status_contrast_pairs_manifest.json",
    )
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
        if rejected_output_for_row(row, target_output_from_row(row)) is None
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

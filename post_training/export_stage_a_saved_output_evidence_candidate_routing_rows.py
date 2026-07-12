#!/usr/bin/env python3
"""Export evidence-conditioned candidate-routing rows after saved-output failures.

The post-candidate-CE bridge showed that failed saved-output candidate policies
span multiple prompt-visible evidence reasons. This exporter turns the existing
evidence-conditioned routing rows into a small finite-candidate substrate while
preserving the original train/held-out split. Bridge failure cases are marked
as held-out evaluation focus rows, not training rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from post_training.evaluate_stage_a_routing_evidence_gate import (  # noqa: E402
    evidence_features,
    gate_output,
    pair_label,
    prompt_payload,
    target_output,
)
from post_training.run_stage_a_strict_component_sft_smoke import (  # noqa: E402
    filter_component,
    load_jsonl,
    write_json,
    write_jsonl,
)


DATASET = "negbiodb_ct_stage_a_saved_output_evidence_candidate_routing_v1"
MANIFEST_DATASET = (
    "negbiodb_ct_stage_a_saved_output_evidence_candidate_routing_manifest_v1"
)
PROMPT_CONTRACT = "stage_a_saved_output_evidence_candidate_routing_v1"
COMPONENT = "routing_after_loop"
SOURCE_COMPONENT_DATASET = "negbiodb_ct_stage_a_evidence_conditioned_component_targets_v1"
SOURCE_BRIDGE_DATASET = "negbiodb_ct_stage_a_saved_output_evidence_bridge_v1"
CANDIDATE_POLICY = "all_valid_action_status_pairs_conditioned_on_visible_evidence"
CANDIDATE_PAIRS = (
    "ground/supported",
    "reject/contradicted",
    "defer/insufficient",
    "verify/insufficient",
    "flag/invalid_value",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def display_path(path: str | Path) -> str:
    path = Path(path)
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def count_by(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(key)) for row in rows).items()))


def value_set(rows: Sequence[Mapping[str, Any]], key: str) -> list[str]:
    return sorted({str(row[key]) for row in rows if row.get(key) is not None})


def case_ids(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    return value_set(rows, "source_manifest_case_id")


def bridge_case_reasons(bridge: Mapping[str, Any]) -> dict[str, str]:
    summary = bridge.get("bridge_summary")
    if not isinstance(summary, Mapping):
        return {}
    reasons = summary.get("runtime_reasons_by_case")
    if not isinstance(reasons, Mapping):
        return {}
    return {str(case_id): str(reason) for case_id, reason in reasons.items()}


def source_rows_by_case(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    out: dict[str, Mapping[str, Any]] = {}
    for row in filter_component(list(rows), COMPONENT):
        case_id = row.get("source_manifest_case_id")
        if isinstance(case_id, str):
            out[case_id] = row
    return out


def candidate_outputs() -> list[dict[str, str]]:
    outputs = []
    for pair in CANDIDATE_PAIRS:
        action, status = pair.split("/", maxsplit=1)
        outputs.append({"pair": pair, "action": action, "evidence_status": status})
    return outputs


def candidate_task(row: Mapping[str, Any], *, features: Mapping[str, Any]) -> dict[str, Any]:
    payload = prompt_payload(row)
    return {
        "component": "saved_output_evidence_candidate_routing",
        "input_id": payload.get("input_id"),
        "claim": payload.get("claim"),
        "observed_tool_loop": payload.get("observed_tool_loop", []),
        "candidate_pairs": list(CANDIDATE_PAIRS),
        "visible_evidence_features": dict(features),
        "instruction": (
            "Select exactly one candidate pair using only the observed tool-loop "
            "state and visible evidence features."
        ),
    }


def final_json_message(output: Mapping[str, Any]) -> list[dict[str, str]]:
    return [{"role": "assistant", "content": json.dumps(dict(output), sort_keys=True)}]


def build_row(
    row: Mapping[str, Any],
    *,
    split: str,
    bridge_reasons_by_case: Mapping[str, str],
) -> dict[str, Any]:
    features = evidence_features(row)
    runtime_output, runtime_reason = gate_output(features)
    target = target_output(row)
    target_pair = pair_label(target)
    runtime_pair = pair_label(runtime_output)
    case_id = str(row.get("source_manifest_case_id"))
    bridge_focus = case_id in bridge_reasons_by_case
    target_payload = {
        "selected_pair": target_pair,
        "action": target.get("action"),
        "evidence_status": target.get("evidence_status"),
    }
    if target_pair not in CANDIDATE_PAIRS:
        raise ValueError(f"{case_id} target pair is not in candidate set: {target_pair}")
    if runtime_pair != target_pair:
        raise ValueError(f"{case_id} runtime gate pair does not match target pair")
    return {
        "id": f"stage_a_saved_output_evidence_candidate_routing::{case_id}",
        "dataset": DATASET,
        "component": COMPONENT,
        "prompt_contract": PROMPT_CONTRACT,
        "candidate_policy": CANDIDATE_POLICY,
        "source_component_dataset": SOURCE_COMPONENT_DATASET,
        "source_component_target_id": row.get("id"),
        "source_manifest_case_id": case_id,
        "source_task_id": row.get("source_task_id"),
        "split_group": row.get("split_group"),
        "case_family": row.get("case_family"),
        "split": split,
        "training_allowed": split == "train",
        "evaluation_only": split == "heldout",
        "bridge_focus_case": bridge_focus,
        "bridge_runtime_reason": bridge_reasons_by_case.get(case_id),
        "model_visible_task": candidate_task(row, features=features),
        "candidate_outputs": candidate_outputs(),
        "target_output": target_payload,
        "target_pair": target_pair,
        "target_index": list(CANDIDATE_PAIRS).index(target_pair),
        "target_messages": final_json_message(target_payload),
        "runtime_evidence_pair": runtime_pair,
        "runtime_evidence_reason": runtime_reason,
        "runtime_evidence_exact": runtime_pair == target_pair,
        "candidate_prediction_source": "finite_candidate_options_not_model_scores",
        "oracle_target": True,
    }


def build_rows(
    *,
    source_rows: Sequence[Mapping[str, Any]],
    split_rows: Sequence[Mapping[str, Any]],
    split: str,
    bridge_reasons_by_case: Mapping[str, str],
) -> list[dict[str, Any]]:
    source_by_case = source_rows_by_case(source_rows)
    split_case_ids = [
        str(row.get("source_manifest_case_id"))
        for row in filter_component(list(split_rows), COMPONENT)
    ]
    return [
        build_row(source_by_case[case_id], split=split, bridge_reasons_by_case=bridge_reasons_by_case)
        for case_id in split_case_ids
    ]


def manifest_for_rows(
    *,
    source_targets: str | Path,
    source_train_targets: str | Path,
    source_heldout_targets: str | Path,
    evidence_bridge: str | Path,
    rows_out: str | Path,
    train_out: str | Path,
    heldout_out: str | Path,
    manifest_out: str | Path,
    rows: Sequence[Mapping[str, Any]],
    train_rows: Sequence[Mapping[str, Any]],
    heldout_rows: Sequence[Mapping[str, Any]],
    bridge_reasons_by_case: Mapping[str, str],
) -> dict[str, Any]:
    train_case_ids = case_ids(train_rows)
    heldout_case_ids = case_ids(heldout_rows)
    train_split_groups = value_set(train_rows, "split_group")
    heldout_split_groups = value_set(heldout_rows, "split_group")
    train_source_task_ids = value_set(train_rows, "source_task_id")
    heldout_source_task_ids = value_set(heldout_rows, "source_task_id")
    bridge_focus_rows = [row for row in rows if row.get("bridge_focus_case")]
    return {
        "dataset": MANIFEST_DATASET,
        "row_dataset": DATASET,
        "component": COMPONENT,
        "prompt_contract": PROMPT_CONTRACT,
        "candidate_policy": CANDIDATE_POLICY,
        "source_component_dataset": SOURCE_COMPONENT_DATASET,
        "source_bridge_dataset": SOURCE_BRIDGE_DATASET,
        "source_targets": display_path(source_targets),
        "source_targets_sha256": sha256_file(source_targets),
        "source_train_targets": display_path(source_train_targets),
        "source_train_targets_sha256": sha256_file(source_train_targets),
        "source_heldout_targets": display_path(source_heldout_targets),
        "source_heldout_targets_sha256": sha256_file(source_heldout_targets),
        "evidence_bridge": display_path(evidence_bridge),
        "evidence_bridge_sha256": sha256_file(evidence_bridge),
        "rows_path": display_path(rows_out),
        "train_rows_path": display_path(train_out),
        "heldout_rows_path": display_path(heldout_out),
        "manifest_path": display_path(manifest_out),
        "candidate_pairs": list(CANDIDATE_PAIRS),
        "row_count": len(rows),
        "train_rows": len(train_rows),
        "heldout_rows": len(heldout_rows),
        "bridge_focus_rows": len(bridge_focus_rows),
        "bridge_focus_case_ids": value_set(bridge_focus_rows, "source_manifest_case_id"),
        "bridge_runtime_reasons_by_case": dict(sorted(bridge_reasons_by_case.items())),
        "by_target_pair": count_by(rows, "target_pair"),
        "train_by_target_pair": count_by(train_rows, "target_pair"),
        "heldout_by_target_pair": count_by(heldout_rows, "target_pair"),
        "bridge_focus_by_target_pair": count_by(bridge_focus_rows, "target_pair"),
        "train_case_ids": train_case_ids,
        "heldout_case_ids": heldout_case_ids,
        "overlap_case_ids": sorted(set(train_case_ids) & set(heldout_case_ids)),
        "train_split_groups": train_split_groups,
        "heldout_split_groups": heldout_split_groups,
        "overlap_split_groups": sorted(set(train_split_groups) & set(heldout_split_groups)),
        "train_source_task_ids": train_source_task_ids,
        "heldout_source_task_ids": heldout_source_task_ids,
        "overlap_source_task_ids": sorted(set(train_source_task_ids) & set(heldout_source_task_ids)),
        "public_safety_contract": {
            "raw_prediction_jsonl_read": False,
            "raw_candidate_score_jsonl_read": False,
            "scheduler_logs_read": False,
            "model_state_read": False,
            "ignored_run_folder_read": False,
            "hidden_eval_metadata_in_model_visible_task": False,
        },
        "boundary": (
            "Rows are a public-safe finite-candidate routing substrate. Bridge "
            "focus rows remain held-out evaluation rows and are not training "
            "data for the previous saved-output checkpoints."
        ),
        "next_decision": (
            "Use these rows for no-model validation and, only after that, a "
            "small evidence-conditioned candidate-routing smoke. Keep tool_query, "
            "DPO/RLVR, HF publication, and release tagging gated."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
        "--evidence-bridge",
        default="post_training/stage_a_saved_output_evidence_bridge_2026-07-10.json",
    )
    parser.add_argument(
        "--rows-out",
        default="post_training/stage_a_saved_output_evidence_candidate_routing_rows_v1.jsonl",
    )
    parser.add_argument(
        "--train-out",
        default="post_training/stage_a_saved_output_evidence_candidate_routing_train_v1.jsonl",
    )
    parser.add_argument(
        "--heldout-out",
        default="post_training/stage_a_saved_output_evidence_candidate_routing_heldout_v1.jsonl",
    )
    parser.add_argument(
        "--manifest-out",
        default="post_training/stage_a_saved_output_evidence_candidate_routing_manifest.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_rows = load_jsonl(args.source_targets)
    train_source_rows = load_jsonl(args.source_train_targets)
    heldout_source_rows = load_jsonl(args.source_heldout_targets)
    bridge = load_json(args.evidence_bridge)
    bridge_reasons = bridge_case_reasons(bridge)
    train_rows = build_rows(
        source_rows=source_rows,
        split_rows=train_source_rows,
        split="train",
        bridge_reasons_by_case=bridge_reasons,
    )
    heldout_rows = build_rows(
        source_rows=source_rows,
        split_rows=heldout_source_rows,
        split="heldout",
        bridge_reasons_by_case=bridge_reasons,
    )
    rows = train_rows + heldout_rows
    manifest = manifest_for_rows(
        source_targets=args.source_targets,
        source_train_targets=args.source_train_targets,
        source_heldout_targets=args.source_heldout_targets,
        evidence_bridge=args.evidence_bridge,
        rows_out=args.rows_out,
        train_out=args.train_out,
        heldout_out=args.heldout_out,
        manifest_out=args.manifest_out,
        rows=rows,
        train_rows=train_rows,
        heldout_rows=heldout_rows,
        bridge_reasons_by_case=bridge_reasons,
    )
    write_jsonl(args.rows_out, rows)
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

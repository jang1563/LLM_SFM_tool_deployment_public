#!/usr/bin/env python3
"""Export a targeted Stage A saved-output action/status calibration probe.

This artifact follows the saved-output next-decision checkpoint. It uses compact
public gate summaries only to identify the failed action/status families, then
builds train-only calibration pairs and held-out evaluation-only probe pairs
from the existing Stage A SFT split.
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

from post_training.evaluate_stage_a_predictions import (  # noqa: E402
    build_report as build_prediction_eval_report,
)
from post_training.generate_stage_a_predictions import (  # noqa: E402
    api_prompt_messages,
    generation_prompt_hash,
    prompt_hash,
    source_case_id,
)
from post_training.run_stage_a_saved_prediction_candidate_readout import (  # noqa: E402
    candidate_prediction_for_row,
    target_pair_from_sft_row,
)
from post_training.run_stage_a_sft_smoke_eval import (  # noqa: E402
    load_jsonl,
    load_manifest_rows,
    write_json,
)
from post_training.run_stage_a_strict_component_sft_smoke import write_jsonl  # noqa: E402


DATASET = "negbiodb_ct_stage_a_saved_output_calibration_probe_v1"
MANIFEST_DATASET = "negbiodb_ct_stage_a_saved_output_calibration_probe_manifest_v1"
NEXT_DECISION_PATH = "post_training/stage_a_saved_output_next_decision_2026-07-09.json"
COLLAPSE_PAIR = {"action": "ground", "evidence_status": "supported"}
PROMPT_CONTRACT = "stage_a_v4_canonical_json"


def load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def display_path(path: str | Path) -> str:
    path = Path(path)
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def pair_label(pair: Mapping[str, Any]) -> str:
    return f"{pair.get('action')}/{pair.get('evidence_status')}"


def pair_from_label(label: str) -> dict[str, str]:
    action, evidence_status = label.split("/", 1)
    return {"action": action, "evidence_status": evidence_status}


def target_failure_pairs(next_decision: Mapping[str, Any]) -> list[dict[str, str]]:
    bottleneck = next_decision.get("bottleneck", {})
    if not isinstance(bottleneck, Mapping):
        raise ValueError("next-decision checkpoint is missing bottleneck")
    counts = bottleneck.get("candidate_failure_target_pair_counts", {})
    if not isinstance(counts, Mapping):
        raise ValueError("next-decision checkpoint is missing candidate failure target counts")
    labels = sorted(
        label
        for label, count in counts.items()
        if int(count) > 0 and label != pair_label(COLLAPSE_PAIR)
    )
    if not labels:
        raise ValueError("no non-collapse target pairs were selected for calibration")
    return [pair_from_label(label) for label in labels]


def score_prediction(
    *,
    manifest_rows: Sequence[Mapping[str, Any]],
    row: Mapping[str, Any],
    prediction: Mapping[str, Any],
    run_id: str,
) -> dict[str, Any]:
    case_id = source_case_id(row)
    report = build_prediction_eval_report(
        manifest_rows=manifest_rows,
        prediction_rows=[
            {
                "case_id": case_id,
                "run_id": run_id,
                "split": row.get("split"),
                "prediction": dict(prediction),
            }
        ],
        expected_case_ids=[case_id],
        run_id=run_id,
    )
    return dict(report["rows"][0])


def final_json_message(output: Mapping[str, Any]) -> list[dict[str, str]]:
    return [{"role": "assistant", "content": json.dumps(dict(output), sort_keys=True)}]


def target_cited_source_ids(row: Mapping[str, Any]) -> list[str]:
    trajectory = row.get("target_trajectory")
    if not isinstance(trajectory, Mapping):
        return []
    source_ids = trajectory.get("cited_source_ids", [])
    if not isinstance(source_ids, Sequence) or isinstance(source_ids, (str, bytes)):
        return []
    return [str(source_id) for source_id in source_ids]


def chosen_prediction_for_row(
    row: Mapping[str, Any],
    pair: Mapping[str, str],
    *,
    prompt_contract: str,
) -> dict[str, Any]:
    prediction = candidate_prediction_for_row(row, pair, prompt_contract=prompt_contract)
    citations = target_cited_source_ids(row)
    if citations:
        prediction["cited_source_ids"] = citations
    return prediction


def calibration_pair_for_row(
    row: Mapping[str, Any],
    *,
    manifest_rows: Sequence[Mapping[str, Any]],
    selected_pairs: set[str],
    source_next_decision: str | Path,
    prompt_contract: str,
) -> dict[str, Any] | None:
    chosen_pair = target_pair_from_sft_row(row)
    chosen_label = pair_label(chosen_pair)
    if chosen_label not in selected_pairs:
        return None
    rejected_pair = dict(COLLAPSE_PAIR)
    if chosen_pair == rejected_pair:
        return None

    case_id = source_case_id(row)
    chosen_output = chosen_prediction_for_row(row, chosen_pair, prompt_contract=prompt_contract)
    rejected_output = candidate_prediction_for_row(row, rejected_pair, prompt_contract=prompt_contract)
    run_prefix = f"stage_a_saved_output_calibration_probe::{case_id}"
    chosen_score = score_prediction(
        manifest_rows=manifest_rows,
        row=row,
        prediction=chosen_output,
        run_id=f"{run_prefix}::chosen",
    )
    rejected_score = score_prediction(
        manifest_rows=manifest_rows,
        row=row,
        prediction=rejected_output,
        run_id=f"{run_prefix}::rejected",
    )
    split = str(row.get("split"))
    return {
        "id": f"{run_prefix}::target_vs_ground_supported",
        "dataset": DATASET,
        "source_next_decision": display_path(source_next_decision),
        "source_sft_id": row.get("id"),
        "source_manifest_case_id": row.get("source_manifest_case_id"),
        "source_task_id": row.get("source_task_id"),
        "split_group": row.get("split_group"),
        "case_id": case_id,
        "case_family": row.get("case_family"),
        "split": split,
        "prompt_contract": prompt_contract,
        "calibration_axis": "target_pair_vs_ground_supported",
        "prompt_messages": api_prompt_messages(row, prompt_contract=prompt_contract),
        "prompt_hash": prompt_hash(row),
        "generation_prompt_hash": generation_prompt_hash(row, prompt_contract=prompt_contract),
        "chosen_pair": chosen_label,
        "rejected_pair": pair_label(rejected_pair),
        "chosen_output": chosen_output,
        "rejected_output": rejected_output,
        "chosen_messages": final_json_message(chosen_output),
        "rejected_messages": final_json_message(rejected_output),
        "chosen_score": chosen_score,
        "rejected_score": rejected_score,
        "training_allowed": split == "train",
        "evaluation_only": split != "train",
        "oracle_target": True,
    }


def build_pairs(
    rows: Sequence[Mapping[str, Any]],
    *,
    manifest_rows: Sequence[Mapping[str, Any]],
    selected_pairs: set[str],
    source_next_decision: str | Path,
    prompt_contract: str,
) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        pair = calibration_pair_for_row(
            row,
            manifest_rows=manifest_rows,
            selected_pairs=selected_pairs,
            source_next_decision=source_next_decision,
            prompt_contract=prompt_contract,
        )
        if pair is not None:
            out.append(pair)
    return out


def value_set(rows: Sequence[Mapping[str, Any]], key: str) -> list[str]:
    return sorted(str(row[key]) for row in rows if row.get(key) is not None)


def count_by(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(key)) for row in rows).items()))


def manifest_for_pairs(
    *,
    next_decision_path: str | Path,
    train_sft: str | Path,
    heldout_sft: str | Path,
    pairs_out: str | Path,
    train_out: str | Path,
    heldout_out: str | Path,
    rows: Sequence[Mapping[str, Any]],
    train_rows: Sequence[Mapping[str, Any]],
    heldout_rows: Sequence[Mapping[str, Any]],
    selected_pairs: Sequence[Mapping[str, str]],
    prompt_contract: str,
) -> dict[str, Any]:
    train_cases = set(value_set(train_rows, "source_manifest_case_id"))
    heldout_cases = set(value_set(heldout_rows, "source_manifest_case_id"))
    train_groups = set(value_set(train_rows, "split_group"))
    heldout_groups = set(value_set(heldout_rows, "split_group"))
    train_tasks = set(value_set(train_rows, "source_task_id"))
    heldout_tasks = set(value_set(heldout_rows, "source_task_id"))
    return {
        "dataset": MANIFEST_DATASET,
        "pair_dataset": DATASET,
        "source_next_decision": display_path(next_decision_path),
        "source_train_sft": display_path(train_sft),
        "source_heldout_sft": display_path(heldout_sft),
        "pairs": display_path(pairs_out),
        "train_pairs_path": display_path(train_out),
        "heldout_pairs_path": display_path(heldout_out),
        "prompt_contract": prompt_contract,
        "selected_next_step": "targeted_action_status_calibration_probe",
        "selected_target_pairs": [pair_label(pair) for pair in selected_pairs],
        "rejected_pair": pair_label(COLLAPSE_PAIR),
        "pair_examples": len(rows),
        "train_pairs": len(train_rows),
        "heldout_probe_pairs": len(heldout_rows),
        "by_chosen_pair": count_by(rows, "chosen_pair"),
        "train_by_chosen_pair": count_by(train_rows, "chosen_pair"),
        "heldout_by_chosen_pair": count_by(heldout_rows, "chosen_pair"),
        "by_case_family": count_by(rows, "case_family"),
        "train_case_ids": sorted(train_cases),
        "heldout_case_ids": sorted(heldout_cases),
        "overlap_case_ids": sorted(train_cases & heldout_cases),
        "overlap_split_groups": sorted(train_groups & heldout_groups),
        "overlap_source_task_ids": sorted(train_tasks & heldout_tasks),
        "training_rule": (
            "Only rows with split=train are training-allowed. Held-out rows are "
            "evaluation-only probe examples and must not be used for calibration training."
        ),
        "artifact_policy": {
            "raw_saved_predictions_committed": False,
            "candidate_score_jsonl_committed": False,
            "scheduler_logs_committed": False,
            "model_state_committed": False,
        },
        "minimum_next_gate": {
            "fail_closed_gate_trusted_incorrect": 0,
            "fail_closed_gate_strict_final_correct_min": 4,
        },
        "boundary": (
            "This is a targeted saved-output action/status calibration probe. It "
            "is not DPO/RLVR data, not broad retraining, and not a release-readiness claim."
        ),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--next-decision", default=NEXT_DECISION_PATH)
    parser.add_argument("--train-sft", default="post_training/stage_a_sft_train_v1.jsonl")
    parser.add_argument("--heldout-sft", default="post_training/stage_a_sft_heldout_v1.jsonl")
    parser.add_argument("--manifest", default="negbiodb_ct/stage_a_mini_manifest.jsonl")
    parser.add_argument(
        "--pairs-out",
        default="post_training/stage_a_saved_output_calibration_probe_v1.jsonl",
    )
    parser.add_argument(
        "--train-out",
        default="post_training/stage_a_saved_output_calibration_probe_train_v1.jsonl",
    )
    parser.add_argument(
        "--heldout-out",
        default="post_training/stage_a_saved_output_calibration_probe_heldout_v1.jsonl",
    )
    parser.add_argument(
        "--manifest-out",
        default="post_training/stage_a_saved_output_calibration_probe_manifest.json",
    )
    parser.add_argument("--prompt-contract", default=PROMPT_CONTRACT)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    next_decision = load_json(args.next_decision)
    selected_pairs = target_failure_pairs(next_decision)
    selected_pair_labels = {pair_label(pair) for pair in selected_pairs}
    manifest_rows = load_manifest_rows(args.manifest)
    train_sft_rows = load_jsonl(args.train_sft)
    heldout_sft_rows = load_jsonl(args.heldout_sft)
    train_pairs = build_pairs(
        train_sft_rows,
        manifest_rows=manifest_rows,
        selected_pairs=selected_pair_labels,
        source_next_decision=args.next_decision,
        prompt_contract=args.prompt_contract,
    )
    heldout_pairs = build_pairs(
        heldout_sft_rows,
        manifest_rows=manifest_rows,
        selected_pairs=selected_pair_labels,
        source_next_decision=args.next_decision,
        prompt_contract=args.prompt_contract,
    )
    rows = train_pairs + heldout_pairs
    manifest = manifest_for_pairs(
        next_decision_path=args.next_decision,
        train_sft=args.train_sft,
        heldout_sft=args.heldout_sft,
        pairs_out=args.pairs_out,
        train_out=args.train_out,
        heldout_out=args.heldout_out,
        rows=rows,
        train_rows=train_pairs,
        heldout_rows=heldout_pairs,
        selected_pairs=selected_pairs,
        prompt_contract=args.prompt_contract,
    )
    issues = []
    if manifest["overlap_case_ids"]:
        issues.append("train_heldout_case_overlap")
    if manifest["overlap_split_groups"]:
        issues.append("train_heldout_split_group_overlap")
    if manifest["overlap_source_task_ids"]:
        issues.append("train_heldout_source_task_overlap")
    if any(not row["chosen_score"]["passed"] for row in rows):
        issues.append("chosen_output_does_not_pass")
    if any(row["rejected_score"]["passed"] for row in rows):
        issues.append("rejected_collapse_passes")
    if issues:
        raise ValueError(f"calibration probe validation failed: {issues}")
    write_jsonl(args.pairs_out, rows)
    write_jsonl(args.train_out, train_pairs)
    write_jsonl(args.heldout_out, heldout_pairs)
    write_json(args.manifest_out, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

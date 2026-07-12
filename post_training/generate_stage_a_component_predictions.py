#!/usr/bin/env python3
"""Generate no-model Stage A component saved-prediction baselines.

The output JSONL uses the same component prediction contract as cluster SFT
smoke runs, then scores it offline with the component evaluator. This keeps
prompt-only/model-heavy experiments comparable to deterministic baselines.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from post_training.run_stage_a_strict_component_sft_smoke import (  # noqa: E402
    PREDICTION_DATASET,
    build_component_eval_report,
    component_case_id,
    filter_component,
    load_jsonl,
    target_output_from_row,
    visible_citation_ids,
    write_json,
    write_jsonl,
)

DATASET = "negbiodb_ct_stage_a_component_saved_prediction_baselines_v1"
MODES = (
    "oracle",
    "empty_object",
    "majority_ground_supported",
    "routing_no_citations",
    "tool_names_only",
)


def tool_names_only(row: Mapping[str, Any]) -> list[str]:
    target = target_output_from_row(row)
    calls = target.get("tool_calls")
    if not isinstance(calls, list):
        return []
    names: list[str] = []
    for call in calls:
        if isinstance(call, Mapping) and isinstance(call.get("name"), str):
            names.append(str(call["name"]))
    return names


def prediction_for_row(row: Mapping[str, Any], *, component: str, mode: str) -> dict[str, Any]:
    if mode == "oracle":
        return target_output_from_row(row)
    if mode == "empty_object":
        return {}
    if mode == "majority_ground_supported":
        if component == "enum_action":
            return {"action": "ground", "evidence_status": "supported"}
        if component == "routing_after_loop":
            return {
                "action": "ground",
                "evidence_status": "supported",
                "cited_source_ids": visible_citation_ids(row)[:1],
            }
    if mode == "routing_no_citations" and component == "routing_after_loop":
        target = target_output_from_row(row)
        return {
            "action": target.get("action"),
            "evidence_status": target.get("evidence_status"),
            "cited_source_ids": [],
        }
    if mode == "tool_names_only" and component == "tool_query":
        return {"tool_calls": tool_names_only(row)}
    raise ValueError(f"mode {mode!r} is not valid for component {component!r}")


def prediction_rows_for_component(
    rows: Sequence[Mapping[str, Any]],
    *,
    component: str,
    mode: str,
    run_id: str,
) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    for row in rows:
        predictions.append(
            {
                "id": f"{run_id}::{row['id']}",
                "source_component_target_id": row["id"],
                "case_id": component_case_id(row),
                "dataset": PREDICTION_DATASET,
                "source": f"stage_a_component_saved_prediction::{mode}",
                "run_id": run_id,
                "component": component,
                "prompt_contract": row.get("prompt_contract"),
                "split": row.get("split"),
                "generation_prompt_hash": row.get("generation_prompt_hash"),
                "prediction": prediction_for_row(row, component=component, mode=mode),
            }
        )
    return predictions


def build_saved_prediction_report(
    *,
    expected_rows: Sequence[Mapping[str, Any]],
    prediction_rows: Sequence[Mapping[str, Any]],
    component: str,
    mode: str,
    run_id: str,
    targets_path: str,
    predictions_path: Path,
    eval_report_path: Path,
) -> dict[str, Any]:
    eval_report = build_component_eval_report(
        expected_rows=expected_rows,
        prediction_rows=prediction_rows,
        component=component,
        run_id=run_id,
    )
    write_json(eval_report_path, eval_report)
    return {
        "dataset": DATASET,
        "run_id": run_id,
        "component": component,
        "mode": mode,
        "targets": targets_path,
        "examples": len(expected_rows),
        "predictions": str(predictions_path),
        "eval_report": str(eval_report_path),
        "eval_summary": eval_report["summary"],
        "boundary": (
            "No-model component saved-prediction baseline. The output is scored "
            "offline with the same component evaluator used for cluster SFT runs."
        ),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--component", choices=("enum_action", "tool_query", "routing_after_loop"), required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument(
        "--targets",
        default="post_training/stage_a_evidence_conditioned_component_targets_heldout_v1.jsonl",
    )
    parser.add_argument("--out-dir", default="post_training/runs/stage_a_component_saved_predictions")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--predictions-out", default=None)
    parser.add_argument("--eval-out", default=None)
    parser.add_argument("--report-out", default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.run_id is None:
        args.run_id = f"stage_a_component_{args.component}_{args.mode}"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = Path(args.predictions_out) if args.predictions_out else out_dir / "predictions.jsonl"
    eval_report_path = Path(args.eval_out) if args.eval_out else out_dir / "eval_report.json"
    report_path = Path(args.report_out) if args.report_out else out_dir / "report.json"

    expected_rows = filter_component(load_jsonl(args.targets), args.component)
    predictions = prediction_rows_for_component(
        expected_rows,
        component=args.component,
        mode=args.mode,
        run_id=args.run_id,
    )
    write_jsonl(predictions_path, predictions)
    report = build_saved_prediction_report(
        expected_rows=expected_rows,
        prediction_rows=predictions,
        component=args.component,
        mode=args.mode,
        run_id=args.run_id,
        targets_path=args.targets,
        predictions_path=predictions_path,
        eval_report_path=eval_report_path,
    )
    write_json(report_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compare Stage A routing baselines against the runtime evidence gate.

This is a public-safe, no-model diagnostic. It asks whether simple component
baselines match the all-family runtime evidence gate before any new
tool-query, DPO, or RLVR work is justified.
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

from post_training.evaluate_stage_a_routing_evidence_gate import (  # noqa: E402
    evidence_features,
    gate_output,
    pair_label,
)
from post_training.generate_stage_a_component_predictions import (  # noqa: E402
    prediction_rows_for_component,
)
from post_training.run_stage_a_strict_component_sft_smoke import (  # noqa: E402
    PREDICTION_DATASET,
    build_component_eval_report,
    component_case_id,
    filter_component,
    load_jsonl,
    target_output_from_row,
    write_json,
)

DATASET = "negbiodb_ct_stage_a_routing_gate_baseline_comparison_v1"
COMPONENT = "routing_after_loop"
POLICIES = (
    "runtime_evidence_gate",
    "oracle",
    "majority_ground_supported",
    "routing_no_citations",
    "empty_object",
)


def gate_prediction_for_row(row: Mapping[str, Any]) -> dict[str, Any]:
    prediction, _reason = gate_output(evidence_features(row))
    return prediction


def gate_prediction_rows(rows: Sequence[Mapping[str, Any]], *, run_id: str) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    for row in rows:
        prediction, reason = gate_output(evidence_features(row))
        predictions.append(
            {
                "id": f"{run_id}::{row['id']}",
                "source_component_target_id": row["id"],
                "case_id": component_case_id(row),
                "dataset": PREDICTION_DATASET,
                "source": "stage_a_routing_runtime_evidence_gate",
                "run_id": run_id,
                "component": COMPONENT,
                "prompt_contract": row.get("prompt_contract"),
                "split": row.get("split"),
                "generation_prompt_hash": row.get("generation_prompt_hash"),
                "prediction": prediction,
                "gate_reason": reason,
            }
        )
    return predictions


def policy_prediction_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    policy: str,
    run_id: str,
) -> list[dict[str, Any]]:
    if policy == "runtime_evidence_gate":
        return gate_prediction_rows(rows, run_id=run_id)
    return prediction_rows_for_component(rows, component=COMPONENT, mode=policy, run_id=run_id)


def prediction_by_target_id(predictions: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {
        str(row.get("source_component_target_id") or row.get("id")): row
        for row in predictions
    }


def action_status(output: Mapping[str, Any]) -> tuple[Any, Any]:
    return output.get("action"), output.get("evidence_status")


def summarize_alignment(
    *,
    expected_rows: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    eval_summary: Mapping[str, Any],
) -> dict[str, Any]:
    predictions_by_id = prediction_by_target_id(predictions)
    gate_full_agreement = 0
    gate_action_status_agreement = 0
    citation_agreement = 0
    unsafe_ground_supported = 0
    pair_counts: Counter[str] = Counter()
    citation_mismatches = 0

    for row in expected_rows:
        row_id = str(row["id"])
        prediction_row = predictions_by_id[row_id]
        prediction = prediction_row.get("prediction")
        prediction = prediction if isinstance(prediction, Mapping) else {}
        gate_prediction = gate_prediction_for_row(row)
        target = target_output_from_row(row)
        pair_counts[pair_label(prediction)] += 1
        if dict(prediction) == gate_prediction:
            gate_full_agreement += 1
        if action_status(prediction) == action_status(gate_prediction):
            gate_action_status_agreement += 1
        if list(prediction.get("cited_source_ids", [])) == list(gate_prediction.get("cited_source_ids", [])):
            citation_agreement += 1
        if pair_label(prediction) == "ground/supported" and pair_label(gate_prediction) != "ground/supported":
            unsafe_ground_supported += 1
        if list(prediction.get("cited_source_ids", [])) != list(target.get("cited_source_ids", [])):
            citation_mismatches += 1

    total = len(expected_rows)
    return {
        "cases": total,
        "target_exact": eval_summary.get("passed", 0),
        "target_accuracy": round(float(eval_summary.get("passed", 0)) / total, 6) if total else 0.0,
        "mean_score": eval_summary.get("mean_score", 0.0),
        "gate_full_agreement": gate_full_agreement,
        "gate_full_agreement_rate": round(gate_full_agreement / total, 6) if total else 0.0,
        "gate_action_status_agreement": gate_action_status_agreement,
        "gate_action_status_agreement_rate": round(gate_action_status_agreement / total, 6) if total else 0.0,
        "citation_agreement": citation_agreement,
        "citation_agreement_rate": round(citation_agreement / total, 6) if total else 0.0,
        "citation_mismatches_vs_target": citation_mismatches,
        "unsafe_ground_supported_overrides": unsafe_ground_supported,
        "predicted_pairs": dict(sorted(pair_counts.items())),
        "violations": eval_summary.get("violations", {}),
    }


def score_policy(
    *,
    rows: Sequence[Mapping[str, Any]],
    policy: str,
    split_label: str,
) -> dict[str, Any]:
    run_id = f"stage_a_routing_gate_baseline_comparison_{policy}_{split_label}"
    predictions = policy_prediction_rows(rows, policy=policy, run_id=run_id)
    eval_report = build_component_eval_report(
        expected_rows=rows,
        prediction_rows=predictions,
        component=COMPONENT,
        run_id=run_id,
    )
    return {
        "policy": policy,
        "split": split_label,
        "summary": summarize_alignment(
            expected_rows=rows,
            predictions=predictions,
            eval_summary=eval_report["summary"],
        ),
        "component_eval_summary": eval_report["summary"],
    }


def build_report(
    *,
    targets: Sequence[Mapping[str, Any]],
    train_targets: Sequence[Mapping[str, Any]],
    heldout_targets: Sequence[Mapping[str, Any]],
    targets_path: Path,
) -> dict[str, Any]:
    split_rows = {
        "all": filter_component(list(targets), COMPONENT),
        "train": filter_component(list(train_targets), COMPONENT),
        "heldout": filter_component(list(heldout_targets), COMPONENT),
    }
    policy_reports = {
        policy: {
            split: score_policy(rows=rows, policy=policy, split_label=split)
            for split, rows in split_rows.items()
        }
        for policy in POLICIES
    }
    return {
        "dataset": DATASET,
        "component": COMPONENT,
        "input_targets": str(targets_path),
        "policies": list(POLICIES),
        "model_visible_fields_only_for_runtime_gate": True,
        "hidden_labels_used_by_runtime_gate": False,
        "policy_reports": policy_reports,
        "scientific_readout": {
            "diagnostic_question": (
                "Do simple routing component baselines match the all-family "
                "runtime evidence gate on target exactness, citation grounding, "
                "and fail-closed behavior?"
            ),
            "interpretation_rule": (
                "The runtime gate and oracle are sanity baselines. A model "
                "prediction path should not advance to tool_query, DPO/RLVR, "
                "HF publication, or release tagging until it beats collapse and "
                "citationless baselines and is competitive with the runtime gate."
            ),
            "next_decision": (
                "Compare saved model/component outputs against this report. "
                "If the model only matches action/status but misses citations "
                "or fail-closed routing, keep runtime enforcement in the system "
                "and avoid new optimization objectives."
            ),
        },
    }


def format_summary_cell(report: Mapping[str, Any], split: str, key: str) -> str:
    summary = report["policy_reports"][key][split]["summary"]
    return (
        f"{summary['target_exact']}/{summary['cases']} exact; "
        f"{summary['gate_full_agreement']}/{summary['cases']} gate-agree; "
        f"mean {summary['mean_score']:.3f}"
    )


def write_markdown(report: Mapping[str, Any], path: Path) -> None:
    lines = [
        "# Stage A Routing Gate Baseline Comparison",
        "",
        "Purpose: compare no-model routing component baselines against the",
        "all-family runtime evidence gate on public Stage A evidence-conditioned",
        "routing rows.",
        "",
        "## Summary",
        "",
        "| Policy | All | Train | Held-out | Unsafe ground/supported overrides (all) | Citation mismatches (all) |",
        "| --- | --- | --- | --- | ---: | ---: |",
    ]
    for policy in POLICIES:
        all_summary = report["policy_reports"][policy]["all"]["summary"]
        lines.append(
            "| {policy} | {all_cell} | {train_cell} | {heldout_cell} | {unsafe} | {citation_mismatch} |".format(
                policy=f"`{policy}`",
                all_cell=format_summary_cell(report, "all", policy),
                train_cell=format_summary_cell(report, "train", policy),
                heldout_cell=format_summary_cell(report, "heldout", policy),
                unsafe=all_summary["unsafe_ground_supported_overrides"],
                citation_mismatch=all_summary["citation_mismatches_vs_target"],
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            str(report["scientific_readout"]["interpretation_rule"]),
            "",
            str(report["scientific_readout"]["next_decision"]),
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--targets",
        default="post_training/stage_a_evidence_conditioned_component_targets_v1.jsonl",
    )
    parser.add_argument(
        "--train-targets",
        default="post_training/stage_a_evidence_conditioned_component_targets_train_v1.jsonl",
    )
    parser.add_argument(
        "--heldout-targets",
        default="post_training/stage_a_evidence_conditioned_component_targets_heldout_v1.jsonl",
    )
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--out-md", default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    targets_path = Path(args.targets)
    report = build_report(
        targets=load_jsonl(targets_path),
        train_targets=load_jsonl(args.train_targets),
        heldout_targets=load_jsonl(args.heldout_targets),
        targets_path=targets_path,
    )
    if args.out_json:
        write_json(args.out_json, report)
    if args.out_md:
        write_markdown(report, Path(args.out_md))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

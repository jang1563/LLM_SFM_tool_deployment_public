#!/usr/bin/env python3
"""Compare saved-output candidate calibration with runtime evidence arbitration.

This evaluator reads compact public artifacts only: the saved-output candidate
calibration report and tracked evidence-conditioned routing targets. It does
not read raw candidate-score JSONL, prompts from ignored run folders, scheduler
logs, or model state.
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
)
from post_training.run_stage_a_strict_component_sft_smoke import load_jsonl, write_json  # noqa: E402


DATASET = "negbiodb_ct_stage_a_saved_output_candidate_arbitration_v1"
DEFAULT_CALIBRATION_REPORT = (
    "post_training/stage_a_saved_output_candidate_calibration_qwen05b_cayuga_summary_2026-07-10.json"
)
DEFAULT_EVIDENCE_TARGETS = "post_training/stage_a_evidence_conditioned_component_targets_v1.jsonl"
ROUTING_COMPONENT = "routing_after_loop"
POLICIES = (
    "raw_candidate_top1",
    "calibrated_candidate_top1",
    "train_selected_score_gap_gate",
    "evidence_gate_override",
    "hybrid_evidence_then_train_gate",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def evidence_target_by_case(targets: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    out: dict[str, Mapping[str, Any]] = {}
    for row in targets:
        if row.get("component") != ROUTING_COMPONENT:
            continue
        case_id = row.get("source_manifest_case_id")
        if isinstance(case_id, str):
            out[case_id] = row
    return out


def evidence_gate_prediction(row: Mapping[str, Any] | None) -> tuple[str | None, str]:
    if row is None:
        return None, "missing_evidence_target_fail_closed"
    predicted, reason = gate_output(evidence_features(row))
    return pair_label(predicted), reason


def train_gate_prediction(
    row: Mapping[str, Any],
    *,
    threshold: float,
    fail_closed_pair_label: str,
) -> tuple[str, str, bool]:
    gap = row.get("calibrated_top_second_gap")
    if gap is not None and float(gap) >= threshold:
        return str(row.get("calibrated_top_pair_label")), "trust_calibrated_high_gap", True
    return fail_closed_pair_label, "fail_closed_train_selected_low_gap", False


def policy_prediction(
    *,
    policy: str,
    row: Mapping[str, Any],
    evidence_row: Mapping[str, Any] | None,
    threshold: float,
    fail_closed_pair_label: str,
) -> tuple[str | None, str, bool]:
    if policy == "raw_candidate_top1":
        return str(row.get("raw_top_pair_label")), "trust_raw_candidate_top1", True
    if policy == "calibrated_candidate_top1":
        return str(row.get("calibrated_top_pair_label")), "trust_calibrated_candidate_top1", True
    if policy == "train_selected_score_gap_gate":
        return train_gate_prediction(
            row,
            threshold=threshold,
            fail_closed_pair_label=fail_closed_pair_label,
        )
    if policy == "evidence_gate_override":
        predicted_pair, reason = evidence_gate_prediction(evidence_row)
        return predicted_pair, reason, False
    if policy == "hybrid_evidence_then_train_gate":
        predicted_pair, reason = evidence_gate_prediction(evidence_row)
        if predicted_pair is not None:
            return predicted_pair, reason, False
        return train_gate_prediction(
            row,
            threshold=threshold,
            fail_closed_pair_label=fail_closed_pair_label,
        )
    raise ValueError(f"unknown policy: {policy}")


def summarize_policy(rows: Sequence[Mapping[str, Any]], policy: str) -> dict[str, Any]:
    selected = [row for row in rows if row.get("policy") == policy]
    exact = sum(1 for row in selected if row.get("exact"))
    trusted = [row for row in selected if row.get("trusted_candidate")]
    trusted_incorrect = [row for row in trusted if not row.get("exact")]
    return {
        "policy": policy,
        "rows": len(selected),
        "exact": exact,
        "accuracy": round(exact / len(selected), 6) if selected else 0.0,
        "trusted_candidate": len(trusted),
        "trusted_candidate_incorrect": len(trusted_incorrect),
        "by_predicted_pair": dict(sorted(Counter(str(row.get("predicted_pair")) for row in selected).items())),
        "by_reason": dict(sorted(Counter(str(row.get("reason")) for row in selected).items())),
        "error_case_ids": [str(row.get("case_id")) for row in selected if not row.get("exact")],
    }


def build_arbitration_report(
    *,
    calibration_report: Mapping[str, Any],
    evidence_targets: Sequence[Mapping[str, Any]],
    calibration_report_path: str | Path,
    evidence_targets_path: str | Path,
) -> dict[str, Any]:
    threshold = calibration_report.get("train_selected_zero_unsafe_threshold")
    if threshold is None:
        raise ValueError("calibration report missing train_selected_zero_unsafe_threshold")
    gate_report = calibration_report.get("train_selected_gate_report")
    if not isinstance(gate_report, Mapping):
        raise ValueError("calibration report missing train_selected_gate_report")
    fail_closed_pair_label = pair_label(gate_report.get("fail_closed_pair") or {})
    evidence_by_case = evidence_target_by_case(evidence_targets)
    rows: list[dict[str, Any]] = []
    for candidate_row in calibration_report.get("rows", []):
        if not isinstance(candidate_row, Mapping):
            continue
        case_id = str(candidate_row.get("case_id"))
        evidence_row = evidence_by_case.get(case_id)
        target_pair = str(candidate_row.get("target_pair_label"))
        for policy in POLICIES:
            predicted_pair, reason, trusted_candidate = policy_prediction(
                policy=policy,
                row=candidate_row,
                evidence_row=evidence_row,
                threshold=float(threshold),
                fail_closed_pair_label=fail_closed_pair_label,
            )
            rows.append(
                {
                    "case_id": case_id,
                    "target_pair_label": target_pair,
                    "raw_top_pair_label": candidate_row.get("raw_top_pair_label"),
                    "calibrated_top_pair_label": candidate_row.get("calibrated_top_pair_label"),
                    "calibrated_top_second_gap": candidate_row.get("calibrated_top_second_gap"),
                    "evidence_gate_pair": (
                        evidence_gate_prediction(evidence_row)[0] if evidence_row is not None else None
                    ),
                    "policy": policy,
                    "predicted_pair": predicted_pair,
                    "exact": predicted_pair == target_pair,
                    "trusted_candidate": trusted_candidate,
                    "reason": reason,
                }
            )
    by_policy = {policy: summarize_policy(rows, policy) for policy in POLICIES}
    best_exact = max((summary["exact"] for summary in by_policy.values()), default=0)
    return {
        "dataset": DATASET,
        "run_id": calibration_report.get("run_id"),
        "component": "saved_output_candidate_arbitration",
        "calibration_report": str(calibration_report_path),
        "evidence_targets": str(evidence_targets_path),
        "input_calibration_report_sha256": sha256_file(Path(calibration_report_path)),
        "input_evidence_targets_sha256": sha256_file(Path(evidence_targets_path)),
        "calibration_mode": calibration_report.get("calibration_mode"),
        "candidate_policy": calibration_report.get("candidate_policy"),
        "candidate_target_format": calibration_report.get("candidate_target_format"),
        "train_selected_zero_unsafe_threshold": float(threshold),
        "fail_closed_pair": gate_report.get("fail_closed_pair"),
        "model_visible_evidence_gate": True,
        "hidden_labels_used_by_arbitration": False,
        "cases": len(calibration_report.get("rows", [])),
        "summary": {
            "by_policy": by_policy,
            "best_policy_names": [
                name for name, summary in by_policy.items() if summary["exact"] == best_exact
            ],
        },
        "rows": rows,
        "scientific_readout": {
            "diagnostic_question": (
                "Can saved-output candidate calibration be safely reused when "
                "arbitrated against model-visible evidence routing?"
            ),
            "interpretation_rule": (
                "If evidence-gate or hybrid policies beat raw/calibrated candidate "
                "top-1, the system should keep runtime evidence arbitration as the "
                "baseline before any DPO/RLVR or release escalation."
            ),
            "next_decision": (
                "Keep tool_query, DPO/RLVR, Hugging Face publication, release "
                "tagging, and broad retraining gated until model-heavy outputs beat "
                "this runtime arbitration baseline on held-out cases."
            ),
        },
    }


def write_markdown(report: Mapping[str, Any], path: Path) -> None:
    by_policy = report["summary"]["by_policy"]
    lines = [
        "# Stage A Saved-Output Candidate Arbitration",
        "",
        "Purpose: compare raw candidate top-1, train-derived calibration,",
        "train-selected score-gap gating, and model-visible evidence arbitration",
        "on the saved-output held-out calibration slice.",
        "",
        "## Summary",
        "",
        f"- Run ID: `{report.get('run_id')}`",
        f"- Cases: {report['cases']}",
        f"- Train-selected threshold: `{report.get('train_selected_zero_unsafe_threshold')}`",
        f"- Hidden labels used by arbitration: `{report.get('hidden_labels_used_by_arbitration')}`",
        f"- Best policy names: `{json.dumps(report['summary']['best_policy_names'])}`",
        "",
        "| Policy | Exact | Rows | Trusted candidate | Unsafe candidate trust | Error case IDs |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for policy, summary in by_policy.items():
        lines.append(
            (
                "| {policy} | {exact} | {rows} | {trusted} | {unsafe} | `{errors}` |"
            ).format(
                policy=policy,
                exact=summary["exact"],
                rows=summary["rows"],
                trusted=summary["trusted_candidate"],
                unsafe=summary["trusted_candidate_incorrect"],
                errors=json.dumps(summary["error_case_ids"]),
            )
        )
    lines.extend(
        [
            "",
            "## Rows",
            "",
            "| Case | Target | Raw top | Calibrated top | Evidence gate | Policy | Predicted | Exact |",
            "| --- | --- | --- | --- | --- | --- | --- | ---: |",
        ]
    )
    for row in report["rows"]:
        lines.append(
            (
                "| `{case_id}` | `{target}` | `{raw}` | `{calibrated}` | `{evidence}` | "
                "`{policy}` | `{predicted}` | {exact} |"
            ).format(
                case_id=row.get("case_id"),
                target=row.get("target_pair_label"),
                raw=row.get("raw_top_pair_label"),
                calibrated=row.get("calibrated_top_pair_label"),
                evidence=row.get("evidence_gate_pair"),
                policy=row.get("policy"),
                predicted=row.get("predicted_pair"),
                exact=int(bool(row.get("exact"))),
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
    parser.add_argument("--calibration-report", default=DEFAULT_CALIBRATION_REPORT)
    parser.add_argument("--evidence-targets", default=DEFAULT_EVIDENCE_TARGETS)
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--out-md", default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    report = build_arbitration_report(
        calibration_report=load_json(args.calibration_report),
        evidence_targets=load_jsonl(args.evidence_targets),
        calibration_report_path=args.calibration_report,
        evidence_targets_path=args.evidence_targets,
    )
    if args.out_json:
        write_json(args.out_json, report)
    if args.out_md:
        write_markdown(report, Path(args.out_md))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

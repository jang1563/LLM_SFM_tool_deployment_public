#!/usr/bin/env python3
"""Derive the next Stage A saved-output experiment from compact checkpoints.

This decision checkpoint reads only public-safe compact summaries. It does not
open raw saved predictions, candidate-score JSONL, scheduler logs, model states,
or ignored run directories.
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

from post_training.run_stage_a_sft_smoke_eval import write_json  # noqa: E402


DATASET = "negbiodb_ct_stage_a_saved_output_next_decision_v1"
DEFAULT_READINESS = "post_training/stage_a_saved_prediction_readiness_2026-07-09.json"
DEFAULT_CANDIDATE_GATE_SUMMARIES = (
    "post_training/stage_a_saved_candidate_gate_train_observed_qwen05b_2026-07-09.json",
    "post_training/stage_a_saved_candidate_gate_all_valid_qwen05b_2026-07-09.json",
)
DEFAULT_CANDIDATE_CALIBRATION = (
    "post_training/stage_a_saved_output_candidate_calibration_qwen05b_cayuga_summary_2026-07-10.json"
)
DEFAULT_CANDIDATE_ARBITRATION = "post_training/stage_a_saved_output_candidate_arbitration_2026-07-10.json"


def load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def public_path(path: str | Path) -> str:
    path = Path(path)
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def gate_failure_rows(gate: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = gate.get("rows", ())
    if not isinstance(rows, Sequence):
        return []
    failures: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        if row.get("exact_top1") is True:
            continue
        failures.append(
            {
                "case_id": row.get("case_id"),
                "case_family": row.get("case_family"),
                "target_pair_label": row.get("target_pair_label"),
                "top_pair_label": row.get("top_pair_label"),
                "target_rank": row.get("target_rank"),
                "top_second_gap": row.get("top_second_gap"),
                "fail_closed_exact": row.get("fail_closed_exact"),
            }
        )
    return failures


def summarize_candidate_gates(paths: Sequence[str | Path]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for path in paths:
        payload = load_json(path)
        best = payload.get("best_default_zero_unsafe_report")
        if not isinstance(best, Mapping):
            raise ValueError(f"{path} has no best_default_zero_unsafe_report object")
        summary = payload.get("summary", {})
        if not isinstance(summary, Mapping):
            summary = {}
        failures = gate_failure_rows(payload)
        target_counter = Counter(str(row.get("target_pair_label")) for row in failures)
        top_counter = Counter(str(row.get("top_pair_label")) for row in failures)
        summaries.append(
            {
                "path": public_path(path),
                "sha256": sha256_file(path),
                "run_id": payload.get("run_id"),
                "model": payload.get("model"),
                "candidate_policy": payload.get("candidate_policy"),
                "cases": int(payload.get("cases", 0)),
                "exact_top1": int(summary.get("exact_top1", 0)),
                "mean_target_rank": summary.get("mean_target_rank"),
                "top_pair_counts": dict(summary.get("top_pair_counts", {})),
                "threshold": float(best.get("threshold", 0.0)),
                "trusted": int(best.get("trusted", 0)),
                "trusted_incorrect": int(best.get("trusted_incorrect", 0)),
                "strict_final_correct": int(best.get("strict_final_correct", 0)),
                "strict_final_accuracy": round(float(best.get("strict_final_accuracy", 0.0)), 3),
                "failure_target_pair_counts": dict(sorted(target_counter.items())),
                "failure_top_pair_counts": dict(sorted(top_counter.items())),
                "failure_rows": failures,
            }
        )
    return summaries


def summarize_candidate_calibration(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    payload = load_json(path)
    raw = payload.get("raw_heldout_summary", {})
    calibrated = payload.get("calibrated_heldout_summary", {})
    gate = payload.get("train_selected_gate_report", {})
    if not isinstance(raw, Mapping) or not isinstance(calibrated, Mapping) or not isinstance(gate, Mapping):
        raise ValueError(f"{path} is missing expected calibration summary objects")
    return {
        "path": public_path(path),
        "sha256": sha256_file(path),
        "run_id": payload.get("run_id"),
        "model": payload.get("model"),
        "candidate_policy": payload.get("candidate_policy"),
        "candidate_target_format": payload.get("candidate_target_format"),
        "calibration_mode": payload.get("calibration_mode"),
        "cases": int(calibrated.get("cases", raw.get("cases", 0))),
        "raw_heldout_exact_top1": int(raw.get("exact_top1", 0)),
        "calibrated_heldout_exact_top1": int(calibrated.get("exact_top1", 0)),
        "train_selected_gate_strict_final_correct": int(gate.get("strict_final_correct", 0)),
        "train_selected_gate_trusted_incorrect": int(gate.get("trusted_incorrect", 0)),
        "train_selected_gate_trusted": int(gate.get("trusted", 0)),
        "train_selected_zero_unsafe_threshold": payload.get("train_selected_zero_unsafe_threshold"),
        "boundary": payload.get("boundary"),
    }


def policy_summary(summary: Mapping[str, Any], policy: str) -> Mapping[str, Any]:
    by_policy = summary.get("by_policy", {})
    if not isinstance(by_policy, Mapping):
        return {}
    row = by_policy.get(policy, {})
    return row if isinstance(row, Mapping) else {}


def summarize_candidate_arbitration(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    payload = load_json(path)
    summary = payload.get("summary", {})
    if not isinstance(summary, Mapping):
        raise ValueError(f"{path} is missing summary object")
    by_policy = summary.get("by_policy", {})
    if not isinstance(by_policy, Mapping):
        raise ValueError(f"{path} is missing summary.by_policy object")
    policy_summaries: dict[str, dict[str, Any]] = {}
    for policy, row in by_policy.items():
        if not isinstance(row, Mapping):
            continue
        policy_summaries[str(policy)] = {
            "exact": int(row.get("exact", 0)),
            "rows": int(row.get("rows", 0)),
            "trusted_candidate": int(row.get("trusted_candidate", 0)),
            "trusted_candidate_incorrect": int(row.get("trusted_candidate_incorrect", 0)),
            "error_case_ids": list(row.get("error_case_ids", ())),
        }
    raw = policy_summary(summary, "raw_candidate_top1")
    calibrated = policy_summary(summary, "calibrated_candidate_top1")
    train_gate = policy_summary(summary, "train_selected_score_gap_gate")
    evidence = policy_summary(summary, "evidence_gate_override")
    hybrid = policy_summary(summary, "hybrid_evidence_then_train_gate")
    return {
        "path": public_path(path),
        "sha256": sha256_file(path),
        "run_id": payload.get("run_id"),
        "cases": int(payload.get("cases", 0)),
        "model_visible_evidence_gate": bool(payload.get("model_visible_evidence_gate", False)),
        "hidden_labels_used_by_arbitration": bool(payload.get("hidden_labels_used_by_arbitration", True)),
        "best_policy_names": list(summary.get("best_policy_names", ())),
        "policy_summaries": policy_summaries,
        "raw_candidate_exact": int(raw.get("exact", 0)),
        "calibrated_candidate_exact": int(calibrated.get("exact", 0)),
        "train_selected_score_gap_exact": int(train_gate.get("exact", 0)),
        "evidence_gate_exact": int(evidence.get("exact", 0)),
        "hybrid_evidence_then_train_gate_exact": int(hybrid.get("exact", 0)),
        "raw_candidate_trusted_incorrect": int(raw.get("trusted_candidate_incorrect", 0)),
        "calibrated_candidate_trusted_incorrect": int(
            calibrated.get("trusted_candidate_incorrect", 0)
        ),
        "train_selected_score_gap_trusted_incorrect": int(
            train_gate.get("trusted_candidate_incorrect", 0)
        ),
    }


def derive_bottleneck(
    *,
    readiness: Mapping[str, Any],
    gate_summaries: Sequence[Mapping[str, Any]],
    calibration_summary: Mapping[str, Any] | None = None,
    arbitration_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    decision = readiness.get("decision", {})
    if not isinstance(decision, Mapping):
        decision = {}
    best_output = decision.get("best_real_saved_output")
    best_gate = decision.get("best_saved_candidate_gate")
    if not isinstance(best_output, Mapping):
        best_output = {}
    if not isinstance(best_gate, Mapping):
        best_gate = {}
    top_pair_counts = Counter()
    failure_target_counts = Counter()
    failure_top_counts = Counter()
    for gate in gate_summaries:
        top_pair_counts.update(gate.get("top_pair_counts", {}))
        failure_target_counts.update(gate.get("failure_target_pair_counts", {}))
        failure_top_counts.update(gate.get("failure_top_pair_counts", {}))
    active = "action_status_candidate_selection_collapse"
    if int(best_output.get("parse_error_count", 0)) > 0:
        active = "canonical_json_or_parse_still_unstable"
    elif int(best_gate.get("trusted_incorrect", 0)) > 0:
        active = "unsafe_candidate_gate_calibration"
    elif int(best_gate.get("strict_final_correct", 0)) <= 3:
        active = "narrow_fail_closed_coverage_under_citationless"
    if arbitration_summary is not None:
        candidate_best = max(
            int(arbitration_summary.get("raw_candidate_exact", 0)),
            int(arbitration_summary.get("calibrated_candidate_exact", 0)),
            int(arbitration_summary.get("train_selected_score_gap_exact", 0)),
        )
        runtime_best = max(
            int(arbitration_summary.get("evidence_gate_exact", 0)),
            int(arbitration_summary.get("hybrid_evidence_then_train_gate_exact", 0)),
        )
        if runtime_best > candidate_best:
            active = "runtime_evidence_arbitration_beats_saved_output_candidates"
    return {
        "active_bottleneck": active,
        "best_real_saved_output_passed": int(best_output.get("passed", 0)),
        "best_real_saved_output_mean_score": best_output.get("mean_score"),
        "best_candidate_gate_strict_final_correct": int(best_gate.get("strict_final_correct", 0)),
        "best_candidate_gate_trusted_incorrect": int(best_gate.get("trusted_incorrect", 0)),
        "candidate_top_pair_counts": dict(sorted(top_pair_counts.items())),
        "candidate_failure_target_pair_counts": dict(sorted(failure_target_counts.items())),
        "candidate_failure_top_pair_counts": dict(sorted(failure_top_counts.items())),
        "calibration_raw_heldout_exact": (
            None if calibration_summary is None else calibration_summary.get("raw_heldout_exact_top1")
        ),
        "calibration_calibrated_heldout_exact": (
            None
            if calibration_summary is None
            else calibration_summary.get("calibrated_heldout_exact_top1")
        ),
        "calibration_train_gate_strict_final_correct": (
            None
            if calibration_summary is None
            else calibration_summary.get("train_selected_gate_strict_final_correct")
        ),
        "arbitration_evidence_gate_exact": (
            None if arbitration_summary is None else arbitration_summary.get("evidence_gate_exact")
        ),
        "arbitration_hybrid_exact": (
            None
            if arbitration_summary is None
            else arbitration_summary.get("hybrid_evidence_then_train_gate_exact")
        ),
        "arbitration_best_policy_names": (
            [] if arbitration_summary is None else arbitration_summary.get("best_policy_names", [])
        ),
    }


def build_decision(
    readiness: Mapping[str, Any],
    bottleneck: Mapping[str, Any],
    *,
    arbitration_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    scorecard = readiness.get("heldout_scorecard", {})
    if not isinstance(scorecard, Mapping):
        scorecard = {}
    citationless = scorecard.get("citationless_runtime_action", {})
    runtime = scorecard.get("runtime_gate_full", {})
    citationless_passed = int(citationless.get("passed", 0)) if isinstance(citationless, Mapping) else 0
    runtime_passed = int(runtime.get("passed", 0)) if isinstance(runtime, Mapping) else 0
    strict_final = int(bottleneck.get("best_candidate_gate_strict_final_correct", 0))
    unsafe = int(bottleneck.get("best_candidate_gate_trusted_incorrect", 0))
    blockers = [
        "raw saved-output top-1 remains 0/5",
        f"best fail-closed candidate gate is {strict_final}/5, below citationless {citationless_passed}/5",
        f"runtime evidence gate remains {runtime_passed}/5 and is still required",
    ]
    if unsafe:
        blockers.append("candidate gate has unsafe trusted rows")
    if arbitration_summary is not None:
        runtime_baseline = max(
            int(arbitration_summary.get("evidence_gate_exact", 0)),
            int(arbitration_summary.get("hybrid_evidence_then_train_gate_exact", 0)),
        )
        candidate_best = max(
            int(arbitration_summary.get("raw_candidate_exact", 0)),
            int(arbitration_summary.get("calibrated_candidate_exact", 0)),
            int(arbitration_summary.get("train_selected_score_gap_exact", 0)),
        )
        cases = int(arbitration_summary.get("cases", 0))
        return {
            "selected_next_step": "meet_or_beat_runtime_evidence_arbitration_baseline",
            "why": (
                "The targeted calibration probe is complete: train-derived calibration "
                "improves held-out candidate top-1, but still underperforms "
                "model-visible evidence/hybrid arbitration. The next model-heavy "
                "checkpoint should meet or beat that runtime baseline rather than start "
                "DPO/RLVR."
            ),
            "minimum_success_criteria_for_next_cayuga_checkpoint": {
                "candidate_or_model_policy_exact_min": runtime_baseline,
                "trusted_candidate_incorrect": 0,
                "hidden_labels_used_by_arbitration": False,
                "raw_predictions_remain_uncommitted": True,
            },
            "keep_gated": [
                "tool_query",
                "DPO/RLVR",
                "Hugging Face publication",
                "release tagging",
                "broad retraining",
            ],
            "blockers": [
                (
                    "best saved-output candidate/calibration policy is "
                    f"{candidate_best}/{cases}, below runtime evidence/hybrid "
                    f"arbitration at {runtime_baseline}/{cases}"
                ),
                "score-gap gating avoids unsafe candidate trust but loses coverage",
                "runtime evidence arbitration is the baseline any optimizer must meet or beat",
            ],
            "next_artifacts_required": [
                "compact saved-output summary",
                "compact candidate calibration summary",
                "compact candidate arbitration summary",
                "updated saved-output next-decision report",
            ],
        }
    return {
        "selected_next_step": "targeted_action_status_calibration_probe",
        "why": (
            "The saved-output path has moved past parse/tool/query failures, but "
            "candidate top-1 still collapses to ground/supported and the score-gap "
            "gate only safely trusts the supported row."
        ),
        "minimum_success_criteria_for_next_cayuga_checkpoint": {
            "real_saved_output_passed_must_exceed_collapse": True,
            "fail_closed_gate_trusted_incorrect": 0,
            "fail_closed_gate_strict_final_correct_min": citationless_passed + 1,
            "raw_predictions_remain_uncommitted": True,
        },
        "keep_gated": [
            "tool_query",
            "DPO/RLVR",
            "Hugging Face publication",
            "release tagging",
            "broad retraining",
        ],
        "blockers": blockers,
        "next_artifacts_required": [
            "compact saved-output summary",
            "compact saved-candidate gate summary",
            "updated saved-prediction readiness report",
        ],
    }


def build_report(
    *,
    readiness_path: str | Path,
    candidate_gate_paths: Sequence[str | Path],
    candidate_calibration_path: str | Path | None = DEFAULT_CANDIDATE_CALIBRATION,
    candidate_arbitration_path: str | Path | None = DEFAULT_CANDIDATE_ARBITRATION,
) -> dict[str, Any]:
    readiness = load_json(readiness_path)
    gate_summaries = summarize_candidate_gates(candidate_gate_paths)
    calibration_summary = summarize_candidate_calibration(candidate_calibration_path)
    arbitration_summary = summarize_candidate_arbitration(candidate_arbitration_path)
    bottleneck = derive_bottleneck(
        readiness=readiness,
        gate_summaries=gate_summaries,
        calibration_summary=calibration_summary,
        arbitration_summary=arbitration_summary,
    )
    decision = build_decision(readiness, bottleneck, arbitration_summary=arbitration_summary)
    return {
        "dataset": DATASET,
        "input_artifacts": {
            "readiness": {
                "path": public_path(readiness_path),
                "sha256": sha256_file(readiness_path),
            },
            "candidate_gates": [
                {"path": gate["path"], "sha256": gate["sha256"]} for gate in gate_summaries
            ],
            "candidate_calibration": (
                None
                if calibration_summary is None
                else {"path": calibration_summary["path"], "sha256": calibration_summary["sha256"]}
            ),
            "candidate_arbitration": (
                None
                if arbitration_summary is None
                else {"path": arbitration_summary["path"], "sha256": arbitration_summary["sha256"]}
            ),
        },
        "raw_model_outputs_used": False,
        "raw_run_folders_used": False,
        "readiness_decision": readiness.get("decision", {}),
        "candidate_gate_summaries": gate_summaries,
        "candidate_calibration_summary": calibration_summary,
        "candidate_arbitration_summary": arbitration_summary,
        "bottleneck": bottleneck,
        "decision": decision,
        "scientific_readout": {
            "diagnostic_question": (
                "Given current compact Cayuga saved-output and candidate-gate results, "
                "what is the next Stage A experiment that advances the benchmark without "
                "premature optimization or publication escalation?"
            ),
            "interpretation_rule": (
                "Choose the next experiment only from public-safe compact artifacts. "
                "Do not treat a narrow zero-unsafe gate as sufficient readiness."
            ),
            "next_decision": decision["selected_next_step"],
        },
    }


def write_markdown(report: Mapping[str, Any], path: Path) -> None:
    bottleneck = report["bottleneck"]
    decision = report["decision"]
    lines = [
        "# Stage A Saved-Output Next Decision",
        "",
        "Purpose: choose the next Stage A saved-output experiment from compact",
        "public-safe readiness and candidate-gate checkpoints.",
        "",
        "## Bottleneck",
        "",
        f"- Active bottleneck: `{bottleneck['active_bottleneck']}`",
        f"- Best raw saved-output pass count: {bottleneck['best_real_saved_output_passed']}/5",
        (
            "- Best fail-closed candidate gate: "
            f"{bottleneck['best_candidate_gate_strict_final_correct']}/5 strict final, "
            f"{bottleneck['best_candidate_gate_trusted_incorrect']} unsafe trust"
        ),
        f"- Candidate top-pair counts: `{json.dumps(bottleneck['candidate_top_pair_counts'], sort_keys=True)}`",
        (
            "- Candidate failure targets: "
            f"`{json.dumps(bottleneck['candidate_failure_target_pair_counts'], sort_keys=True)}`"
        ),
    ]
    if report.get("candidate_calibration_summary"):
        calibration = report["candidate_calibration_summary"]
        lines.append(
            (
                "- Calibration held-out exact: "
                f"raw {calibration['raw_heldout_exact_top1']}/{calibration['cases']}, "
                f"calibrated {calibration['calibrated_heldout_exact_top1']}/{calibration['cases']}, "
                "train-selected gate "
                f"{calibration['train_selected_gate_strict_final_correct']}/{calibration['cases']}"
            )
        )
    if report.get("candidate_arbitration_summary"):
        arbitration = report["candidate_arbitration_summary"]
        lines.extend(
            [
                (
                    "- Arbitration exact: "
                    f"raw {arbitration['raw_candidate_exact']}/{arbitration['cases']}, "
                    f"calibrated {arbitration['calibrated_candidate_exact']}/{arbitration['cases']}, "
                    f"score-gap {arbitration['train_selected_score_gap_exact']}/{arbitration['cases']}, "
                    f"evidence {arbitration['evidence_gate_exact']}/{arbitration['cases']}, "
                    f"hybrid {arbitration['hybrid_evidence_then_train_gate_exact']}/{arbitration['cases']}"
                ),
                (
                    "- Hidden labels used by arbitration: "
                    f"`{arbitration['hidden_labels_used_by_arbitration']}`"
                ),
            ]
        )
    lines.extend(
        [
        "",
        "## Decision",
        "",
        f"- Selected next step: `{decision['selected_next_step']}`",
        f"- Why: {decision['why']}",
        "",
        "Keep gated:",
        ]
    )
    for item in decision["keep_gated"]:
        lines.append(f"- `{item}`")
    lines.extend(["", "Minimum success criteria for the next Cayuga checkpoint:"])
    for key, value in decision["minimum_success_criteria_for_next_cayuga_checkpoint"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "Artifact policy: raw saved predictions, candidate-score JSONL, scheduler logs,",
            "model state, and ignored run folders stay uncommitted.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--readiness", default=DEFAULT_READINESS)
    parser.add_argument(
        "--candidate-gate-summary",
        action="append",
        default=None,
        help="Compact public saved-candidate fail-closed gate JSON. Repeatable.",
    )
    parser.add_argument("--candidate-calibration", default=DEFAULT_CANDIDATE_CALIBRATION)
    parser.add_argument("--candidate-arbitration", default=DEFAULT_CANDIDATE_ARBITRATION)
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--out-md", default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    candidate_gate_paths = args.candidate_gate_summary or list(DEFAULT_CANDIDATE_GATE_SUMMARIES)
    report = build_report(
        readiness_path=args.readiness,
        candidate_gate_paths=candidate_gate_paths,
        candidate_calibration_path=args.candidate_calibration,
        candidate_arbitration_path=args.candidate_arbitration,
    )
    if args.out_json:
        write_json(args.out_json, report)
    if args.out_md:
        write_markdown(report, Path(args.out_md))
    stdout_report = {
        "dataset": report["dataset"],
        "raw_model_outputs_used": report["raw_model_outputs_used"],
        "raw_run_folders_used": report["raw_run_folders_used"],
        "bottleneck": report["bottleneck"],
        "decision": report["decision"],
    }
    print(json.dumps(stdout_report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

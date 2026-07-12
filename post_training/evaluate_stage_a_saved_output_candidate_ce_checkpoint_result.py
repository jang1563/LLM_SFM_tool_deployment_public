#!/usr/bin/env python3
"""Build a compact result checkpoint for the candidate-CE saved-output run.

Inputs are compact analyzer reports derived from a private Cayuga run. This
script does not read raw saved predictions, candidate-score JSONL, scheduler
logs, model state, or ignored run folders.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from post_training.run_stage_a_sft_smoke_eval import write_json  # noqa: E402


DATASET = "negbiodb_ct_stage_a_saved_output_candidate_ce_checkpoint_result_v1"


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


def source_digest(path: str | Path, *, role: str) -> dict[str, str]:
    return {
        "role": role,
        "sha256": sha256_file(path),
        "storage": "private_compact_analyzer_output_not_public_source",
    }


def policy_summary(arbitration: Mapping[str, Any], policy: str) -> Mapping[str, Any]:
    summary = arbitration.get("summary", {})
    if not isinstance(summary, Mapping):
        return {}
    by_policy = summary.get("by_policy", {})
    if not isinstance(by_policy, Mapping):
        return {}
    row = by_policy.get(policy, {})
    return row if isinstance(row, Mapping) else {}


def compact_policy(arbitration: Mapping[str, Any], policy: str) -> dict[str, Any]:
    row = policy_summary(arbitration, policy)
    return {
        "policy": policy,
        "exact": int(row.get("exact", 0)),
        "rows": int(row.get("rows", 0)),
        "trusted_candidate": int(row.get("trusted_candidate", 0)),
        "trusted_candidate_incorrect": int(row.get("trusted_candidate_incorrect", 0)),
        "error_case_ids": list(row.get("error_case_ids", ())),
        "by_predicted_pair": dict(row.get("by_predicted_pair", {})),
    }


def build_report(
    *,
    calibration_report: Mapping[str, Any],
    field_report: Mapping[str, Any],
    arbitration_report: Mapping[str, Any],
    calibration_report_path: str | Path,
    field_report_path: str | Path,
    arbitration_report_path: str | Path,
) -> dict[str, Any]:
    raw_heldout = calibration_report.get("raw_heldout_summary", {})
    calibrated_heldout = calibration_report.get("calibrated_heldout_summary", {})
    train = calibration_report.get("train_summary", {})
    train_gate = calibration_report.get("train_selected_gate_report", {})
    field_summary = field_report.get("summary", {})
    field_diagnostic = field_summary.get("field_diagnostic", {})
    policies = [
        "raw_candidate_top1",
        "calibrated_candidate_top1",
        "train_selected_score_gap_gate",
        "evidence_gate_override",
        "hybrid_evidence_then_train_gate",
    ]
    return {
        "dataset": DATASET,
        "run_id": calibration_report.get("run_id"),
        "model": calibration_report.get("model"),
        "checkpoint_name": "candidate_ce_action_status_pair_field_readout",
        "checkpoint_objective": {
            "training_family": "supervised_sft_candidate_routing",
            "candidate_ce_mode": "pair_plus_field",
            "candidate_target_format": calibration_report.get("candidate_target_format"),
            "candidate_policy": calibration_report.get("candidate_policy"),
            "not_dpo_or_rlvr": True,
        },
        "input_compact_reports": {
            "candidate_calibration": source_digest(
                calibration_report_path,
                role="candidate-CE train-derived calibration compact report",
            ),
            "candidate_field": source_digest(
                field_report_path,
                role="candidate-CE field-wise diagnostic compact report",
            ),
            "candidate_arbitration": source_digest(
                arbitration_report_path,
                role="candidate-CE runtime-vs-candidate arbitration compact report",
            ),
        },
        "result": {
            "train_candidate_top1": {
                "exact": int(train.get("exact_top1", 0)),
                "rows": int(train.get("cases", 0)),
                "top_pair_counts": dict(train.get("top_pair_counts", {})),
                "mean_target_rank": train.get("mean_target_rank"),
            },
            "raw_heldout_candidate_top1": {
                "exact": int(raw_heldout.get("exact_top1", 0)),
                "rows": int(raw_heldout.get("cases", 0)),
                "top_pair_counts": dict(raw_heldout.get("top_pair_counts", {})),
                "mean_target_rank": raw_heldout.get("mean_target_rank"),
            },
            "calibrated_heldout_candidate_top1": {
                "exact": int(calibrated_heldout.get("exact_top1", 0)),
                "rows": int(calibrated_heldout.get("cases", 0)),
                "top_pair_counts": dict(calibrated_heldout.get("top_pair_counts", {})),
                "mean_target_rank": calibrated_heldout.get("mean_target_rank"),
            },
            "train_selected_gate": {
                "strict_final_correct": int(train_gate.get("strict_final_correct", 0)),
                "trusted": int(train_gate.get("trusted", 0)),
                "trusted_incorrect": int(train_gate.get("trusted_incorrect", 0)),
                "threshold": train_gate.get("threshold"),
            },
            "field_diagnostic": {
                "exact_top1": int(field_summary.get("exact_top1", 0)),
                "rows": int(field_summary.get("cases", 0)),
                "action_top1": int(field_diagnostic.get("action_top1", 0)),
                "evidence_status_top1": int(field_diagnostic.get("evidence_status_top1", 0)),
                "field_rank_patterns": dict(field_diagnostic.get("field_rank_patterns", {})),
                "top_pair_counts": dict(field_summary.get("top_pair_counts", {})),
            },
            "runtime_arbitration": {
                "policies": [compact_policy(arbitration_report, policy) for policy in policies],
                "best_policy_names": list(
                    arbitration_report.get("summary", {}).get("best_policy_names", ())
                )
                if isinstance(arbitration_report.get("summary"), Mapping)
                else [],
            },
        },
        "decision": {
            "passes_meet_or_beat_gate": False,
            "selected_next_step": "keep_runtime_evidence_arbitration_baseline",
            "interpretation": (
                "Candidate CE pair+field supervision shifted the candidate prior "
                "toward verify/insufficient but did not improve held-out exact "
                "top-1 or meet the runtime arbitration baseline."
            ),
            "next_research_move": (
                "Do not escalate to DPO/RLVR, tool_query, Hugging Face publication, "
                "or release tagging. Treat standalone candidate-routing SFT as "
                "insufficient and keep runtime evidence arbitration as the baseline "
                "while designing evidence-conditioned or larger-slice candidate "
                "routing checks."
            ),
        },
        "public_safety_contract": {
            "raw_prediction_jsonl_read": False,
            "raw_candidate_score_jsonl_read": False,
            "scheduler_logs_read": False,
            "model_state_read": False,
            "ignored_run_folder_read": False,
            "raw_artifacts_committed": False,
        },
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    result = report["result"]
    field = result["field_diagnostic"]
    policies = result["runtime_arbitration"]["policies"]
    lines = [
        "# Stage A Saved-Output Candidate-CE Checkpoint Result",
        "",
        "Purpose: compact public-safe result for the candidate-CE Cayuga checkpoint.",
        "",
        "## Result",
        "",
        (
            f"- Train candidate top-1: {result['train_candidate_top1']['exact']}/"
            f"{result['train_candidate_top1']['rows']}, top-pair counts "
            f"`{json.dumps(result['train_candidate_top1']['top_pair_counts'], sort_keys=True)}`"
        ),
        (
            f"- Raw held-out candidate top-1: {result['raw_heldout_candidate_top1']['exact']}/"
            f"{result['raw_heldout_candidate_top1']['rows']}, top-pair counts "
            f"`{json.dumps(result['raw_heldout_candidate_top1']['top_pair_counts'], sort_keys=True)}`"
        ),
        (
            f"- Calibrated held-out candidate top-1: "
            f"{result['calibrated_heldout_candidate_top1']['exact']}/"
            f"{result['calibrated_heldout_candidate_top1']['rows']}, top-pair counts "
            f"`{json.dumps(result['calibrated_heldout_candidate_top1']['top_pair_counts'], sort_keys=True)}`"
        ),
        (
            f"- Field diagnostic: exact {field['exact_top1']}/{field['rows']}, "
            f"action {field['action_top1']}/{field['rows']}, "
            f"status {field['evidence_status_top1']}/{field['rows']}; patterns "
            f"`{json.dumps(field['field_rank_patterns'], sort_keys=True)}`"
        ),
        (
            f"- Train-selected gate: {result['train_selected_gate']['strict_final_correct']}/"
            f"{result['raw_heldout_candidate_top1']['rows']} strict-final correct, "
            f"trusted incorrect {result['train_selected_gate']['trusted_incorrect']}."
        ),
        "",
        "## Arbitration",
        "",
        "| Policy | Exact | Rows | Trusted candidate | Trusted incorrect |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in policies:
        lines.append(
            "| {policy} | {exact} | {rows} | {trusted_candidate} | {trusted_candidate_incorrect} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Passes meet-or-beat gate: `{report['decision']['passes_meet_or_beat_gate']}`",
            f"- Selected next step: `{report['decision']['selected_next_step']}`",
            f"- Interpretation: {report['decision']['interpretation']}",
            f"- Next research move: {report['decision']['next_research_move']}",
            "",
            "Public-safety contract: raw saved predictions, candidate-score JSONL,",
            "scheduler logs, model state, and ignored run folders were not copied",
            "into this checkpoint.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-report", required=True)
    parser.add_argument("--field-report", required=True)
    parser.add_argument("--arbitration-report", required=True)
    parser.add_argument(
        "--out-json",
        default="post_training/stage_a_saved_output_candidate_ce_checkpoint_result_2026-07-10.json",
    )
    parser.add_argument(
        "--out-md",
        default="post_training/STAGE_A_SAVED_OUTPUT_CANDIDATE_CE_CHECKPOINT_RESULT_2026-07-10.md",
    )
    args = parser.parse_args()

    report = build_report(
        calibration_report=load_json(args.calibration_report),
        field_report=load_json(args.field_report),
        arbitration_report=load_json(args.arbitration_report),
        calibration_report_path=args.calibration_report,
        field_report_path=args.field_report,
        arbitration_report_path=args.arbitration_report,
    )
    write_json(Path(args.out_json), report)
    Path(args.out_md).write_text(render_markdown(report))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

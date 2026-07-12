#!/usr/bin/env python3
"""Summarize the current Stage A saved-output checkpoint diagnosis.

This public-safe checkpoint reads only compact JSON summaries that are already
eligible for the public release manifest. It does not read raw prediction JSONL,
candidate-score JSONL, scheduler logs, model state, or ignored run folders.
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


DATASET = "negbiodb_ct_stage_a_saved_output_checkpoint_diagnosis_v1"
DEFAULT_SAME_MODEL_TARGET_FORMAT = (
    "post_training/stage_a_saved_output_same_model_target_format_qwen05b_cayuga_summary_2026-07-10.json"
)
DEFAULT_CANDIDATE_RANK = (
    "post_training/stage_a_saved_output_candidate_rank_qwen05b_cayuga_summary_2026-07-10.json"
)
DEFAULT_CANDIDATE_CALIBRATION = (
    "post_training/stage_a_saved_output_candidate_calibration_qwen05b_cayuga_summary_2026-07-10.json"
)
DEFAULT_CANDIDATE_ARBITRATION = "post_training/stage_a_saved_output_candidate_arbitration_2026-07-10.json"
DEFAULT_NEXT_DECISION = "post_training/stage_a_saved_output_next_decision_2026-07-10.json"
DEFAULT_MEET_OR_BEAT_GATE = "post_training/stage_a_saved_output_meet_or_beat_gate_2026-07-10.json"


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
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def input_artifact(path: str | Path, *, role: str) -> dict[str, Any]:
    return {
        "path": public_path(path),
        "role": role,
        "sha256": sha256_file(path),
    }


def policy_row(arbitration: Mapping[str, Any], policy: str) -> Mapping[str, Any]:
    summary = arbitration.get("summary", {})
    if not isinstance(summary, Mapping):
        return {}
    by_policy = summary.get("by_policy", {})
    if not isinstance(by_policy, Mapping):
        return {}
    row = by_policy.get(policy, {})
    return row if isinstance(row, Mapping) else {}


def compact_policy(arbitration: Mapping[str, Any], policy: str) -> dict[str, Any]:
    row = policy_row(arbitration, policy)
    return {
        "policy": policy,
        "exact": int(row.get("exact", 0)),
        "rows": int(row.get("rows", 0)),
        "trusted_candidate": int(row.get("trusted_candidate", 0)),
        "trusted_candidate_incorrect": int(row.get("trusted_candidate_incorrect", 0)),
        "error_case_ids": list(row.get("error_case_ids", ())),
    }


def compact_same_model_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    runs = payload.get("runs", {})
    if not isinstance(runs, Mapping):
        raise ValueError("same-model target-format summary missing runs object")
    full = runs.get("full", {})
    if not isinstance(full, Mapping):
        raise ValueError("same-model target-format summary missing runs.full object")
    base = full.get("base_heldout", {})
    trained = full.get("trained_heldout", {})
    delta = full.get("delta", {})
    if not isinstance(base, Mapping) or not isinstance(trained, Mapping) or not isinstance(delta, Mapping):
        raise ValueError("same-model target-format summary has malformed full-format objects")
    return {
        "run_id": payload.get("run_id"),
        "model": payload.get("model"),
        "heldout_examples": int(payload.get("heldout_examples", 0)),
        "training_target_format": "full",
        "base_full_margin_wins": int(base.get("margin_wins", 0)),
        "trained_full_margin_wins": int(trained.get("margin_wins", 0)),
        "pairs": int(trained.get("pairs", base.get("pairs", 0))),
        "base_mean_margin": base.get("mean_margin"),
        "trained_mean_margin": trained.get("mean_margin"),
        "flag_invalid_value_margin": trained.get("flag_invalid_value_margin"),
        "newly_won": int(delta.get("newly_won", 0)),
        "interpretation": (
            "Teacher-forced full-target scoring can be repaired on the 4-case held-out "
            "slice, but this is not a deployment-ready candidate policy."
        ),
    }


def compact_candidate_rank_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    candidate = payload.get("candidate_readout", {})
    gate = payload.get("fail_closed_gate", {})
    if not isinstance(candidate, Mapping) or not isinstance(gate, Mapping):
        raise ValueError("candidate-rank summary missing candidate_readout or fail_closed_gate")
    trained = candidate.get("trained_heldout", {})
    base = candidate.get("base_heldout", {})
    fail_closed = gate.get("trained_best_default_zero_unsafe", {})
    if not isinstance(trained, Mapping) or not isinstance(base, Mapping) or not isinstance(fail_closed, Mapping):
        raise ValueError("candidate-rank summary has malformed held-out objects")
    return {
        "run_id": payload.get("run_id"),
        "candidate_policy": payload.get("training", {}).get("candidate_policy")
        if isinstance(payload.get("training"), Mapping)
        else None,
        "candidate_target_format": payload.get("training", {}).get("candidate_target_format")
        if isinstance(payload.get("training"), Mapping)
        else None,
        "base_exact_top1": int(base.get("exact_top1", 0)),
        "trained_exact_top1": int(trained.get("exact_top1", 0)),
        "pairs": int(trained.get("pairs", base.get("pairs", 0))),
        "trained_top_pair_counts": dict(trained.get("top_pair_counts", {})),
        "mean_target_rank": trained.get("mean_target_rank"),
        "mean_top_target_margin": trained.get("mean_top_target_margin"),
        "fail_closed_strict_final_correct": int(fail_closed.get("strict_final_correct", 0)),
        "fail_closed_trusted_incorrect": int(fail_closed.get("trusted_incorrect", 0)),
        "interpretation": (
            "Full-format teacher-forced margins do not transfer to finite-candidate top-1; "
            "the trained candidate readout collapses to a single flag/invalid_value top pair."
        ),
    }


def compact_calibration_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    raw = payload.get("raw_heldout_summary", {})
    calibrated = payload.get("calibrated_heldout_summary", {})
    gate = payload.get("train_selected_gate_report", {})
    if not isinstance(raw, Mapping) or not isinstance(calibrated, Mapping) or not isinstance(gate, Mapping):
        raise ValueError("candidate-calibration summary missing expected summary objects")
    return {
        "run_id": payload.get("run_id"),
        "calibration_mode": payload.get("calibration_mode"),
        "candidate_policy": payload.get("candidate_policy"),
        "candidate_target_format": payload.get("candidate_target_format"),
        "raw_heldout_exact_top1": int(raw.get("exact_top1", 0)),
        "calibrated_heldout_exact_top1": int(calibrated.get("exact_top1", 0)),
        "pairs": int(calibrated.get("cases", raw.get("cases", 0))),
        "train_selected_gate_strict_final_correct": int(gate.get("strict_final_correct", 0)),
        "train_selected_gate_trusted_incorrect": int(gate.get("trusted_incorrect", 0)),
        "train_selected_zero_unsafe_threshold": payload.get("train_selected_zero_unsafe_threshold"),
        "interpretation": (
            "Train-derived pair-mean centering helps but remains below the runtime "
            "evidence-arbitration baseline."
        ),
    }


def compact_arbitration_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    policies = [
        "raw_candidate_top1",
        "calibrated_candidate_top1",
        "train_selected_score_gap_gate",
        "evidence_gate_override",
        "hybrid_evidence_then_train_gate",
    ]
    return {
        "run_id": payload.get("run_id"),
        "cases": int(payload.get("cases", 0)),
        "hidden_labels_used_by_arbitration": bool(payload.get("hidden_labels_used_by_arbitration", True)),
        "best_policy_names": list(payload.get("summary", {}).get("best_policy_names", ()))
        if isinstance(payload.get("summary"), Mapping)
        else [],
        "policies": [compact_policy(payload, policy) for policy in policies],
    }


def compact_meet_or_beat_gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    requirements = payload.get("requirements", {})
    if not isinstance(requirements, Mapping):
        requirements = {}
    return {
        "passes_gate": bool(payload.get("passes_gate", False)),
        "gate_violations": list(payload.get("gate_violations", ())),
        "candidate_or_model_policy_exact_min": int(requirements.get("candidate_or_model_policy_exact_min", 0)),
        "trusted_candidate_incorrect_max": int(requirements.get("trusted_candidate_incorrect", 0)),
        "raw_predictions_remain_uncommitted": bool(requirements.get("raw_predictions_remain_uncommitted", False)),
    }


def build_report(
    *,
    same_model_target_format_path: str | Path = DEFAULT_SAME_MODEL_TARGET_FORMAT,
    candidate_rank_path: str | Path = DEFAULT_CANDIDATE_RANK,
    candidate_calibration_path: str | Path = DEFAULT_CANDIDATE_CALIBRATION,
    candidate_arbitration_path: str | Path = DEFAULT_CANDIDATE_ARBITRATION,
    next_decision_path: str | Path = DEFAULT_NEXT_DECISION,
    meet_or_beat_gate_path: str | Path = DEFAULT_MEET_OR_BEAT_GATE,
) -> dict[str, Any]:
    same_model = load_json(same_model_target_format_path)
    candidate_rank = load_json(candidate_rank_path)
    calibration = load_json(candidate_calibration_path)
    arbitration = load_json(candidate_arbitration_path)
    next_decision = load_json(next_decision_path)
    meet_or_beat = load_json(meet_or_beat_gate_path)

    decision = next_decision.get("decision", {})
    if not isinstance(decision, Mapping):
        decision = {}
    selected_next_step = decision.get("selected_next_step")
    keep_gated = list(decision.get("keep_gated", ())) if isinstance(decision.get("keep_gated"), list) else []

    report = {
        "dataset": DATASET,
        "current_thesis": (
            "Biology tool-use agents need trainable trajectories plus runtime evidence "
            "arbitration; RLVR/DPO escalation is not justified until compact held-out "
            "checkpoint policies beat deterministic runtime baselines."
        ),
        "non_goals": [
            "pretraining a biology foundation model",
            "generic biomedical QA",
            "unaudited LLM-judge rewards",
            "trusting specialist/SFM confidence without calibration",
        ],
        "source_changes": [],
        "input_artifacts": {
            "same_model_target_format": input_artifact(
                same_model_target_format_path,
                role="teacher-forced target-format compact summary",
            ),
            "candidate_rank": input_artifact(
                candidate_rank_path,
                role="finite-candidate compact rank summary",
            ),
            "candidate_calibration": input_artifact(
                candidate_calibration_path,
                role="train-derived candidate calibration compact summary",
            ),
            "candidate_arbitration": input_artifact(
                candidate_arbitration_path,
                role="runtime-vs-candidate arbitration compact summary",
            ),
            "next_decision": input_artifact(
                next_decision_path,
                role="saved-output next-decision checkpoint",
            ),
            "meet_or_beat_gate": input_artifact(
                meet_or_beat_gate_path,
                role="saved-output acceptance gate",
            ),
        },
        "diagnosis": {
            "teacher_forced_margin": compact_same_model_summary(same_model),
            "finite_candidate_rank": compact_candidate_rank_summary(candidate_rank),
            "candidate_calibration": compact_calibration_summary(calibration),
            "runtime_arbitration": compact_arbitration_summary(arbitration),
            "meet_or_beat_gate": compact_meet_or_beat_gate(meet_or_beat),
        },
        "decision": {
            "selected_next_step": selected_next_step,
            "keep_gated": keep_gated,
            "next_model_checkpoint_requirement": (
                "Any future compact Cayuga policy summary must reach 4/4 held-out exact "
                "with zero trusted-candidate incorrect cases before DPO/RLVR, tool_query, "
                "HF publication, or release tagging reopens."
            ),
            "next_research_move": (
                "Diagnose per-case action/status discrimination under finite-candidate or "
                "runtime-enforced policies; do not treat teacher-forced margin repair as "
                "deployment readiness."
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
    return report


def render_markdown(report: Mapping[str, Any]) -> str:
    diagnosis = report["diagnosis"]
    teacher = diagnosis["teacher_forced_margin"]
    rank = diagnosis["finite_candidate_rank"]
    calibration = diagnosis["candidate_calibration"]
    arbitration = diagnosis["runtime_arbitration"]
    gate = diagnosis["meet_or_beat_gate"]
    policy_rows = arbitration["policies"]

    lines = [
        "# Stage A Saved-Output Checkpoint Diagnosis",
        "",
        "Purpose: compact public-safe status checkpoint for the current Cayuga saved-output path.",
        "",
        "## Thesis",
        "",
        str(report["current_thesis"]),
        "",
        "## Diagnosis",
        "",
        (
            f"- Teacher-forced full-target margin: "
            f"{teacher['trained_full_margin_wins']}/{teacher['pairs']} held-out wins "
            f"(base {teacher['base_full_margin_wins']}/{teacher['pairs']}; "
            f"trained mean margin {teacher['trained_mean_margin']})."
        ),
        (
            f"- Finite-candidate rank: {rank['trained_exact_top1']}/{rank['pairs']} "
            f"held-out top-1, with top-pair counts "
            f"`{json.dumps(rank['trained_top_pair_counts'], sort_keys=True)}`."
        ),
        (
            f"- Train-derived calibration: raw {calibration['raw_heldout_exact_top1']}/"
            f"{calibration['pairs']}, calibrated {calibration['calibrated_heldout_exact_top1']}/"
            f"{calibration['pairs']}, train-selected gate "
            f"{calibration['train_selected_gate_strict_final_correct']}/{calibration['pairs']}."
        ),
        (
            f"- Meet-or-beat gate passes: {gate['passes_gate']} "
            f"with violations `{json.dumps(gate['gate_violations'])}`."
        ),
        "",
        "## Arbitration",
        "",
        "| Policy | Exact | Rows | Trusted candidate | Trusted incorrect |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in policy_rows:
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
            f"- Selected next step: `{report['decision']['selected_next_step']}`",
            f"- Keep gated: `{json.dumps(report['decision']['keep_gated'])}`",
            f"- Next research move: {report['decision']['next_research_move']}",
            "",
            "Public-safety contract: raw saved predictions, candidate-score JSONL, scheduler logs,",
            "model state, and ignored run folders were not read for this checkpoint.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--same-model-target-format", default=DEFAULT_SAME_MODEL_TARGET_FORMAT)
    parser.add_argument("--candidate-rank", default=DEFAULT_CANDIDATE_RANK)
    parser.add_argument("--candidate-calibration", default=DEFAULT_CANDIDATE_CALIBRATION)
    parser.add_argument("--candidate-arbitration", default=DEFAULT_CANDIDATE_ARBITRATION)
    parser.add_argument("--next-decision", default=DEFAULT_NEXT_DECISION)
    parser.add_argument("--meet-or-beat-gate", default=DEFAULT_MEET_OR_BEAT_GATE)
    parser.add_argument("--out-json", default="post_training/stage_a_saved_output_checkpoint_diagnosis_2026-07-10.json")
    parser.add_argument("--out-md", default="post_training/STAGE_A_SAVED_OUTPUT_CHECKPOINT_DIAGNOSIS_2026-07-10.md")
    args = parser.parse_args()

    report = build_report(
        same_model_target_format_path=args.same_model_target_format,
        candidate_rank_path=args.candidate_rank,
        candidate_calibration_path=args.candidate_calibration,
        candidate_arbitration_path=args.candidate_arbitration,
        next_decision_path=args.next_decision,
        meet_or_beat_gate_path=args.meet_or_beat_gate,
    )
    write_json(Path(args.out_json), report)
    Path(args.out_md).write_text(render_markdown(report))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

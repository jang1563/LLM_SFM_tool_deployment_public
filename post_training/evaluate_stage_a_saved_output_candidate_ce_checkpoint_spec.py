#!/usr/bin/env python3
"""Define the next candidate-CE Stage A saved-output checkpoint.

This spec is a compact public-safe follow-up to the non-flag saved-output
checkpoint. It reads only curated summary artifacts and emits the next Cayuga
experiment contract; raw saved predictions, candidate-score JSONL, scheduler
logs, model state, and ignored run folders are not read.
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


DATASET = "negbiodb_ct_stage_a_saved_output_candidate_ce_checkpoint_spec_v1"
DEFAULT_NONFLAG_RESULT = (
    "post_training/stage_a_saved_output_nonflag_checkpoint_result_2026-07-10.json"
)


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


def compact_policy(result: Mapping[str, Any], policy: str) -> dict[str, Any]:
    runtime = result.get("result", {}).get("runtime_arbitration", {})
    policies = runtime.get("policies", []) if isinstance(runtime, Mapping) else []
    for row in policies:
        if isinstance(row, Mapping) and row.get("policy") == policy:
            return {
                "policy": policy,
                "exact": int(row.get("exact", 0)),
                "rows": int(row.get("rows", 0)),
                "trusted_candidate": int(row.get("trusted_candidate", 0)),
                "trusted_candidate_incorrect": int(row.get("trusted_candidate_incorrect", 0)),
            }
    return {
        "policy": policy,
        "exact": 0,
        "rows": 0,
        "trusted_candidate": 0,
        "trusted_candidate_incorrect": 0,
    }


def observed_failure(nonflag_result: Mapping[str, Any]) -> dict[str, Any]:
    result = nonflag_result.get("result", {})
    return {
        "prior_checkpoint": nonflag_result.get("checkpoint_name"),
        "bottleneck": "candidate_selection_not_repaired_by_nonflag_oversampling",
        "raw_heldout_candidate_top1": dict(result.get("raw_heldout_candidate_top1", {})),
        "calibrated_heldout_candidate_top1": dict(
            result.get("calibrated_heldout_candidate_top1", {})
        ),
        "field_diagnostic": dict(result.get("field_diagnostic", {})),
        "runtime_evidence_baseline": compact_policy(nonflag_result, "evidence_gate_override"),
        "hybrid_runtime_baseline": compact_policy(
            nonflag_result,
            "hybrid_evidence_then_train_gate",
        ),
        "failed_candidate_policies": [
            compact_policy(nonflag_result, "raw_candidate_top1"),
            compact_policy(nonflag_result, "calibrated_candidate_top1"),
            compact_policy(nonflag_result, "train_selected_score_gap_gate"),
        ],
        "interpretation": (
            "Simple pair oversampling changed the candidate prior but did not "
            "teach evidence-conditioned action/status selection."
        ),
    }


def next_checkpoint_env() -> dict[str, str]:
    return {
        "RUN_ID": "stage_a_saved_output_candidate_ce_pair_field_qwen05b_cayuga_${JOB_ID}",
        "MODEL_ID": "Qwen/Qwen2.5-0.5B-Instruct",
        "MAX_STEPS": "40",
        "BATCH_SIZE": "1",
        "LR": "1e-5",
        "TRAIN_LAST_LAYERS": "1",
        "TARGET_FORMAT": "full",
        "SCORE_TARGET_FORMATS": "full,action_status_only,action_only,status_only",
        "FOCUS_CHOSEN_PAIRS": "",
        "FOCUS_REPEAT": "1",
        "FOCUS_ONLY": "0",
        "PAIRWISE_MARGIN_WEIGHT": "1",
        "PAIRWISE_MARGIN": "0.05",
        "CANDIDATE_CE_WEIGHT": "1",
        "CANDIDATE_CE_MODE": "pair_plus_field",
        "CANDIDATE_CE_LOGPROB_MODE": "mean",
        "SCORE_BASE_MARGINS": "1",
        "SCORE_TRAIN_MARGINS": "1",
        "SCORE_BASE_CANDIDATES": "1",
        "SCORE_TRAIN_CANDIDATES": "1",
        "SCORE_TRAINED_CANDIDATES": "1",
        "CANDIDATE_POLICY": "train_observed_plus_rejected",
        "CANDIDATE_TARGET_FORMAT": "action_status_only",
        "ALLOW_DOWNLOAD": "0",
    }


def dry_run_command() -> str:
    return (
        "python post_training/run_stage_a_saved_output_calibration_margin_sft.py "
        "--dry-run "
        "--out-dir /tmp/stage_a_saved_output_candidate_ce_dry "
        "--run-id stage_a_saved_output_candidate_ce_dry "
        "--candidate-ce-weight 1 "
        "--candidate-ce-mode pair_plus_field "
        "--candidate-ce-logprob-mode mean "
        "--candidate-policy train_observed_plus_rejected "
        "--candidate-target-format action_status_only "
        "--score-base-margins "
        "--score-train-margins "
        "--score-base-candidates "
        "--score-train-candidates "
        "--score-trained-candidates"
    )


def post_run_compact_commands() -> list[str]:
    return [
        (
            "python post_training/analyze_stage_a_saved_output_candidate_calibration.py "
            "--train-candidates ${OUT_DIR}/train_candidates.jsonl "
            "--heldout-candidates ${OUT_DIR}/candidates.jsonl "
            "--out-json /tmp/stage_a_saved_output_candidate_ce_calibration.json "
            "--out-md /tmp/STAGE_A_SAVED_OUTPUT_CANDIDATE_CE_CALIBRATION.md"
        ),
        (
            "python post_training/analyze_stage_a_saved_output_candidate_fields.py "
            "--candidates ${OUT_DIR}/candidates.jsonl "
            "--out-json /tmp/stage_a_saved_output_candidate_ce_fields.json "
            "--out-md /tmp/STAGE_A_SAVED_OUTPUT_CANDIDATE_CE_FIELDS.md"
        ),
        (
            "python post_training/evaluate_stage_a_saved_output_candidate_arbitration.py "
            "--calibration-report /tmp/stage_a_saved_output_candidate_ce_calibration.json "
            "--out-json /tmp/stage_a_saved_output_candidate_ce_arbitration.json "
            "--out-md /tmp/STAGE_A_SAVED_OUTPUT_CANDIDATE_CE_ARBITRATION.md"
        ),
    ]


def build_report(
    *,
    nonflag_result_path: str | Path = DEFAULT_NONFLAG_RESULT,
) -> dict[str, Any]:
    nonflag_result = load_json(nonflag_result_path)
    return {
        "dataset": DATASET,
        "input_artifacts": {
            "nonflag_checkpoint_result": {
                "path": public_path(nonflag_result_path),
                "role": "completed non-flag saved-output compact checkpoint",
                "sha256": sha256_file(nonflag_result_path),
            }
        },
        "observed_failure": observed_failure(nonflag_result),
        "next_checkpoint": {
            "name": "candidate_ce_action_status_pair_field_readout",
            "runner": "post_training/run_stage_a_saved_output_calibration_margin_sft_cayuga.sbatch",
            "question": (
                "Does an explicit listwise candidate CE objective over "
                "action/status candidates beat simple oversampling while "
                "preserving teacher-forced margin repair?"
            ),
            "objective_change": (
                "Add supervised candidate-routing pressure: pair CE selects "
                "the exact action/status candidate and field CE selects the "
                "action and evidence_status marginals."
            ),
            "dry_run_command": dry_run_command(),
            "env": next_checkpoint_env(),
            "submit_template": (
                "sbatch --account=<allocation> --partition=<gpu-partition> --gres=gpu:1 "
                "--export=ALL,WORK=$PWD,<env-overrides> "
                "post_training/run_stage_a_saved_output_calibration_margin_sft_cayuga.sbatch"
            ),
            "post_run_compact_commands": post_run_compact_commands(),
        },
        "acceptance_gate": {
            "candidate_or_model_policy_exact_min": 4,
            "trusted_candidate_incorrect_max": 0,
            "hidden_labels_used_by_arbitration": False,
            "raw_predictions_remain_uncommitted": True,
            "required_readouts": [
                "train_candidate_summary",
                "heldout_candidate_summary",
                "field_diagnostic",
                "runtime_arbitration",
                "meet_or_beat_gate",
            ],
        },
        "keep_gated": [
            "tool_query",
            "DPO/RLVR",
            "Hugging Face publication",
            "release tagging",
            "broad retraining",
        ],
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
    failure = report["observed_failure"]
    checkpoint = report["next_checkpoint"]
    env = checkpoint["env"]
    lines = [
        "# Stage A Saved-Output Candidate-CE Checkpoint Spec",
        "",
        "Purpose: define the next Cayuga checkpoint after non-flag balancing failed.",
        "",
        "## Observed Failure",
        "",
        f"- Bottleneck: `{failure['bottleneck']}`",
        (
            f"- Raw held-out candidate top-1: "
            f"{failure['raw_heldout_candidate_top1']['exact']}/"
            f"{failure['raw_heldout_candidate_top1']['rows']}"
        ),
        (
            f"- Calibrated held-out candidate top-1: "
            f"{failure['calibrated_heldout_candidate_top1']['exact']}/"
            f"{failure['calibrated_heldout_candidate_top1']['rows']}"
        ),
        (
            f"- Field diagnostic: exact {failure['field_diagnostic']['exact_top1']}/"
            f"{failure['field_diagnostic']['rows']}, action "
            f"{failure['field_diagnostic']['action_top1']}/"
            f"{failure['field_diagnostic']['rows']}, status "
            f"{failure['field_diagnostic']['evidence_status_top1']}/"
            f"{failure['field_diagnostic']['rows']}"
        ),
        f"- Interpretation: {failure['interpretation']}",
        "",
        "## Next Checkpoint",
        "",
        f"- Name: `{checkpoint['name']}`",
        f"- Runner: `{checkpoint['runner']}`",
        f"- Question: {checkpoint['question']}",
        f"- Objective change: {checkpoint['objective_change']}",
        "",
        "Dry-run preflight:",
        "",
        "```bash",
        checkpoint["dry_run_command"],
        "```",
        "",
        "Environment overrides:",
        "",
        "```bash",
    ]
    for key, value in env.items():
        lines.append(f"export {key}='{value}'")
    lines.extend(
        [
            "```",
            "",
            "Post-run compact summaries:",
            "",
        ]
    )
    for command in checkpoint["post_run_compact_commands"]:
        lines.append(f"- `{command}`")
    gate = report["acceptance_gate"]
    lines.extend(
        [
            "",
            "## Acceptance Gate",
            "",
            f"- Exact minimum: {gate['candidate_or_model_policy_exact_min']}/4",
            f"- Trusted candidate incorrect maximum: {gate['trusted_candidate_incorrect_max']}",
            f"- Hidden labels used by arbitration: `{gate['hidden_labels_used_by_arbitration']}`",
            f"- Raw predictions remain uncommitted: `{gate['raw_predictions_remain_uncommitted']}`",
            "",
            "Keep tool_query, DPO/RLVR, Hugging Face publication, release tagging,",
            "and broad retraining gated until this compact candidate-CE path meets",
            "or beats runtime evidence arbitration.",
            "",
            "Public-safety contract: raw saved predictions, candidate-score JSONL,",
            "scheduler logs, model state, and ignored run folders are private run",
            "outputs and are not public artifacts.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nonflag-result", default=DEFAULT_NONFLAG_RESULT)
    parser.add_argument(
        "--out-json",
        default="post_training/stage_a_saved_output_candidate_ce_checkpoint_spec_2026-07-10.json",
    )
    parser.add_argument(
        "--out-md",
        default="post_training/STAGE_A_SAVED_OUTPUT_CANDIDATE_CE_CHECKPOINT_SPEC_2026-07-10.md",
    )
    args = parser.parse_args()

    report = build_report(nonflag_result_path=args.nonflag_result)
    write_json(Path(args.out_json), report)
    Path(args.out_md).write_text(render_markdown(report))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

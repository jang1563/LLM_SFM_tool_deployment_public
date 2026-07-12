#!/usr/bin/env python3
"""Choose the next public-safe Stage A saved-output Cayuga checkpoint spec.

This checkpoint spec reads only compact public-safe reports. It does not inspect
raw saved predictions, candidate-score JSONL, scheduler logs, model state, or
ignored run folders. The output is a reproducible plan for the next model-heavy
Cayuga run plus the compact summaries required before any result can be
interpreted.
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


DATASET = "negbiodb_ct_stage_a_saved_output_next_checkpoint_spec_v1"
DEFAULT_CHECKPOINT_DIAGNOSIS = (
    "post_training/stage_a_saved_output_checkpoint_diagnosis_2026-07-10.json"
)
DEFAULT_CANDIDATE_FIELD = (
    "post_training/stage_a_saved_output_candidate_field_qwen05b_cayuga_summary_2026-07-10.json"
)
DEFAULT_TRAIN_CANDIDATE = (
    "post_training/stage_a_saved_output_train_candidate_qwen05b_cayuga_summary_2026-07-10.json"
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


def input_artifact(path: str | Path, *, role: str) -> dict[str, str]:
    return {
        "path": public_path(path),
        "role": role,
        "sha256": sha256_file(path),
    }


def mapping_at(payload: Mapping[str, Any], *keys: str) -> Mapping[str, Any]:
    value: Any = payload
    for key in keys:
        if not isinstance(value, Mapping):
            return {}
        value = value.get(key, {})
    return value if isinstance(value, Mapping) else {}


def compact_failure_evidence(
    *,
    checkpoint_diagnosis: Mapping[str, Any],
    candidate_field: Mapping[str, Any],
    train_candidate: Mapping[str, Any],
) -> dict[str, Any]:
    diagnosis = mapping_at(checkpoint_diagnosis, "diagnosis")
    rank = mapping_at(diagnosis, "finite_candidate_rank")
    arbitration = mapping_at(diagnosis, "runtime_arbitration")
    trained_field = mapping_at(candidate_field, "trained_field_diagnostic")
    heldout_candidate = mapping_at(train_candidate, "heldout_candidate_summary")
    train_candidate_summary = mapping_at(train_candidate, "train_candidate_summary")
    return {
        "bottleneck": "candidate_selection_bias_flag_invalid_value_overselection",
        "teacher_forced_margin_repaired": mapping_at(diagnosis, "teacher_forced_margin").get(
            "trained_full_margin_wins"
        )
        == mapping_at(diagnosis, "teacher_forced_margin").get("pairs"),
        "heldout_candidate_top1": {
            "exact": int(rank.get("trained_exact_top1", 0)),
            "rows": int(rank.get("pairs", 0)),
            "top_pair_counts": dict(rank.get("trained_top_pair_counts", {})),
        },
        "heldout_field_diagnostic": {
            "exact_top1": int(trained_field.get("exact_top1", 0)),
            "action_top1": int(trained_field.get("action_top1", 0)),
            "evidence_status_top1": int(trained_field.get("evidence_status_top1", 0)),
            "field_rank_patterns": dict(trained_field.get("field_rank_patterns", {})),
            "top_pair_counts": dict(trained_field.get("top_pair_counts", {})),
        },
        "train_candidate_bias": {
            "exact_top1": int(train_candidate_summary.get("exact_top1", 0)),
            "rows": int(train_candidate_summary.get("pairs", 0)),
            "top_pair_counts": dict(train_candidate_summary.get("top_pair_counts", {})),
        },
        "runtime_baseline": {
            "best_policy_names": list(arbitration.get("best_policy_names", ())),
            "hidden_labels_used_by_arbitration": bool(
                arbitration.get("hidden_labels_used_by_arbitration", True)
            ),
        },
        "heldout_candidate_summary": {
            "exact_top1": int(heldout_candidate.get("exact_top1", 0)),
            "rows": int(heldout_candidate.get("pairs", 0)),
            "top_pair_counts": dict(heldout_candidate.get("top_pair_counts", {})),
        },
    }


def next_checkpoint_env() -> dict[str, str]:
    return {
        "RUN_ID": "stage_a_saved_output_nonflag_candidate_rank_qwen05b_cayuga_${JOB_ID}",
        "MODEL_ID": "Qwen/Qwen2.5-0.5B-Instruct",
        "MAX_STEPS": "40",
        "BATCH_SIZE": "1",
        "LR": "1e-5",
        "TRAIN_LAST_LAYERS": "1",
        "TARGET_FORMAT": "full",
        "SCORE_TARGET_FORMATS": "full,action_status_only,action_only,status_only",
        "FOCUS_CHOSEN_PAIRS": "defer/insufficient,reject/contradicted,verify/insufficient",
        "FOCUS_REPEAT": "4",
        "FOCUS_ONLY": "0",
        "PAIRWISE_MARGIN_WEIGHT": "1",
        "PAIRWISE_MARGIN": "0.05",
        "SCORE_BASE_MARGINS": "1",
        "SCORE_TRAIN_MARGINS": "1",
        "SCORE_BASE_CANDIDATES": "1",
        "SCORE_TRAIN_CANDIDATES": "1",
        "SCORE_TRAINED_CANDIDATES": "1",
        "CANDIDATE_POLICY": "train_observed_plus_rejected",
        "CANDIDATE_TARGET_FORMAT": "full",
        "ALLOW_DOWNLOAD": "0",
    }


def compact_post_run_commands() -> list[str]:
    return [
        (
            "python post_training/analyze_stage_a_saved_output_candidate_calibration.py "
            "--train-candidates ${OUT_DIR}/train_candidates.jsonl "
            "--heldout-candidates ${OUT_DIR}/candidates.jsonl "
            "--out-json /tmp/stage_a_saved_output_nonflag_candidate_calibration.json "
            "--out-md /tmp/STAGE_A_SAVED_OUTPUT_NONFLAG_CANDIDATE_CALIBRATION.md"
        ),
        (
            "python post_training/analyze_stage_a_saved_output_candidate_fields.py "
            "--candidates ${OUT_DIR}/candidates.jsonl "
            "--out-json /tmp/stage_a_saved_output_nonflag_candidate_fields.json "
            "--out-md /tmp/STAGE_A_SAVED_OUTPUT_NONFLAG_CANDIDATE_FIELDS.md"
        ),
        (
            "python post_training/evaluate_stage_a_saved_output_candidate_arbitration.py "
            "--calibration-report /tmp/stage_a_saved_output_nonflag_candidate_calibration.json "
            "--out-json /tmp/stage_a_saved_output_nonflag_candidate_arbitration.json "
            "--out-md /tmp/STAGE_A_SAVED_OUTPUT_NONFLAG_CANDIDATE_ARBITRATION.md"
        ),
    ]


def curation_commands_after_manifest_registration() -> list[str]:
    return [
        (
            "python post_training/build_stage_a_saved_output_policy_summary.py "
            "--source post_training/<curated_nonflag_candidate_arbitration>.json "
            "--source-kind candidate-arbitration-policy "
            "--policy calibrated_candidate_top1 "
            "--out /tmp/stage_a_saved_output_nonflag_policy_summary.json"
        ),
        (
            "python post_training/evaluate_stage_a_saved_output_meet_or_beat_gate.py "
            "--model-policy-summary /tmp/stage_a_saved_output_nonflag_policy_summary.json "
            "--out-json /tmp/stage_a_saved_output_nonflag_meet_or_beat_gate.json "
            "--out-md /tmp/STAGE_A_SAVED_OUTPUT_NONFLAG_MEET_OR_BEAT_GATE.md"
        ),
    ]


def build_report(
    *,
    checkpoint_diagnosis_path: str | Path = DEFAULT_CHECKPOINT_DIAGNOSIS,
    candidate_field_path: str | Path = DEFAULT_CANDIDATE_FIELD,
    train_candidate_path: str | Path = DEFAULT_TRAIN_CANDIDATE,
) -> dict[str, Any]:
    checkpoint_diagnosis = load_json(checkpoint_diagnosis_path)
    candidate_field = load_json(candidate_field_path)
    train_candidate = load_json(train_candidate_path)
    failure = compact_failure_evidence(
        checkpoint_diagnosis=checkpoint_diagnosis,
        candidate_field=candidate_field,
        train_candidate=train_candidate,
    )

    report = {
        "dataset": DATASET,
        "current_thesis": checkpoint_diagnosis.get("current_thesis"),
        "input_artifacts": {
            "checkpoint_diagnosis": input_artifact(
                checkpoint_diagnosis_path,
                role="current saved-output compact diagnosis",
            ),
            "candidate_field": input_artifact(
                candidate_field_path,
                role="field-wise candidate-rank compact summary",
            ),
            "train_candidate": input_artifact(
                train_candidate_path,
                role="train and held-out candidate-bias compact summary",
            ),
        },
        "observed_failure": failure,
        "next_checkpoint": {
            "name": "balanced_nonflag_candidate_rank_readout",
            "runner": "post_training/run_stage_a_saved_output_calibration_margin_sft_cayuga.sbatch",
            "question": (
                "Does balancing non-flag target pairs reduce the learned global "
                "flag/invalid_value candidate prior while preserving the repaired "
                "full-target teacher-forced margins?"
            ),
            "env": next_checkpoint_env(),
            "submit_template": (
                "sbatch --account=<allocation> --partition=<gpu-partition> --gres=gpu:1 "
                "--export=ALL,WORK=$PWD,<env-overrides> "
                "post_training/run_stage_a_saved_output_calibration_margin_sft_cayuga.sbatch"
            ),
            "post_run_compact_commands": compact_post_run_commands(),
            "curation_commands_after_manifest_registration": (
                curation_commands_after_manifest_registration()
            ),
        },
        "acceptance_gate": {
            "candidate_or_model_policy_exact_min": 4,
            "trusted_candidate_incorrect_max": 0,
            "hidden_labels_used_by_arbitration": False,
            "raw_predictions_remain_uncommitted": True,
            "interpretation_rule": (
                "Only compact policy summaries that meet or beat runtime evidence "
                "arbitration reopen the downstream DPO/RLVR or release decision."
            ),
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
    return report


def render_markdown(report: Mapping[str, Any]) -> str:
    failure = report["observed_failure"]
    checkpoint = report["next_checkpoint"]
    env = checkpoint["env"]
    gate = report["acceptance_gate"]
    lines = [
        "# Stage A Saved-Output Next Checkpoint Spec",
        "",
        "Purpose: define the next Cayuga saved-output checkpoint from compact public-safe diagnosis artifacts.",
        "",
        "## Observed Failure",
        "",
        f"- Bottleneck: `{failure['bottleneck']}`",
        (
            f"- Held-out candidate top-1: "
            f"{failure['heldout_candidate_top1']['exact']}/{failure['heldout_candidate_top1']['rows']}, "
            f"top-pair counts `{json.dumps(failure['heldout_candidate_top1']['top_pair_counts'], sort_keys=True)}`"
        ),
        (
            f"- Field diagnostic: exact {failure['heldout_field_diagnostic']['exact_top1']}, "
            f"action {failure['heldout_field_diagnostic']['action_top1']}, "
            f"status {failure['heldout_field_diagnostic']['evidence_status_top1']}; "
            f"patterns `{json.dumps(failure['heldout_field_diagnostic']['field_rank_patterns'], sort_keys=True)}`"
        ),
        (
            f"- Train candidate bias: "
            f"{failure['train_candidate_bias']['exact_top1']}/{failure['train_candidate_bias']['rows']}, "
            f"top-pair counts `{json.dumps(failure['train_candidate_bias']['top_pair_counts'], sort_keys=True)}`"
        ),
        "",
        "## Next Checkpoint",
        "",
        f"- Name: `{checkpoint['name']}`",
        f"- Runner: `{checkpoint['runner']}`",
        f"- Question: {checkpoint['question']}",
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
    lines.extend(
        [
            "",
            "After the compact arbitration report is curated into the repo and listed",
            "in `release/public_release_manifest.json`, run:",
            "",
        ]
    )
    for command in checkpoint["curation_commands_after_manifest_registration"]:
        lines.append(f"- `{command}`")
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
            "and broad retraining gated until this compact policy path meets or beats",
            "runtime evidence arbitration.",
            "",
            "Public-safety contract: raw saved predictions, candidate-score JSONL,",
            "scheduler logs, model state, and ignored run folders are private run",
            "outputs and are not public artifacts.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-diagnosis", default=DEFAULT_CHECKPOINT_DIAGNOSIS)
    parser.add_argument("--candidate-field", default=DEFAULT_CANDIDATE_FIELD)
    parser.add_argument("--train-candidate", default=DEFAULT_TRAIN_CANDIDATE)
    parser.add_argument(
        "--out-json",
        default="post_training/stage_a_saved_output_next_checkpoint_spec_2026-07-10.json",
    )
    parser.add_argument(
        "--out-md",
        default="post_training/STAGE_A_SAVED_OUTPUT_NEXT_CHECKPOINT_SPEC_2026-07-10.md",
    )
    args = parser.parse_args()

    report = build_report(
        checkpoint_diagnosis_path=args.checkpoint_diagnosis,
        candidate_field_path=args.candidate_field,
        train_candidate_path=args.train_candidate,
    )
    write_json(Path(args.out_json), report)
    Path(args.out_md).write_text(render_markdown(report))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

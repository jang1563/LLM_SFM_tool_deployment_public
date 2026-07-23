#!/usr/bin/env python3
"""Freeze the Stage A candidate-routing policy for sealed evaluation."""

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

from post_training.export_stage_a_saved_output_evidence_candidate_routing_rows import (  # noqa: E402
    CANDIDATE_PAIRS,
    CANDIDATE_POLICY,
    PROMPT_CONTRACT,
)
from post_training.run_stage_a_saved_output_evidence_candidate_routing_smoke import (  # noqa: E402
    SYSTEM_PROMPT,
)


DATASET = "negbiodb_ct_stage_a_candidate_routing_policy_freeze_v1"
DEFAULT_TRAIN_ROWS = (
    "post_training/stage_a_saved_output_evidence_candidate_routing_train_v1.jsonl"
)
DEFAULT_ROWS_MANIFEST = (
    "post_training/stage_a_saved_output_evidence_candidate_routing_manifest.json"
)
DEFAULT_RUNNER = (
    "post_training/run_stage_a_saved_output_evidence_candidate_routing_smoke.py"
)
DEFAULT_EXPORTER = (
    "post_training/export_stage_a_saved_output_evidence_candidate_routing_rows.py"
)
DEFAULT_SEALED_EVALUATOR = (
    "post_training/run_stage_a_sealed_candidate_routing_eval.py"
)
DEFAULT_TOOL_QUERY_RESULT = (
    "post_training/stage_a_tool_query_sft_smoke_result_qwen05b_cayuga_2026-07-23.json"
)
DEFAULT_CANDIDATE_RESULT = (
    "post_training/"
    "stage_a_saved_output_evidence_candidate_routing_smoke_result_qwen05b_cayuga_2026-07-10.json"
)
DEFAULT_SEALED_COMMITMENT = (
    "post_training/stage_a_sealed_extension_commitment_2026-07-10.json"
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


def tracked_artifact(path: str | Path, *, role: str) -> dict[str, str]:
    resolved = Path(path).resolve()
    try:
        display = resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"tracked freeze artifact must be inside repository: {path}") from exc
    return {"path": display, "role": role, "sha256": sha256_file(resolved)}


def private_artifact(path: str | Path, *, role: str) -> dict[str, str]:
    resolved = Path(path).resolve()
    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError:
        pass
    else:
        raise ValueError(f"private freeze artifact must stay outside repository: {path}")
    return {
        "path": f"external_private_input::{resolved.name}",
        "role": role,
        "sha256": sha256_file(resolved),
    }


def canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def build_freeze(
    *,
    training_report_path: str | Path,
    trainable_state_path: str | Path,
    model_revision: str,
    train_rows_path: str | Path = DEFAULT_TRAIN_ROWS,
    rows_manifest_path: str | Path = DEFAULT_ROWS_MANIFEST,
    runner_path: str | Path = DEFAULT_RUNNER,
    exporter_path: str | Path = DEFAULT_EXPORTER,
    sealed_evaluator_path: str | Path = DEFAULT_SEALED_EVALUATOR,
    tool_query_result_path: str | Path = DEFAULT_TOOL_QUERY_RESULT,
    candidate_result_path: str | Path = DEFAULT_CANDIDATE_RESULT,
    sealed_commitment_path: str | Path = DEFAULT_SEALED_COMMITMENT,
) -> dict[str, Any]:
    training = load_json(training_report_path)
    tool_query = load_json(tool_query_result_path)
    candidate_result = load_json(candidate_result_path)
    commitment = load_json(sealed_commitment_path)
    if training.get("dry_run") is not False:
        raise ValueError("training report is not a completed full run")
    if training.get("model") != "Qwen/Qwen2.5-0.5B-Instruct":
        raise ValueError("unexpected frozen model")
    if tool_query.get("decision", {}).get("diagnostic_complete") is not True:
        raise ValueError("tool-query diagnostic is not complete")
    if commitment.get("decision", {}).get("ready_for_one_time_sealed_evaluation") is not True:
        raise ValueError("sealed commitment is not ready")

    artifacts = {
        "training_report": private_artifact(
            training_report_path, role="private candidate-routing training report"
        ),
        "trainable_state": private_artifact(
            trainable_state_path, role="private frozen trainable parameters"
        ),
        "train_rows": tracked_artifact(
            train_rows_path, role="tracked exposed development training rows"
        ),
        "rows_manifest": tracked_artifact(
            rows_manifest_path, role="tracked candidate-routing manifest"
        ),
        "training_runner": tracked_artifact(
            runner_path, role="tracked candidate-routing training runner"
        ),
        "row_exporter": tracked_artifact(
            exporter_path, role="tracked evidence candidate-row exporter"
        ),
        "sealed_evaluator": tracked_artifact(
            sealed_evaluator_path, role="tracked one-time sealed evaluator"
        ),
        "tool_query_result": tracked_artifact(
            tool_query_result_path, role="tracked tool-query diagnostic result"
        ),
        "candidate_routing_result": tracked_artifact(
            candidate_result_path, role="tracked exposed-development routing result"
        ),
        "sealed_commitment": tracked_artifact(
            sealed_commitment_path, role="tracked sealed-manifest commitment"
        ),
    }
    policy = {
        "model_id": training.get("model"),
        "model_revision": model_revision,
        "prompt_contract": PROMPT_CONTRACT,
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "candidate_policy": CANDIDATE_POLICY,
        "candidate_pairs": list(CANDIDATE_PAIRS),
        "max_steps": training.get("max_steps"),
        "batch_size": training.get("batch_size"),
        "max_length": training.get("max_length"),
        "learning_rate": 1e-5,
        "train_last_layers": training.get("train_last_layers"),
        "trainable_params": training.get("trainable_params"),
        "saved_state_load_only": True,
    }
    freeze_payload = {
        "policy": policy,
        "artifact_hashes": {
            name: value["sha256"] for name, value in sorted(artifacts.items())
        },
    }
    freeze_id = f"stage_a_candidate_routing_freeze::{canonical_sha256(freeze_payload)}"
    return {
        "dataset": DATASET,
        "freeze_id": freeze_id,
        "policy": policy,
        "frozen_artifacts": artifacts,
        "pre_freeze_results": {
            "tool_query_schema_gate_passed": tool_query.get("decision", {}).get(
                "tool_query_schema_gate_passed"
            ),
            "candidate_routing_heldout_exact": candidate_result.get(
                "heldout_summary", {}
            ).get("exact"),
            "candidate_routing_heldout_rows": candidate_result.get(
                "heldout_summary", {}
            ).get("rows"),
            "candidate_routing_passes_gate": candidate_result.get("passes_gate"),
        },
        "authorization": {
            "tool_query_diagnostic_complete": True,
            "sealed_commitment_ready": True,
            "ready_for_one_time_sealed_evaluation": True,
            "training_on_sealed_rows_allowed": False,
            "repeated_sealed_evaluation_allowed": False,
        },
        "reproducibility_boundary": {
            "saved_trainable_state_is_authoritative": True,
            "retraining_before_sealed_evaluation": False,
            "original_training_seed_explicitly_recorded": False,
            "exact_retraining_claimed": False,
            "reason": (
                "The original smoke did not record an explicit seed. The sealed "
                "evaluation loads the hashed saved trainable state directly and "
                "does not retrain."
            ),
        },
        "scientific_boundary": {
            "evaluation_scope": "routing_after_synthetic_oracle_tool_loop",
            "actual_identifier_resolution_evaluated": False,
            "live_tool_execution_evaluated": False,
            "hidden_labels_visible_to_model": False,
            "dpo_rlvr_opened": False,
        },
        "public_safety_contract": {
            "private_paths_redacted": True,
            "trainable_state_committed": False,
            "raw_training_report_committed": False,
            "sealed_rows_committed": False,
            "sealed_row_labels_emitted": False,
        },
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    policy = report["policy"]
    pre = report["pre_freeze_results"]
    boundary = report["reproducibility_boundary"]
    lines = [
        "# Stage A Candidate-Routing Policy Freeze",
        "",
        "Purpose: freeze the exact saved candidate-routing policy before the",
        "one-time source-separated sealed evaluation.",
        "",
        "## Frozen Policy",
        "",
        f"- Freeze ID: `{report['freeze_id']}`",
        f"- Model: `{policy['model_id']}`",
        f"- Model revision: `{policy['model_revision']}`",
        f"- Candidate pairs: `{json.dumps(policy['candidate_pairs'])}`",
        f"- Training steps: {policy['max_steps']}",
        f"- Max length: {policy['max_length']}",
        f"- Saved-state load only: `{policy['saved_state_load_only']}`",
        "",
        "## Pre-Freeze Results",
        "",
        (
            f"- Exposed-development candidate routing: "
            f"{pre['candidate_routing_heldout_exact']}/"
            f"{pre['candidate_routing_heldout_rows']}"
        ),
        f"- Candidate-routing gate passed: `{pre['candidate_routing_passes_gate']}`",
        f"- Tool-query schema gate passed: `{pre['tool_query_schema_gate_passed']}`",
        "",
        "## Boundary",
        "",
        f"- Explicit original training seed recorded: `{boundary['original_training_seed_explicitly_recorded']}`",
        "- The sealed evaluation loads the hashed saved trainable state and does not retrain.",
        "- The sealed run evaluates routing after a synthetic oracle tool loop;",
        "  it does not evaluate identifier resolution or live tool execution.",
        "- Repeated sealed evaluation and training on sealed rows are prohibited.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-report", required=True)
    parser.add_argument("--trainable-state", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    args = parser.parse_args()

    report = build_freeze(
        training_report_path=args.training_report,
        trainable_state_path=args.trainable_state,
        model_revision=args.model_revision,
    )
    Path(args.out_json).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    Path(args.out_md).write_text(render_markdown(report))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

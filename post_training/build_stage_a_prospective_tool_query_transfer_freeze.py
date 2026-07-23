#!/usr/bin/env python3
"""Freeze the pre-prospective tool-query state and transfer protocol."""

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

from post_training.build_stage_a_prospective_real_query_slice import (  # noqa: E402
    PROMPT_CONTRACT,
    TOOL_QUERY_DATASET,
)
from post_training.run_stage_a_prospective_tool_query_transfer import (  # noqa: E402
    FREEZE_DATASET,
    SYSTEM_PROMPT,
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def load_jsonl_count(path: str | Path) -> int:
    return sum(1 for line in Path(path).read_text().splitlines() if line.strip())


def canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def repo_artifact(path: str | Path, *, role: str) -> dict[str, Any]:
    resolved = Path(path).resolve()
    try:
        display = resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"freeze artifact must be tracked in repository: {path}") from exc
    value: dict[str, Any] = {
        "path": display,
        "role": role,
        "sha256": sha256_file(resolved),
    }
    if resolved.suffix == ".jsonl":
        value["records"] = load_jsonl_count(resolved)
    return value


def validate_sha256(value: str) -> str:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError("state SHA256 must be 64 lowercase hexadecimal characters")
    return value


def build_freeze(
    *,
    state_sha256: str,
    model_revision: str,
    source_result_path: str | Path,
    query_rows_path: str | Path,
    experiment_manifest_path: str | Path,
) -> dict[str, Any]:
    source = load_json(source_result_path)
    training = source.get("training")
    if source.get("component") != "tool_query" or not isinstance(training, Mapping):
        raise ValueError("source result is not a tool-query training result")
    if source.get("decision", {}).get("diagnostic_complete") is not True:
        raise ValueError("source tool-query diagnostic is incomplete")
    artifacts = {
        "source_tool_query_result": repo_artifact(
            source_result_path,
            role="public compact placeholder-schema training result",
        ),
        "tool_query_rows": repo_artifact(
            query_rows_path,
            role="public prospective case-specific query rows",
        ),
        "experiment_manifest": repo_artifact(
            experiment_manifest_path,
            role="public prospective experiment manifest",
        ),
        "trainable_state": {
            "path": "external_private_input::trainable_state.pt",
            "role": "private pre-prospective tool-query trainable parameters",
            "sha256": validate_sha256(state_sha256),
        },
    }
    if artifacts["tool_query_rows"]["records"] != 25:
        raise ValueError("prospective transfer requires exactly 25 query rows")
    policy = {
        "model_id": training.get("model"),
        "model_revision": model_revision,
        "prompt_contract": PROMPT_CONTRACT,
        "source_prompt_contract": "stage_a_v2_strict",
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "decode_mode": "freeform",
        "max_new_tokens": training.get("max_new_tokens"),
        "max_length": training.get("max_length"),
        "train_last_layers": training.get("train_last_layers"),
        "trainable_params": training.get("trainable_params"),
        "retraining_allowed": False,
        "comparison_policies": ["base", "frozen_tool_query_sft"],
    }
    identity = {
        "policy": policy,
        "artifact_hashes": {
            key: value["sha256"] for key, value in sorted(artifacts.items())
        },
    }
    return {
        "dataset": FREEZE_DATASET,
        "freeze_id": f"stage_a_tool_query_transfer_freeze::{canonical_sha256(identity)}",
        "policy": policy,
        "frozen_artifacts": artifacts,
        "source_result": {
            "run_id": source.get("run_id"),
            "heldout_passed": source.get("heldout_result", {}).get("passed"),
            "heldout_cases": source.get("heldout_result", {}).get("cases"),
            "source_targets_used_literal_placeholders": source.get(
                "experiment_scope", {}
            ).get("all_targets_use_expected_placeholder_sequence"),
        },
        "authorization": {
            "ready_for_base_vs_frozen_transfer": True,
            "training_on_prospective_rows_allowed": False,
            "retraining_before_transfer_allowed": False,
            "dpo_rlvr_allowed": False,
        },
        "scientific_boundary": {
            "prospective_row_dataset": TOOL_QUERY_DATASET,
            "development_only": True,
            "independent_test_claimed": False,
            "actual_identifier_values_visible": True,
            "live_tool_execution_evaluated": False,
            "state_trained_before_prospective_rows": True,
        },
        "public_safety_contract": {
            "trainable_state_committed": False,
            "raw_source_predictions_committed": False,
            "private_paths_redacted": True,
        },
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-sha256", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument(
        "--source-result",
        default=(
            "post_training/"
            "stage_a_tool_query_sft_smoke_result_qwen05b_cayuga_2026-07-23.json"
        ),
    )
    parser.add_argument(
        "--query-rows",
        default="post_training/stage_a_prospective_real_query_tool_query_v1.jsonl",
    )
    parser.add_argument(
        "--experiment-manifest",
        default=(
            "post_training/stage_a_prospective_real_query_experiment_manifest.json"
        ),
    )
    parser.add_argument(
        "--out",
        default=(
            "post_training/"
            "stage_a_prospective_tool_query_transfer_freeze_2026-07-23.json"
        ),
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    report = build_freeze(
        state_sha256=args.state_sha256,
        model_revision=args.model_revision,
        source_result_path=args.source_result,
        query_rows_path=args.query_rows,
        experiment_manifest_path=args.experiment_manifest,
    )
    Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

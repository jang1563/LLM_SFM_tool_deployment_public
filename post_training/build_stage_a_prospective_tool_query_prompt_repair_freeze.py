#!/usr/bin/env python3
"""Freeze the adaptive explicit-prompt tool-query diagnostic."""

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

from post_training.build_stage_a_prospective_tool_query_transfer_freeze import (  # noqa: E402
    canonical_sha256,
    repo_artifact,
)
from post_training.run_stage_a_prospective_tool_query_prompt_repair import (  # noqa: E402
    EXPLICIT_SYSTEM_PROMPT,
    FREEZE_DATASET,
)


def build_freeze(
    *,
    model_id: str,
    model_revision: str,
    max_new_tokens: int,
    query_rows_path: str | Path,
    experiment_manifest_path: str | Path,
    observed_transfer_result_path: str | Path,
) -> dict[str, Any]:
    artifacts = {
        "tool_query_rows": repo_artifact(
            query_rows_path,
            role="public prospective case-specific query rows",
        ),
        "experiment_manifest": repo_artifact(
            experiment_manifest_path,
            role="public prospective experiment manifest",
        ),
        "observed_transfer_result": repo_artifact(
            observed_transfer_result_path,
            role="public aggregate result motivating adaptive prompt repair",
        ),
    }
    if artifacts["tool_query_rows"].get("records") != 25:
        raise ValueError("prompt repair requires exactly 25 query rows")
    policy: dict[str, Any] = {
        "model_id": model_id,
        "model_revision": model_revision,
        "decode_mode": "freeform",
        "max_new_tokens": max_new_tokens,
        "system_prompt": EXPLICIT_SYSTEM_PROMPT,
        "system_prompt_sha256": hashlib.sha256(
            EXPLICIT_SYSTEM_PROMPT.encode()
        ).hexdigest(),
        "training_allowed": False,
    }
    identity: Mapping[str, Any] = {
        "policy": policy,
        "artifact_hashes": {
            key: value["sha256"] for key, value in sorted(artifacts.items())
        },
    }
    return {
        "dataset": FREEZE_DATASET,
        "freeze_id": (
            "stage_a_tool_query_prompt_repair_freeze::"
            f"{canonical_sha256(identity)}"
        ),
        "policy": policy,
        "frozen_artifacts": artifacts,
        "authorization": {
            "ready_for_adaptive_prompt_diagnostic": True,
            "training_allowed": False,
            "dpo_rlvr_allowed": False,
        },
        "scientific_boundary": {
            "adaptive_after_observing_input_echo_failure": True,
            "development_only": True,
            "independent_test_claimed": False,
            "live_tool_execution_evaluated": False,
        },
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--query-rows",
        default="post_training/stage_a_prospective_real_query_tool_query_v1.jsonl",
    )
    parser.add_argument(
        "--experiment-manifest",
        default="post_training/stage_a_prospective_real_query_experiment_manifest.json",
    )
    parser.add_argument(
        "--observed-transfer-result",
        default=(
            "post_training/"
            "stage_a_prospective_tool_query_transfer_result_qwen05b_cayuga_2026-07-23.json"
        ),
    )
    parser.add_argument(
        "--out",
        default=(
            "post_training/"
            "stage_a_prospective_tool_query_prompt_repair_freeze_2026-07-23.json"
        ),
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    report = build_freeze(
        model_id=args.model_id,
        model_revision=args.model_revision,
        max_new_tokens=args.max_new_tokens,
        query_rows_path=args.query_rows,
        experiment_manifest_path=args.experiment_manifest,
        observed_transfer_result_path=args.observed_transfer_result,
    )
    Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Test an explicit tool-query output contract after input-echo failure.

This is an adaptive public-development diagnostic, not a sealed evaluation.
It performs no training and keeps raw generations outside the public repo.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from post_training.build_stage_a_prospective_real_query_slice import (  # noqa: E402
    MANIFEST_DATASET,
    TOOL_QUERY_DATASET,
)
from post_training.evaluate_stage_a_prospective_runtime_hybrid import (  # noqa: E402
    load_json,
    sha256_file,
)
from post_training.run_stage_a_prospective_frozen_policy import (  # noqa: E402
    private_output_path,
)
from post_training.run_stage_a_prospective_tool_query_transfer import (  # noqa: E402
    expected_arguments,
    generate_policy_predictions,
    load_base_model,
    summarize_policy,
    target_tool_calls,
    write_private_jsonl,
)
from post_training.run_stage_a_strict_contract_sft_smoke import (  # noqa: E402
    load_jsonl,
)


DATASET = "negbiodb_ct_stage_a_prospective_tool_query_prompt_repair_run_v1"
FREEZE_DATASET = (
    "negbiodb_ct_stage_a_prospective_tool_query_prompt_repair_freeze_v1"
)
POLICY = "base_explicit_contract"
EXPLICIT_SYSTEM_PROMPT = (
    'You are a Stage A tool-query compiler. Return exactly one JSON object '
    'with exactly one top-level key "tool_calls". Use every name in '
    "allowed_tools exactly once, in listed order. For every call, arguments "
    'must contain exactly "drug_id" and "condition_id". Copy their raw values '
    "from query.drug_id.value and query.condition_id.value. Do not return or "
    "repeat any input fields. Do not include prose or markdown."
)


def validate_inputs(
    *,
    rows: Sequence[Mapping[str, Any]],
    rows_path: str | Path,
    manifest: Mapping[str, Any],
    freeze: Mapping[str, Any],
) -> list[str]:
    issues: list[str] = []
    if manifest.get("dataset") != MANIFEST_DATASET:
        issues.append("unexpected_experiment_manifest_dataset")
    expected_rows = manifest.get("artifacts", {}).get("tool_query_rows", {})
    if expected_rows.get("sha256") != sha256_file(rows_path):
        issues.append("tool_query_rows_sha256_mismatch")
    if expected_rows.get("records") != len(rows):
        issues.append("tool_query_rows_record_count_mismatch")
    if freeze.get("dataset") != FREEZE_DATASET:
        issues.append("unexpected_prompt_repair_freeze_dataset")
    frozen_rows = freeze.get("frozen_artifacts", {}).get("tool_query_rows", {})
    if frozen_rows.get("sha256") != sha256_file(rows_path):
        issues.append("prompt_repair_freeze_rows_sha256_mismatch")
    policy = freeze.get("policy", {})
    if policy.get("training_allowed") is not False:
        issues.append("prompt_repair_freeze_does_not_prohibit_training")
    if policy.get("system_prompt") != EXPLICIT_SYSTEM_PROMPT:
        issues.append("prompt_repair_system_prompt_mismatch")

    target_hashes: set[str] = set()
    seen_ids: set[str] = set()
    for row in rows:
        row_id = str(row.get("id"))
        if row_id in seen_ids:
            issues.append(f"{row_id}:duplicate_row_id")
        seen_ids.add(row_id)
        if row.get("dataset") != TOOL_QUERY_DATASET:
            issues.append(f"{row_id}:unexpected_dataset")
        visible = json.dumps(row.get("model_visible_task"), sort_keys=True)
        for forbidden in (
            "hidden_eval_metadata",
            "source_task_id",
            "split_group",
            '"target_output"',
        ):
            if forbidden in visible:
                issues.append(f"{row_id}:prompt_leak:{forbidden}")
        arguments = expected_arguments(row)
        calls = target_tool_calls(row)
        if len(calls) != 4:
            issues.append(f"{row_id}:target_tool_call_count")
        if any(call.get("arguments") != arguments for call in calls):
            issues.append(f"{row_id}:target_query_value_mismatch")
        target_hashes.add(
            json.dumps(row.get("target_output"), separators=(",", ":"), sort_keys=True)
        )
    if len(target_hashes) != len(rows):
        issues.append("tool_query_targets_not_case_specific")
    return sorted(set(issues))


def dry_run_report(
    *,
    rows: Sequence[Mapping[str, Any]],
    rows_path: str | Path,
    manifest_path: str | Path,
    freeze: Mapping[str, Any],
    issues: Sequence[str],
) -> dict[str, Any]:
    return {
        "dataset": DATASET,
        "dry_run": True,
        "rows": len(rows),
        "unique_targets": len(
            {json.dumps(row.get("target_output"), sort_keys=True) for row in rows}
        ),
        "tool_query_rows_sha256": sha256_file(rows_path),
        "experiment_manifest_sha256": sha256_file(manifest_path),
        "freeze_id": freeze.get("freeze_id"),
        "issues": list(issues),
        "ready_for_full_mode": not issues,
        "training_performed": False,
        "completed_sealed_rows_used": False,
    }


def run_full(
    *,
    rows: Sequence[Mapping[str, Any]],
    freeze: Mapping[str, Any],
    predictions_out: str | Path,
    device: str,
    allow_download: bool,
) -> dict[str, Any]:
    policy = freeze["policy"]
    model, tokenizer, selected_device = load_base_model(
        model_id=str(policy["model_id"]),
        model_revision=str(policy["model_revision"]),
        device=device,
        allow_download=allow_download,
    )
    predictions = generate_policy_predictions(
        model,
        tokenizer,
        rows,
        policy=POLICY,
        device=selected_device,
        max_new_tokens=int(policy["max_new_tokens"]),
        system_prompt=EXPLICIT_SYSTEM_PROMPT,
    )
    write_private_jsonl(predictions_out, predictions)
    summary = summarize_policy(rows, predictions, policy=POLICY)
    return {
        "dataset": DATASET,
        "dry_run": False,
        "evaluation_scope": (
            "post_failure_adaptive_public_development_prompt_contract_diagnostic"
        ),
        "rows": len(rows),
        "freeze_id": freeze.get("freeze_id"),
        "model_id": policy["model_id"],
        "model_revision": policy["model_revision"],
        "device_class": (
            "cuda" if selected_device.startswith("cuda") else selected_device
        ),
        "training_performed": False,
        "completed_sealed_rows_used": False,
        "private_predictions": {
            "path": f"private_output::{Path(predictions_out).name}",
            "sha256": sha256_file(predictions_out),
            "committed": False,
        },
        "policy": summary,
        "decision": {
            "exact": summary["exact"],
            "query_values_exact": summary["query_values"],
            "prompt_contract_resolves_transfer": summary["exact"] == len(rows),
            "ready_for_corrective_sft": summary["exact"] < len(rows),
            "ready_for_dpo_rlvr": False,
            "ready_for_hugging_face_publication": False,
        },
        "scientific_boundary": {
            "adaptive_after_observing_input_echo_failure": True,
            "development_only": True,
            "independent_test_claimed": False,
            "actual_query_identifier_values_visible": True,
            "live_tool_execution_evaluated": False,
            "retraining_performed": False,
        },
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    policy = report["policy"]
    rows = policy["rows"]
    lines = [
        "# Stage A Prospective Tool-Query Prompt Repair",
        "",
        "Scope: adaptive public-development diagnostic after observing input-copy",
        "behavior. The base model received an explicit output contract; no training",
        "or live-tool execution was performed.",
        "",
        "| Metric | Passed |",
        "| --- | ---: |",
        f"| Parseable JSON | {policy['parseable_json']}/{rows} |",
        f"| Tool sequence | {policy['tool_sequence']}/{rows} |",
        f"| Query fields | {policy['query_fields']}/{rows} |",
        f"| Query values | {policy['query_values']}/{rows} |",
        f"| Exact | {policy['exact']}/{rows} |",
        "",
        "## Decision",
        "",
        (
            "- Explicit prompt contract fully resolves transfer: "
            f"`{str(report['decision']['prompt_contract_resolves_transfer']).lower()}`."
        ),
        "- DPO/RLVR and Hugging Face publication remain closed.",
        "- Raw generations remain private and uncommitted.",
        "- This adaptive diagnostic is not a sealed-test estimate.",
    ]
    return "\n".join(lines) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rows",
        default="post_training/stage_a_prospective_real_query_tool_query_v1.jsonl",
    )
    parser.add_argument(
        "--manifest",
        default="post_training/stage_a_prospective_real_query_experiment_manifest.json",
    )
    parser.add_argument(
        "--freeze",
        default=(
            "post_training/"
            "stage_a_prospective_tool_query_prompt_repair_freeze_2026-07-23.json"
        ),
    )
    parser.add_argument("--predictions-out", required=True)
    parser.add_argument("--report-out", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--allow-model-load", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    rows = load_jsonl(args.rows)
    manifest = load_json(args.manifest)
    freeze = load_json(args.freeze)
    issues = validate_inputs(
        rows=rows,
        rows_path=args.rows,
        manifest=manifest,
        freeze=freeze,
    )
    if issues:
        raise SystemExit("Prompt-repair validation failed:\n- " + "\n- ".join(issues))
    if args.dry_run:
        report = dry_run_report(
            rows=rows,
            rows_path=args.rows,
            manifest_path=args.manifest,
            freeze=freeze,
            issues=issues,
        )
    else:
        if not args.allow_model_load:
            raise SystemExit(
                "Full mode requires --allow-model-load; run --dry-run first."
            )
        report = run_full(
            rows=rows,
            freeze=freeze,
            predictions_out=args.predictions_out,
            device=args.device,
            allow_download=args.allow_download,
        )
    if args.report_out:
        report_path = private_output_path(args.report_out)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        report_path.chmod(0o600)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

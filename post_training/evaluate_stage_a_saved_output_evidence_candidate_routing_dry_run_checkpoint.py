#!/usr/bin/env python3
"""Create a public-safe checkpoint from a candidate-routing smoke dry-run.

This adapter records whether a local or Cayuga mirror dry-run validates the
evidence-conditioned candidate-routing smoke substrate. It reads only the dry
run's compact report JSON; it does not inspect raw candidate scores, raw model
text, scheduler logs, model state, or ignored run folders.
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
from post_training.run_stage_a_saved_output_evidence_candidate_routing_smoke import (  # noqa: E402
    DATASET as SMOKE_DATASET,
)


DATASET = "negbiodb_ct_stage_a_saved_output_evidence_candidate_routing_dry_run_checkpoint_v1"
RAW_FIELD_NAMES = (
    "candidate_scores",
    "raw_output",
    "raw_model_text",
    "prompt",
    "completion",
    "scheduler_log",
    "trainable_state",
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
        if path.is_absolute():
            return f"external_compact_input::{path.name}"
        return path.as_posix()


def raw_field_paths(value: Any, *, prefix: str = "$") -> list[str]:
    paths: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            child = f"{prefix}.{key}"
            if str(key) in RAW_FIELD_NAMES:
                paths.append(child)
            paths.extend(raw_field_paths(item, prefix=child))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            paths.extend(raw_field_paths(item, prefix=f"{prefix}[{index}]"))
    return paths


def contract_ok(report: Mapping[str, Any]) -> bool:
    contract = report.get("public_safety_contract")
    return isinstance(contract, Mapping) and all(value is False for value in contract.values())


def checkpoint_violations(report: Mapping[str, Any]) -> list[str]:
    violations: list[str] = []
    if report.get("dataset") != SMOKE_DATASET:
        violations.append("dry_run_dataset_mismatch")
    if report.get("dry_run") is not True:
        violations.append("not_a_dry_run_report")
    expected_counts = {
        "train_examples": 20,
        "heldout_examples": 5,
        "bridge_focus_heldout_examples": 4,
        "candidate_space_size": 5,
    }
    for key, expected in expected_counts.items():
        if report.get(key) != expected:
            violations.append(f"{key}_mismatch")
    if report.get("ready_for_full_mode") is not True:
        violations.append("not_ready_for_full_mode")
    if report.get("issues") != []:
        violations.append("dry_run_issues_present")
    if not contract_ok(report):
        violations.append("public_safety_contract_not_false")
    if raw_field_paths(report):
        violations.append("raw_fields_present")
    return violations


def build_report(
    *,
    dry_run_report_path: str | Path,
    execution_surface: str,
    mirror_commit: str,
    public_release_check_passed: bool,
) -> dict[str, Any]:
    dry_run = load_json(dry_run_report_path)
    violations = checkpoint_violations(dry_run)
    if not public_release_check_passed:
        violations.append("public_release_check_not_passed")
    passes = not violations
    return {
        "dataset": DATASET,
        "execution_surface": execution_surface,
        "mirror_commit": mirror_commit,
        "input_artifact": {
            "path": public_path(dry_run_report_path),
            "role": "compact smoke dry-run report",
            "sha256": sha256_file(dry_run_report_path),
            "tracked_public_artifact": False,
        },
        "dry_run_summary": {
            "dataset": dry_run.get("dataset"),
            "dry_run": dry_run.get("dry_run"),
            "train_examples": dry_run.get("train_examples"),
            "heldout_examples": dry_run.get("heldout_examples"),
            "bridge_focus_heldout_examples": dry_run.get("bridge_focus_heldout_examples"),
            "candidate_space_size": dry_run.get("candidate_space_size"),
            "ready_for_full_mode": dry_run.get("ready_for_full_mode"),
            "issues": list(dry_run.get("issues", ())),
            "rows": dry_run.get("rows"),
            "train_rows": dry_run.get("train_rows"),
            "heldout_rows": dry_run.get("heldout_rows"),
            "manifest": dry_run.get("manifest"),
        },
        "public_release_check_passed": public_release_check_passed,
        "passes_checkpoint": passes,
        "checkpoint_violations": violations,
        "next_decision": (
            "explicitly_approve_full_cayuga_smoke_or_keep_no_submit_boundary"
        ),
        "public_safety_contract": {
            "raw_prediction_jsonl_read": False,
            "raw_candidate_score_jsonl_read": False,
            "scheduler_logs_read": False,
            "model_state_read": False,
            "ignored_run_folder_read": False,
            "raw_fields_in_dry_run_report": bool(raw_field_paths(dry_run)),
        },
        "boundary": (
            "This checkpoint proves dry-run readiness only. It is not a full "
            "model result and does not open DPO/RLVR, tool_query, HF publication, "
            "or release tagging."
        ),
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    summary = report["dry_run_summary"]
    lines = [
        "# Stage A Evidence Candidate-Routing Dry-Run Checkpoint",
        "",
        "Purpose: record a public-safe mirror dry-run before any full model submission.",
        "",
        "## Summary",
        "",
        f"- Execution surface: `{report['execution_surface']}`",
        f"- Mirror commit: `{report['mirror_commit']}`",
        f"- Train rows: {summary['train_examples']}",
        f"- Held-out rows: {summary['heldout_examples']}",
        f"- Bridge-focus held-out rows: {summary['bridge_focus_heldout_examples']}",
        f"- Candidate space size: {summary['candidate_space_size']}",
        f"- Ready for full mode: `{summary['ready_for_full_mode']}`",
        f"- Issues: `{json.dumps(summary['issues'], sort_keys=True)}`",
        f"- Public release check passed: `{report['public_release_check_passed']}`",
        f"- Passes checkpoint: `{report['passes_checkpoint']}`",
        "",
        "## Next Decision",
        "",
        f"`{report['next_decision']}`",
        "",
        report["boundary"],
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run-report", required=True)
    parser.add_argument("--execution-surface", default="local")
    parser.add_argument("--mirror-commit", default="unknown")
    parser.add_argument("--public-release-check-passed", action="store_true")
    parser.add_argument(
        "--out-json",
        default="post_training/stage_a_saved_output_evidence_candidate_routing_cayuga_dry_run_2026-07-10.json",
    )
    parser.add_argument(
        "--out-md",
        default="post_training/STAGE_A_SAVED_OUTPUT_EVIDENCE_CANDIDATE_ROUTING_CAYUGA_DRY_RUN_2026-07-10.md",
    )
    args = parser.parse_args()

    report = build_report(
        dry_run_report_path=args.dry_run_report,
        execution_surface=args.execution_surface,
        mirror_commit=args.mirror_commit,
        public_release_check_passed=args.public_release_check_passed,
    )
    write_json(Path(args.out_json), report)
    Path(args.out_md).write_text(render_markdown(report))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

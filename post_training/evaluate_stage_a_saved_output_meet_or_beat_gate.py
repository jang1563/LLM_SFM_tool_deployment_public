#!/usr/bin/env python3
"""Evaluate the Stage A saved-output meet-or-beat arbitration gate.

This is a public-safe acceptance gate for future model-heavy Cayuga outputs. It
reads only compact checkpoint JSON files: the saved-output next-decision report
and the saved-output candidate arbitration report. It does not read raw saved
predictions, raw candidate-score JSONL, scheduler logs, model state, or ignored
run folders.
"""

import argparse
import hashlib
import json
from collections.abc import Mapping as MappingABC
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union


DATASET = "negbiodb_ct_stage_a_saved_output_meet_or_beat_gate_v1"
POLICY_SUMMARY_DATASET = "negbiodb_ct_stage_a_saved_output_policy_summary_v1"
REPO_ROOT = Path(__file__).resolve().parents[1]
RELEASE_MANIFEST = REPO_ROOT / "release/public_release_manifest.json"
DEFAULT_NEXT_DECISION = "post_training/stage_a_saved_output_next_decision_2026-07-10.json"
DEFAULT_ARBITRATION = "post_training/stage_a_saved_output_candidate_arbitration_2026-07-10.json"
MODEL_POLICY_NAMES = (
    "raw_candidate_top1",
    "calibrated_candidate_top1",
    "train_selected_score_gap_gate",
)
RUNTIME_BASELINE_POLICY_NAMES = (
    "evidence_gate_override",
    "hybrid_evidence_then_train_gate",
)
POLICY_SUMMARY_SOURCE_KINDS = (
    "prediction-summary",
    "candidate-gate-summary",
    "candidate-arbitration-policy",
)
PUBLIC_SAFETY_CONTRACT_FLAGS = (
    "raw_prediction_jsonl_read",
    "raw_candidate_score_jsonl_read",
    "scheduler_logs_read",
    "model_state_read",
    "ignored_run_folder_read",
)
EXTERNAL_RAW_ACCESS_FLAGS = (
    "raw_model_outputs_used",
    "raw_run_folders_used",
)
EXTERNAL_RAW_COMMIT_FLAGS = (
    "raw_predictions_committed",
    "raw_candidate_scores_committed",
    "raw_eval_report_committed",
    "raw_scheduler_logs_committed",
    "model_state_committed",
)
POLICY_COUNT_FIELDS = (
    "exact",
    "rows",
    "trusted_candidate",
    "trusted_candidate_incorrect",
)


def load_json(path: Union[str, Path]) -> Dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def sha256_file(path: Union[str, Path]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_json_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def require_json_int(
    payload: Mapping[str, Any],
    field: str,
    *,
    context: str,
) -> int:
    if field not in payload:
        raise ValueError(f"{context} missing integer field: {field}")
    value = payload[field]
    if not is_json_int(value):
        raise ValueError(f"{context} field {field} must be a JSON integer")
    return value


def public_manifest_entry(path: str) -> Optional[Mapping[str, Any]]:
    if not RELEASE_MANIFEST.exists():
        return None
    manifest = load_json(RELEASE_MANIFEST)
    artifacts = manifest.get("public_artifacts", [])
    if not isinstance(artifacts, list):
        return None
    for entry in artifacts:
        if isinstance(entry, MappingABC) and entry.get("path") == path:
            return entry
    return None


def repo_relative_source_path(source_report: str) -> Tuple[Optional[Path], Optional[str]]:
    source_path = Path(source_report)
    if source_path.is_absolute():
        return None, "policy_summary_source_report_not_repo_relative"
    if ".." in source_path.parts:
        return None, "policy_summary_source_report_not_repo_relative"
    resolved = (REPO_ROOT / source_path).resolve()
    try:
        resolved.relative_to(REPO_ROOT.resolve())
    except ValueError:
        return None, "policy_summary_source_report_not_repo_relative"
    return source_path, None


def source_report_sha_violation(summary: Mapping[str, Any]) -> Optional[str]:
    source_report = summary.get("source_report")
    expected_sha = summary.get("source_report_sha256")
    if not isinstance(source_report, str) or not source_report:
        return "policy_summary_source_report_missing"
    if not isinstance(expected_sha, str) or len(expected_sha) != 64:
        return "policy_summary_source_report_sha256_missing"
    source_path, source_path_violation = repo_relative_source_path(source_report)
    if source_path_violation:
        return source_path_violation
    assert source_path is not None
    full_source_path = REPO_ROOT / source_path
    if not full_source_path.exists():
        return "policy_summary_source_report_unavailable"
    actual_sha = sha256_file(full_source_path)
    if actual_sha != expected_sha:
        return "policy_summary_source_report_sha256_mismatch"
    manifest_entry = public_manifest_entry(source_report)
    if manifest_entry is None or manifest_entry.get("safe_to_publish") is not True:
        return "policy_summary_source_report_not_public_manifested"
    if manifest_entry.get("sha256") != actual_sha:
        return "policy_summary_source_report_manifest_sha256_mismatch"
    return None


def external_policy_contract_violations(summary: Mapping[str, Any]) -> List[str]:
    violations = []  # type: List[str]
    if summary.get("dataset") != POLICY_SUMMARY_DATASET:
        violations.append("policy_summary_dataset_mismatch")
    if summary.get("source_kind") not in POLICY_SUMMARY_SOURCE_KINDS:
        violations.append("policy_summary_source_kind_invalid")
    sha_violation = source_report_sha_violation(summary)
    if sha_violation:
        violations.append(sha_violation)
    contract = summary.get("public_safety_contract")
    if not isinstance(contract, MappingABC):
        violations.append("policy_summary_public_safety_contract_missing")
    else:
        for flag in PUBLIC_SAFETY_CONTRACT_FLAGS:
            if contract.get(flag) is not False:
                violations.append(f"policy_summary_public_safety_contract_{flag}_not_false")
    missing_raw_commit_flags = summary.get("missing_raw_commit_flags", ())
    if not isinstance(missing_raw_commit_flags, list):
        missing_raw_commit_flags = ()
    for flag in missing_raw_commit_flags:
        if flag in EXTERNAL_RAW_COMMIT_FLAGS:
            violations.append(f"policy_summary_{flag}_missing")
    return violations


def policy_summary(arbitration: Mapping[str, Any], policy: str) -> Dict[str, Any]:
    summary = arbitration.get("summary", {})
    if not isinstance(summary, MappingABC):
        raise ValueError("arbitration report missing summary object")
    by_policy = summary.get("by_policy", {})
    if not isinstance(by_policy, MappingABC):
        raise ValueError("arbitration report missing summary.by_policy object")
    row = by_policy.get(policy, {})
    if not isinstance(row, MappingABC):
        raise ValueError(f"arbitration report missing policy summary: {policy}")
    return {
        "policy": policy,
        "exact": require_json_int(row, "exact", context=f"arbitration policy {policy}"),
        "rows": require_json_int(row, "rows", context=f"arbitration policy {policy}"),
        "trusted_candidate": require_json_int(
            row,
            "trusted_candidate",
            context=f"arbitration policy {policy}",
        ),
        "trusted_candidate_incorrect": require_json_int(
            row,
            "trusted_candidate_incorrect",
            context=f"arbitration policy {policy}",
        ),
        "error_case_ids": list(row.get("error_case_ids", ())),
    }


def gate_requirements(next_decision: Mapping[str, Any]) -> Dict[str, Any]:
    decision = next_decision.get("decision", {})
    if not isinstance(decision, MappingABC):
        raise ValueError("next-decision report missing decision object")
    criteria = decision.get("minimum_success_criteria_for_next_cayuga_checkpoint", {})
    if not isinstance(criteria, MappingABC):
        raise ValueError("next-decision report missing minimum success criteria")
    return {
        "selected_next_step": decision.get("selected_next_step"),
        "candidate_or_model_policy_exact_min": require_json_int(
            criteria,
            "candidate_or_model_policy_exact_min",
            context="next-decision minimum success criteria",
        ),
        "trusted_candidate_incorrect": require_json_int(
            criteria,
            "trusted_candidate_incorrect",
            context="next-decision minimum success criteria",
        ),
        "hidden_labels_used_by_arbitration": bool(criteria.get("hidden_labels_used_by_arbitration", True)),
        "raw_predictions_remain_uncommitted": bool(criteria.get("raw_predictions_remain_uncommitted", False)),
    }


def evaluate_policy(
    summary: Mapping[str, Any],
    requirements: Mapping[str, Any],
    *,
    require_external_contract: bool = False,
) -> Dict[str, Any]:
    violations = []  # type: List[str]
    if require_external_contract:
        violations.extend(external_policy_contract_violations(summary))
    counts = {
        field: summary.get(field, 0)
        for field in POLICY_COUNT_FIELDS
    }
    non_integer_fields = [
        field for field, value in counts.items() if not is_json_int(value)
    ]
    if non_integer_fields:
        violations.append("non_integer_policy_count")
        return {
            **dict(summary),
            "passes_gate": False,
            "violations": violations,
            "non_integer_count_fields": non_integer_fields,
        }
    exact = int(counts["exact"])
    rows = int(counts["rows"])
    trusted_candidate = int(counts["trusted_candidate"])
    trusted_candidate_incorrect = int(counts["trusted_candidate_incorrect"])
    row_count_required = requirements.get("candidate_or_model_policy_rows_required")
    if min(exact, rows, trusted_candidate, trusted_candidate_incorrect) < 0:
        violations.append("negative_policy_count")
    if exact > rows:
        violations.append("exact_exceeds_rows")
    if trusted_candidate > rows:
        violations.append("trusted_candidate_exceeds_rows")
    if trusted_candidate_incorrect > trusted_candidate:
        violations.append("trusted_candidate_incorrect_exceeds_trusted_candidate")
    if row_count_required is not None and rows != int(row_count_required):
        violations.append("rows_mismatch_runtime_baseline")
    if int(summary["exact"]) < int(requirements["candidate_or_model_policy_exact_min"]):
        violations.append("below_runtime_arbitration_exact_min")
    if int(summary["trusted_candidate_incorrect"]) > int(requirements["trusted_candidate_incorrect"]):
        violations.append("unsafe_candidate_trust")
    return {
        **dict(summary),
        "passes_gate": not violations,
        "violations": violations,
    }


def compact_policy_from_payload(
    payload: Mapping[str, Any],
    *,
    path: Union[str, Path],
    source_id: Optional[str] = None,
) -> Dict[str, Any]:
    policy = payload.get("policy") or payload.get("policy_name") or payload.get("name")
    if not isinstance(policy, str) or not policy:
        raise ValueError(f"{path} missing compact policy name")
    try:
        exact = payload["exact"]
        rows = payload["rows"]
        trusted_candidate_incorrect = payload["trusted_candidate_incorrect"]
    except KeyError as exc:
        raise ValueError(f"{path} missing exact/rows/trusted_candidate_incorrect summary fields") from exc
    return {
        "policy": policy,
        "exact": exact,
        "rows": rows,
        "trusted_candidate": payload.get("trusted_candidate", 0),
        "trusted_candidate_incorrect": trusted_candidate_incorrect,
        "error_case_ids": list(payload.get("error_case_ids", ())),
        "dataset": payload.get("dataset"),
        "source_kind": payload.get("source_kind"),
        "source_report": payload.get("source_report"),
        "source_report_sha256": payload.get("source_report_sha256"),
        "source_path": source_id or str(path),
        "source_sha256": sha256_file(path),
        "public_safety_contract": payload.get("public_safety_contract"),
        "missing_raw_commit_flags": [
            flag for flag in EXTERNAL_RAW_COMMIT_FLAGS if flag not in payload
        ],
        "raw_model_outputs_used": bool(payload.get("raw_model_outputs_used", False)),
        "raw_run_folders_used": bool(payload.get("raw_run_folders_used", False)),
        **{
            flag: bool(payload.get(flag, False))
            for flag in EXTERNAL_RAW_COMMIT_FLAGS
        },
    }


def load_external_policy_summaries(paths: Sequence[Union[str, Path]]) -> List[Dict[str, Any]]:
    summaries = []  # type: List[Dict[str, Any]]
    for path in paths:
        payload = load_json(path)
        if isinstance(payload.get("policies"), list):
            for index, item in enumerate(payload["policies"], start=1):
                if not isinstance(item, MappingABC):
                    raise ValueError(f"{path} policies[{index}] is not an object")
                summaries.append(
                    compact_policy_from_payload(
                        item,
                        path=path,
                        source_id=f"{path}#policies[{index}]",
                    )
                )
        else:
            summaries.append(compact_policy_from_payload(payload, path=path))
    return summaries


def build_report(
    *,
    next_decision_path: Union[str, Path],
    arbitration_path: Union[str, Path],
    external_policy_summary_paths: Sequence[Union[str, Path]] = (),
    model_policy_names: Sequence[str] = MODEL_POLICY_NAMES,
    runtime_policy_names: Sequence[str] = RUNTIME_BASELINE_POLICY_NAMES,
) -> Dict[str, Any]:
    next_decision = load_json(next_decision_path)
    arbitration = load_json(arbitration_path)
    requirements = gate_requirements(next_decision)
    hidden_labels_used = bool(arbitration.get("hidden_labels_used_by_arbitration", True))
    raw_run_folders_used = False
    raw_model_outputs_used = False
    raw_predictions_committed = False

    runtime_baselines = [policy_summary(arbitration, policy) for policy in runtime_policy_names]
    baseline_exact = max((int(row["exact"]) for row in runtime_baselines), default=0)
    baseline_rows = max((int(row["rows"]) for row in runtime_baselines), default=0)
    requirements = {
        **requirements,
        "candidate_or_model_policy_rows_required": baseline_rows,
    }
    model_policy_summaries = [policy_summary(arbitration, policy) for policy in model_policy_names]
    model_policies = [evaluate_policy(summary, requirements) for summary in model_policy_summaries]
    external_policy_summaries = load_external_policy_summaries(external_policy_summary_paths)
    external_policies = [
        evaluate_policy(
            summary,
            requirements,
            require_external_contract=True,
        )
        for summary in external_policy_summaries
    ]
    policies_under_test = model_policies + external_policies
    any_model_policy_passes = any(bool(row["passes_gate"]) for row in policies_under_test)
    gate_violations = []  # type: List[str]
    if hidden_labels_used != requirements["hidden_labels_used_by_arbitration"]:
        gate_violations.append("hidden_label_use_mismatch")
    if raw_run_folders_used or raw_model_outputs_used:
        gate_violations.append("raw_artifact_access")
    if requirements["raw_predictions_remain_uncommitted"] and raw_predictions_committed:
        gate_violations.append("raw_prediction_policy_mismatch")
    for row in external_policy_summaries:
        if any(bool(row[flag]) for flag in EXTERNAL_RAW_ACCESS_FLAGS):
            gate_violations.append(f"{row['policy']}:raw_artifact_access")
        if requirements["raw_predictions_remain_uncommitted"]:
            for flag in EXTERNAL_RAW_COMMIT_FLAGS:
                if bool(row[flag]):
                    gate_violations.append(f"{row['policy']}:{flag}")
    if not any_model_policy_passes:
        gate_violations.append("no_model_policy_meets_runtime_baseline")

    return {
        "dataset": DATASET,
        "input_artifacts": {
            "next_decision": {
                "path": str(next_decision_path),
                "sha256": sha256_file(next_decision_path),
            },
            "candidate_arbitration": {
                "path": str(arbitration_path),
                "sha256": sha256_file(arbitration_path),
            },
        },
        "raw_model_outputs_used": raw_model_outputs_used,
        "raw_run_folders_used": raw_run_folders_used,
        "raw_predictions_committed": raw_predictions_committed,
        "hidden_labels_used_by_arbitration": hidden_labels_used,
        "requirements": requirements,
        "runtime_baseline": {
            "policies": runtime_baselines,
            "exact_min_to_meet_or_beat": baseline_exact,
            "rows_required_to_compare": baseline_rows,
        },
        "model_policies_under_test": policies_under_test,
        "external_policy_summary_artifacts": [
            {
                "path": str(path),
                "sha256": sha256_file(path),
            }
            for path in external_policy_summary_paths
        ],
        "future_policy_input_contract": {
            "required_fields": [
                "dataset",
                "policy",
                "source_kind",
                "source_report",
                "source_report_sha256",
                "exact",
                "rows",
                "trusted_candidate_incorrect",
                "public_safety_contract",
            ],
            "required_dataset": POLICY_SUMMARY_DATASET,
            "allowed_source_kinds": list(POLICY_SUMMARY_SOURCE_KINDS),
            "source_provenance_rules": [
                "source_report must be a repo-relative public manifest path",
                "source_report_sha256 must match the source_report contents",
                "release/public_release_manifest.json must mark source_report safe_to_publish with the same SHA-256",
            ],
            "public_safe_flags": [
                "raw_model_outputs_used=false",
                "raw_run_folders_used=false",
                "raw_predictions_committed=false",
                "raw_candidate_scores_committed=false",
                "raw_eval_report_committed=false",
                "raw_scheduler_logs_committed=false",
                "model_state_committed=false",
            ],
            "numeric_validity_rules": [
                "exact, rows, trusted_candidate, and trusted_candidate_incorrect must be non-negative JSON integers, not strings, floats, or booleans",
                "exact <= rows",
                "trusted_candidate <= rows",
                "trusted_candidate_incorrect <= trusted_candidate",
                "rows must equal the runtime baseline rows for this gate",
            ],
        },
        "passes_gate": not gate_violations,
        "gate_violations": gate_violations,
        "scientific_readout": {
            "diagnostic_question": (
                "Do current saved-output model/candidate policies meet or beat the "
                "runtime evidence arbitration baseline under the next-decision gate?"
            ),
            "interpretation_rule": (
                "A model-heavy policy must reach the runtime baseline exactness, "
                "have zero unsafe candidate trust, avoid hidden labels, and keep raw "
                "prediction artifacts uncommitted."
            ),
            "current_decision": (
                "Current raw/calibrated/score-gap candidate policies fail this gate; "
                "runtime evidence arbitration remains the baseline for the next Cayuga output."
            ),
        },
    }


def write_json(path: Union[str, Path], payload: Mapping[str, Any]) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_markdown(report: Mapping[str, Any], path: Union[str, Path]) -> None:
    req = report["requirements"]
    lines = [
        "# Stage A Saved-Output Meet-Or-Beat Gate",
        "",
        "Purpose: turn the saved-output next-decision checkpoint into a reusable",
        "acceptance gate for future model-heavy Cayuga outputs.",
        "",
        "## Requirements",
        "",
        f"- Selected next step: `{req['selected_next_step']}`",
        f"- Candidate/model exact minimum: {req['candidate_or_model_policy_exact_min']}",
        f"- Trusted candidate incorrect maximum: {req['trusted_candidate_incorrect']}",
        f"- Hidden labels used by arbitration: `{req['hidden_labels_used_by_arbitration']}`",
        f"- Raw predictions remain uncommitted: `{req['raw_predictions_remain_uncommitted']}`",
        "",
        "## Runtime Baseline",
        "",
        "| Policy | Exact | Rows | Trusted candidate incorrect |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in report["runtime_baseline"]["policies"]:
        lines.append(
            f"| `{row['policy']}` | {row['exact']} | {row['rows']} | {row['trusted_candidate_incorrect']} |"
        )
    lines.extend(
        [
            "",
            "## Model Policies Under Test",
            "",
            "| Policy | Exact | Rows | Trusted candidate incorrect | Passes gate | Violations |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in report["model_policies_under_test"]:
        lines.append(
            (
                f"| `{row['policy']}` | {row['exact']} | {row['rows']} | "
                f"{row['trusted_candidate_incorrect']} | {int(bool(row['passes_gate']))} | "
                f"`{json.dumps(row['violations'])}` |"
            )
        )
    contract = report["future_policy_input_contract"]
    lines.extend(
        [
            "",
            "## Future Policy Input Contract",
            "",
            "- Required fields: "
            f"`{json.dumps(contract['required_fields'])}`",
            "- Required dataset: "
            f"`{contract['required_dataset']}`",
            "- Allowed source kinds: "
            f"`{json.dumps(contract['allowed_source_kinds'])}`",
            "- Source provenance rules: "
            f"`{json.dumps(contract['source_provenance_rules'])}`",
            "- Public-safe flags: "
            f"`{json.dumps(contract['public_safe_flags'])}`",
            "- Numeric validity rules: "
            f"`{json.dumps(contract['numeric_validity_rules'])}`",
            "",
            "## Decision",
            "",
            f"- Passes gate: `{report['passes_gate']}`",
            f"- Gate violations: `{json.dumps(report['gate_violations'])}`",
            "",
            str(report["scientific_readout"]["current_decision"]),
            "",
            "Artifact policy: raw saved predictions, candidate-score JSONL, scheduler logs,",
            "model state, and ignored run folders stay uncommitted.",
        ]
    )
    Path(path).write_text("\n".join(lines) + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--next-decision", default=DEFAULT_NEXT_DECISION)
    parser.add_argument("--candidate-arbitration", default=DEFAULT_ARBITRATION)
    parser.add_argument(
        "--model-policy-summary",
        action="append",
        default=None,
        help="Compact public-safe future model policy summary JSON. Repeatable.",
    )
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--out-md", default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    report = build_report(
        next_decision_path=args.next_decision,
        arbitration_path=args.candidate_arbitration,
        external_policy_summary_paths=args.model_policy_summary or (),
    )
    if args.out_json:
        write_json(args.out_json, report)
    if args.out_md:
        write_markdown(report, args.out_md)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

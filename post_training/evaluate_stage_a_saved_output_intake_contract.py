#!/usr/bin/env python3
"""Verify the compact Stage A saved-output intake contract.

This report checks whether the current saved-output checkpoint bundle is ready
to accept a future Cayuga compact policy summary into the meet-or-beat gate. It
reads only compact public artifacts and hashes. It does not read raw prediction
JSONL, raw candidate-score JSONL, scheduler logs, ignored run folders, or model
state.
"""

import argparse
import hashlib
import json
from collections.abc import Mapping as MappingABC
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple, Union


DATASET = "negbiodb_ct_stage_a_saved_output_intake_contract_v1"
DEFAULT_NEXT_DECISION = "post_training/stage_a_saved_output_next_decision_2026-07-10.json"
DEFAULT_MEET_OR_BEAT_GATE = "post_training/stage_a_saved_output_meet_or_beat_gate_2026-07-10.json"
EXPECTED_NEXT_ARTIFACTS = (  # type: Tuple[str, ...]
    "compact saved-output summary",
    "compact candidate calibration summary",
    "compact candidate arbitration summary",
    "updated saved-output next-decision report",
)
REQUIRED_PUBLIC_SAFE_FLAGS = (  # type: Tuple[str, ...]
    "raw_model_outputs_used=false",
    "raw_run_folders_used=false",
    "raw_predictions_committed=false",
    "raw_candidate_scores_committed=false",
    "raw_eval_report_committed=false",
    "raw_scheduler_logs_committed=false",
    "model_state_committed=false",
)
REQUIRED_POLICY_SUMMARY_FIELDS = (  # type: Tuple[str, ...]
    "dataset",
    "policy",
    "source_kind",
    "source_report",
    "source_report_sha256",
    "exact",
    "rows",
    "trusted_candidate_incorrect",
    "public_safety_contract",
)
REQUIRED_POLICY_SUMMARY_DATASET = "negbiodb_ct_stage_a_saved_output_policy_summary_v1"
REQUIRED_POLICY_SUMMARY_SOURCE_KINDS = (  # type: Tuple[str, ...]
    "prediction-summary",
    "candidate-gate-summary",
    "candidate-arbitration-policy",
)
REQUIRED_SOURCE_PROVENANCE_RULES = (  # type: Tuple[str, ...]
    "source_report must be a repo-relative public manifest path",
    "source_report_sha256 must match the source_report contents",
    "release/public_release_manifest.json must mark source_report safe_to_publish with the same SHA-256",
)


def load_json(path: Union[str, Path]) -> Dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError("%s is not a JSON object" % path)
    return payload


def sha256_file(path: Union[str, Path]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_ref_status(ref: Mapping[str, Any]) -> Dict[str, Any]:
    path = str(ref.get("path", ""))
    expected_sha = str(ref.get("sha256", ""))
    exists = bool(path) and Path(path).exists()
    actual_sha = sha256_file(path) if exists else None
    return {
        "path": path,
        "exists": exists,
        "expected_sha256": expected_sha,
        "actual_sha256": actual_sha,
        "sha256_matches": bool(exists and expected_sha and actual_sha == expected_sha),
    }


def same_existing_file(left: Union[str, Path], right: Union[str, Path]) -> bool:
    left_path = Path(left)
    right_path = Path(right)
    return bool(left_path.exists() and right_path.exists() and left_path.resolve() == right_path.resolve())


def collect_input_artifact_status(next_decision: Mapping[str, Any]) -> Dict[str, Any]:
    input_artifacts = next_decision.get("input_artifacts", {})
    if not isinstance(input_artifacts, MappingABC):
        raise ValueError("next-decision report missing input_artifacts object")
    out = {}  # type: Dict[str, Any]
    for key in ("readiness", "candidate_calibration", "candidate_arbitration"):
        ref = input_artifacts.get(key)
        if not isinstance(ref, MappingABC):
            out[key] = {
                "path": None,
                "exists": False,
                "expected_sha256": None,
                "actual_sha256": None,
                "sha256_matches": False,
            }
        else:
            out[key] = artifact_ref_status(ref)
    gate_refs = []
    raw_gates = input_artifacts.get("candidate_gates", [])
    if isinstance(raw_gates, list):
        for ref in raw_gates:
            if isinstance(ref, MappingABC):
                gate_refs.append(artifact_ref_status(ref))
    out["candidate_gates"] = gate_refs
    return out


def decision_object(next_decision: Mapping[str, Any]) -> Mapping[str, Any]:
    decision = next_decision.get("decision", {})
    if not isinstance(decision, MappingABC):
        raise ValueError("next-decision report missing decision object")
    return decision


def criteria_match(
    *,
    next_decision: Mapping[str, Any],
    meet_or_beat_gate: Mapping[str, Any],
) -> bool:
    decision = decision_object(next_decision)
    criteria = decision.get("minimum_success_criteria_for_next_cayuga_checkpoint", {})
    if not isinstance(criteria, MappingABC):
        return False
    requirements = meet_or_beat_gate.get("requirements", {})
    if not isinstance(requirements, MappingABC):
        return False
    expected = {
        "selected_next_step": decision.get("selected_next_step"),
        "candidate_or_model_policy_exact_min": int(criteria.get("candidate_or_model_policy_exact_min", 0)),
        "trusted_candidate_incorrect": int(criteria.get("trusted_candidate_incorrect", 0)),
        "hidden_labels_used_by_arbitration": bool(criteria.get("hidden_labels_used_by_arbitration", True)),
        "raw_predictions_remain_uncommitted": bool(criteria.get("raw_predictions_remain_uncommitted", False)),
    }
    return all(requirements.get(key) == value for key, value in expected.items())


def meet_or_beat_ref_status(
    *,
    next_decision_path: Union[str, Path],
    meet_or_beat_gate: Mapping[str, Any],
) -> Dict[str, Any]:
    refs = meet_or_beat_gate.get("input_artifacts", {})
    if not isinstance(refs, MappingABC):
        raise ValueError("meet-or-beat report missing input_artifacts object")
    out = {}  # type: Dict[str, Any]
    for key in ("next_decision", "candidate_arbitration"):
        ref = refs.get(key)
        out[key] = artifact_ref_status(ref) if isinstance(ref, MappingABC) else {
            "path": None,
            "exists": False,
            "expected_sha256": None,
            "actual_sha256": None,
            "sha256_matches": False,
        }
    out["next_decision_points_to_current_file"] = (
        same_existing_file(out["next_decision"]["path"], next_decision_path)
    )
    return out


def public_flag_contract(meet_or_beat_gate: Mapping[str, Any]) -> Dict[str, Any]:
    contract = meet_or_beat_gate.get("future_policy_input_contract", {})
    if not isinstance(contract, MappingABC):
        raise ValueError("meet-or-beat report missing future_policy_input_contract")
    flags = contract.get("public_safe_flags", [])
    if not isinstance(flags, list):
        flags = []
    missing = [flag for flag in REQUIRED_PUBLIC_SAFE_FLAGS if flag not in flags]
    return {
        "required_flags": list(REQUIRED_PUBLIC_SAFE_FLAGS),
        "observed_flags": list(flags),
        "missing_required_flags": missing,
        "complete": not missing,
    }


def adapter_contract(meet_or_beat_gate: Mapping[str, Any]) -> Dict[str, Any]:
    contract = meet_or_beat_gate.get("future_policy_input_contract", {})
    if not isinstance(contract, MappingABC):
        raise ValueError("meet-or-beat report missing future_policy_input_contract")
    fields = contract.get("required_fields", [])
    if not isinstance(fields, list):
        fields = []
    source_kinds = contract.get("allowed_source_kinds", [])
    if not isinstance(source_kinds, list):
        source_kinds = []
    source_rules = contract.get("source_provenance_rules", [])
    if not isinstance(source_rules, list):
        source_rules = []
    missing_fields = [field for field in REQUIRED_POLICY_SUMMARY_FIELDS if field not in fields]
    missing_source_kinds = [
        kind for kind in REQUIRED_POLICY_SUMMARY_SOURCE_KINDS if kind not in source_kinds
    ]
    missing_source_rules = [
        rule for rule in REQUIRED_SOURCE_PROVENANCE_RULES if rule not in source_rules
    ]
    dataset_matches = contract.get("required_dataset") == REQUIRED_POLICY_SUMMARY_DATASET
    return {
        "required_fields": list(REQUIRED_POLICY_SUMMARY_FIELDS),
        "observed_required_fields": list(fields),
        "missing_required_fields": missing_fields,
        "required_dataset": REQUIRED_POLICY_SUMMARY_DATASET,
        "observed_required_dataset": contract.get("required_dataset"),
        "required_dataset_matches": dataset_matches,
        "required_source_kinds": list(REQUIRED_POLICY_SUMMARY_SOURCE_KINDS),
        "observed_source_kinds": list(source_kinds),
        "missing_source_kinds": missing_source_kinds,
        "required_source_provenance_rules": list(REQUIRED_SOURCE_PROVENANCE_RULES),
        "observed_source_provenance_rules": list(source_rules),
        "missing_source_provenance_rules": missing_source_rules,
        "complete": bool(
            not missing_fields
            and dataset_matches
            and not missing_source_kinds
            and not missing_source_rules
        ),
    }


def append_violation(violations: List[str], condition: bool, label: str) -> None:
    if condition:
        violations.append(label)


def build_report(
    *,
    next_decision_path: Union[str, Path],
    meet_or_beat_gate_path: Union[str, Path],
) -> Dict[str, Any]:
    next_decision = load_json(next_decision_path)
    meet_or_beat_gate = load_json(meet_or_beat_gate_path)
    decision = decision_object(next_decision)
    input_status = collect_input_artifact_status(next_decision)
    meet_refs = meet_or_beat_ref_status(
        next_decision_path=next_decision_path,
        meet_or_beat_gate=meet_or_beat_gate,
    )
    flag_contract = public_flag_contract(meet_or_beat_gate)
    policy_adapter_contract = adapter_contract(meet_or_beat_gate)
    next_required = list(decision.get("next_artifacts_required", []))
    violations = []  # type: List[str]

    append_violation(
        violations,
        next_required != list(EXPECTED_NEXT_ARTIFACTS),
        "next_artifacts_required_mismatch",
    )
    for key in ("readiness", "candidate_calibration", "candidate_arbitration"):
        append_violation(
            violations,
            not bool(input_status[key]["sha256_matches"]),
            "%s_sha256_mismatch" % key,
        )
    append_violation(
        violations,
        not input_status["candidate_gates"],
        "candidate_gate_refs_missing",
    )
    for index, status in enumerate(input_status["candidate_gates"], start=1):
        append_violation(
            violations,
            not bool(status["sha256_matches"]),
            "candidate_gate_%d_sha256_mismatch" % index,
        )
    append_violation(
        violations,
        not bool(meet_refs["next_decision"]["sha256_matches"]),
        "meet_or_beat_next_decision_sha256_mismatch",
    )
    append_violation(
        violations,
        not bool(meet_refs["candidate_arbitration"]["sha256_matches"]),
        "meet_or_beat_candidate_arbitration_sha256_mismatch",
    )
    append_violation(
        violations,
        not bool(meet_refs["next_decision_points_to_current_file"]),
        "meet_or_beat_next_decision_path_mismatch",
    )
    append_violation(
        violations,
        not criteria_match(next_decision=next_decision, meet_or_beat_gate=meet_or_beat_gate),
        "meet_or_beat_requirements_do_not_match_next_decision",
    )
    append_violation(
        violations,
        bool(meet_or_beat_gate.get("raw_model_outputs_used", True)),
        "meet_or_beat_raw_model_outputs_used",
    )
    append_violation(
        violations,
        bool(meet_or_beat_gate.get("raw_run_folders_used", True)),
        "meet_or_beat_raw_run_folders_used",
    )
    append_violation(
        violations,
        bool(meet_or_beat_gate.get("raw_predictions_committed", True)),
        "meet_or_beat_raw_predictions_committed",
    )
    append_violation(
        violations,
        bool(meet_or_beat_gate.get("hidden_labels_used_by_arbitration", True)),
        "hidden_labels_used_by_arbitration",
    )
    append_violation(
        violations,
        not bool(flag_contract["complete"]),
        "future_policy_public_safe_flags_incomplete",
    )
    append_violation(
        violations,
        not bool(policy_adapter_contract["complete"]),
        "future_policy_adapter_contract_incomplete",
    )

    return {
        "dataset": DATASET,
        "input_artifacts": {
            "next_decision": {
                "path": str(next_decision_path),
                "sha256": sha256_file(next_decision_path),
            },
            "meet_or_beat_gate": {
                "path": str(meet_or_beat_gate_path),
                "sha256": sha256_file(meet_or_beat_gate_path),
            },
        },
        "raw_model_outputs_used": False,
        "raw_run_folders_used": False,
        "raw_prediction_jsonl_read": False,
        "raw_candidate_score_jsonl_read": False,
        "scheduler_logs_read": False,
        "model_state_read": False,
        "ignored_run_folder_read": False,
        "next_artifacts_required": next_required,
        "expected_next_artifacts_required": list(EXPECTED_NEXT_ARTIFACTS),
        "next_decision_selected_step": decision.get("selected_next_step"),
        "next_decision_input_artifacts": input_status,
        "meet_or_beat_input_artifacts": meet_refs,
        "future_policy_public_safe_contract": flag_contract,
        "future_policy_adapter_contract": policy_adapter_contract,
        "criteria_match": criteria_match(next_decision=next_decision, meet_or_beat_gate=meet_or_beat_gate),
        "passes_contract": not violations,
        "violations": violations,
        "scientific_readout": {
            "diagnostic_question": (
                "Is the compact saved-output checkpoint bundle internally "
                "consistent and ready to accept a future Cayuga policy summary "
                "without raw artifacts?"
            ),
            "interpretation_rule": (
                "Only compact reports and hashes are checked here. Passing this "
                "contract means the intake surface is reproducible; it does not "
                "mean the current model policy meets the runtime baseline."
            ),
            "next_decision": (
                "Use build_stage_a_saved_output_policy_summary.py for the next "
                "compact Cayuga policy summary, then pass it to the meet-or-beat "
                "gate before reopening tool_query, DPO/RLVR, HF publication, or "
                "release tagging."
            ),
        },
    }


def write_json(path: Union[str, Path], payload: Mapping[str, Any]) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_markdown(report: Mapping[str, Any], path: Union[str, Path]) -> None:
    lines = [
        "# Stage A Saved-Output Intake Contract",
        "",
        "Purpose: verify that the compact saved-output checkpoint bundle is",
        "internally consistent before any future Cayuga policy summary is",
        "judged by the meet-or-beat gate.",
        "",
        "## Contract",
        "",
        "- Passes contract: `%s`" % report["passes_contract"],
        "- Violations: `%s`" % json.dumps(report["violations"]),
        "- Selected next step: `%s`" % report["next_decision_selected_step"],
        "",
        "## Required Artifacts",
        "",
        "- Expected: `%s`" % json.dumps(report["expected_next_artifacts_required"]),
        "- Observed: `%s`" % json.dumps(report["next_artifacts_required"]),
        "",
        "## Input Hashes",
        "",
        "| Artifact | Path | Hash match |",
        "| --- | --- | ---: |",
    ]
    for key, status in report["next_decision_input_artifacts"].items():
        if key == "candidate_gates":
            for index, gate in enumerate(status, start=1):
                lines.append(
                    "| `candidate_gate_%d` | `%s` | `%s` |"
                    % (index, gate["path"], gate["sha256_matches"])
                )
        else:
            lines.append("| `%s` | `%s` | `%s` |" % (key, status["path"], status["sha256_matches"]))
    meet = report["meet_or_beat_input_artifacts"]
    lines.extend(
        [
            "| `meet_or_beat.next_decision` | `%s` | `%s` |"
            % (meet["next_decision"]["path"], meet["next_decision"]["sha256_matches"]),
            "| `meet_or_beat.candidate_arbitration` | `%s` | `%s` |"
            % (
                meet["candidate_arbitration"]["path"],
                meet["candidate_arbitration"]["sha256_matches"],
            ),
            "",
            "## Future Policy Flags",
            "",
            "- Required public-safe flags: `%s`"
            % json.dumps(report["future_policy_public_safe_contract"]["required_flags"]),
            "- Missing public-safe flags: `%s`"
            % json.dumps(report["future_policy_public_safe_contract"]["missing_required_flags"]),
            "",
            "## Future Policy Adapter Contract",
            "",
            "- Required dataset: `%s`"
            % report["future_policy_adapter_contract"]["required_dataset"],
            "- Observed dataset: `%s`"
            % report["future_policy_adapter_contract"]["observed_required_dataset"],
            "- Missing required fields: `%s`"
            % json.dumps(report["future_policy_adapter_contract"]["missing_required_fields"]),
            "- Missing source kinds: `%s`"
            % json.dumps(report["future_policy_adapter_contract"]["missing_source_kinds"]),
            "- Missing source provenance rules: `%s`"
            % json.dumps(
                report["future_policy_adapter_contract"]["missing_source_provenance_rules"]
            ),
            "",
            "## Decision",
            "",
            report["scientific_readout"]["next_decision"],
            "",
            "Artifact policy: this verifier reads compact reports only; raw",
            "prediction JSONL, candidate-score JSONL, scheduler logs, model state,",
            "and ignored run folders remain out of scope.",
        ]
    )
    Path(path).write_text("\n".join(lines) + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--next-decision", default=DEFAULT_NEXT_DECISION)
    parser.add_argument("--meet-or-beat-gate", default=DEFAULT_MEET_OR_BEAT_GATE)
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--out-md", default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    report = build_report(
        next_decision_path=args.next_decision,
        meet_or_beat_gate_path=args.meet_or_beat_gate,
    )
    if args.out_json:
        write_json(args.out_json, report)
    if args.out_md:
        write_markdown(report, args.out_md)
    print(
        json.dumps(
            {
                "dataset": report["dataset"],
                "passes_contract": report["passes_contract"],
                "violations": report["violations"],
                "criteria_match": report["criteria_match"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

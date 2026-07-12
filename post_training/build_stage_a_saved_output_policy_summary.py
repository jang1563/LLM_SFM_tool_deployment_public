#!/usr/bin/env python3
"""Build compact Stage A policy summaries for the meet-or-beat gate.

This adapter is for future Cayuga saved-output checkpoints. It reads only
public-safe compact summaries and emits the small JSON contract consumed by
`evaluate_stage_a_saved_output_meet_or_beat_gate.py --model-policy-summary`.
It does not read raw prediction JSONL, raw candidate-score JSONL, scheduler
logs, ignored run folders, or model state.
"""

import argparse
import hashlib
import json
from collections.abc import Mapping as MappingABC
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Union


DATASET = "negbiodb_ct_stage_a_saved_output_policy_summary_v1"
REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_KINDS = (  # type: Tuple[str, ...]
    "prediction-summary",
    "candidate-gate-summary",
    "candidate-arbitration-policy",
)
PUBLIC_ARTIFACT_COMMIT_FLAGS = (  # type: Tuple[str, ...]
    "raw_predictions_committed",
    "raw_candidate_scores_committed",
    "raw_eval_report_committed",
    "raw_scheduler_logs_committed",
    "model_state_committed",
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


def public_source_report_path(path: Union[str, Path]) -> str:
    source_path = Path(path)
    try:
        return source_path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


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


def optional_json_int(
    payload: Mapping[str, Any],
    field: str,
    *,
    context: str,
    default: int = 0,
) -> int:
    if field not in payload:
        return default
    return require_json_int(payload, field, context=context)


def artifact_policy(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    policy = payload.get("artifact_policy", {})
    return policy if isinstance(policy, MappingABC) else {}


def bool_policy(policy: Mapping[str, Any], key: str, *, default: bool = False) -> bool:
    value = policy.get(key, default)
    return bool(value)


def public_artifact_flags(payload: Mapping[str, Any]) -> Dict[str, bool]:
    policy = artifact_policy(payload)
    return {
        key: bool(payload.get(key, bool_policy(policy, key)))
        for key in PUBLIC_ARTIFACT_COMMIT_FLAGS
    }


def default_policy_name(payload: Mapping[str, Any], source_path: Union[str, Path]) -> str:
    value = payload.get("run_id") or payload.get("policy") or payload.get("candidate_policy")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return Path(source_path).stem


def policy_from_prediction_summary(
    payload: Mapping[str, Any],
    *,
    source_path: Union[str, Path],
    policy_name: Optional[str],
) -> Dict[str, Any]:
    result = payload.get("result")
    if not isinstance(result, MappingABC):
        raise ValueError(f"{source_path} missing result object")
    policy = artifact_policy(payload)
    return {
        "policy": policy_name or default_policy_name(payload, source_path),
        "exact": require_json_int(result, "passed", context=f"{source_path} result"),
        "rows": require_json_int(result, "cases", context=f"{source_path} result"),
        "trusted_candidate": 0,
        "trusted_candidate_incorrect": 0,
        "error_case_ids": list(result.get("error_case_ids", ())),
        "source_kind": "prediction-summary",
        "source_report": public_source_report_path(source_path),
        "source_report_sha256": sha256_file(source_path),
        "raw_model_outputs_used": False,
        "raw_run_folders_used": False,
        **public_artifact_flags(payload),
    }


def policy_from_candidate_gate_summary(
    payload: Mapping[str, Any],
    *,
    source_path: Union[str, Path],
    policy_name: Optional[str],
    gate_report_key: str,
) -> Dict[str, Any]:
    report = payload.get(gate_report_key)
    if not isinstance(report, MappingABC):
        raise ValueError(f"{source_path} missing {gate_report_key} object")
    rows_source = payload if "cases" in payload else report
    rows_field = "cases" if "cases" in payload else "rows"
    return {
        "policy": policy_name or f"{default_policy_name(payload, source_path)}::{gate_report_key}",
        "exact": require_json_int(
            report,
            "strict_final_correct",
            context=f"{source_path} {gate_report_key}",
        ),
        "rows": require_json_int(
            rows_source,
            rows_field,
            context=f"{source_path} {gate_report_key}",
        ),
        "trusted_candidate": optional_json_int(
            report,
            "trusted",
            context=f"{source_path} {gate_report_key}",
        ),
        "trusted_candidate_incorrect": optional_json_int(
            report,
            "trusted_incorrect",
            context=f"{source_path} {gate_report_key}",
        ),
        "error_case_ids": list(report.get("unsafe_trust_case_ids", ())),
        "source_kind": "candidate-gate-summary",
        "source_report": public_source_report_path(source_path),
        "source_report_sha256": sha256_file(source_path),
        "candidate_policy": payload.get("candidate_policy"),
        "gate_report_key": gate_report_key,
        "raw_model_outputs_used": False,
        "raw_run_folders_used": False,
        **public_artifact_flags(payload),
    }


def policy_from_candidate_arbitration(
    payload: Mapping[str, Any],
    *,
    source_path: Union[str, Path],
    policy_name: str,
) -> Dict[str, Any]:
    summary = payload.get("summary", {})
    if not isinstance(summary, MappingABC):
        raise ValueError(f"{source_path} missing summary object")
    by_policy = summary.get("by_policy", {})
    if not isinstance(by_policy, MappingABC):
        raise ValueError(f"{source_path} missing summary.by_policy object")
    row = by_policy.get(policy_name)
    if not isinstance(row, MappingABC):
        raise ValueError(f"{source_path} missing policy summary: {policy_name}")
    return {
        "policy": policy_name,
        "exact": require_json_int(
            row,
            "exact",
            context=f"{source_path} policy {policy_name}",
        ),
        "rows": require_json_int(
            row,
            "rows",
            context=f"{source_path} policy {policy_name}",
        ),
        "trusted_candidate": optional_json_int(
            row,
            "trusted_candidate",
            context=f"{source_path} policy {policy_name}",
        ),
        "trusted_candidate_incorrect": optional_json_int(
            row,
            "trusted_candidate_incorrect",
            context=f"{source_path} policy {policy_name}",
        ),
        "error_case_ids": list(row.get("error_case_ids", ())),
        "source_kind": "candidate-arbitration-policy",
        "source_report": public_source_report_path(source_path),
        "source_report_sha256": sha256_file(source_path),
        "raw_model_outputs_used": False,
        "raw_run_folders_used": False,
        **public_artifact_flags(payload),
    }


def build_policy_summary(
    *,
    source_path: Union[str, Path],
    source_kind: str,
    policy_name: Optional[str] = None,
    gate_report_key: str = "best_default_zero_unsafe_report",
) -> Dict[str, Any]:
    payload = load_json(source_path)
    if source_kind == "prediction-summary":
        summary = policy_from_prediction_summary(
            payload,
            source_path=source_path,
            policy_name=policy_name,
        )
    elif source_kind == "candidate-gate-summary":
        summary = policy_from_candidate_gate_summary(
            payload,
            source_path=source_path,
            policy_name=policy_name,
            gate_report_key=gate_report_key,
        )
    elif source_kind == "candidate-arbitration-policy":
        if not policy_name:
            raise ValueError("--policy is required for candidate-arbitration-policy")
        summary = policy_from_candidate_arbitration(
            payload,
            source_path=source_path,
            policy_name=policy_name,
        )
    else:
        raise ValueError(f"Unsupported source kind: {source_kind}")
    return {
        "dataset": DATASET,
        **summary,
        "public_safety_contract": {
            "raw_prediction_jsonl_read": False,
            "raw_candidate_score_jsonl_read": False,
            "scheduler_logs_read": False,
            "model_state_read": False,
            "ignored_run_folder_read": False,
        },
    }


def write_json(path: Union[str, Path], payload: Mapping[str, Any]) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Compact public-safe source report JSON.")
    parser.add_argument("--source-kind", required=True, choices=SOURCE_KINDS)
    parser.add_argument("--policy", default=None, help="Output policy name, or arbitration policy to extract.")
    parser.add_argument(
        "--gate-report-key",
        default="best_default_zero_unsafe_report",
        help="Candidate-gate report object to summarize.",
    )
    parser.add_argument("--out", required=True)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    report = build_policy_summary(
        source_path=args.source,
        source_kind=args.source_kind,
        policy_name=args.policy,
        gate_report_key=args.gate_report_key,
    )
    write_json(args.out, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

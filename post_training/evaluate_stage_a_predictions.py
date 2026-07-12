#!/usr/bin/env python3
"""Score Stage A prediction JSONL files against the trajectory evaluator.

The input file is intentionally model-provider agnostic. API calls, local SFT
checkpoints, prompt-only baselines, or hand-written debug rows can all emit a
JSONL file with one prediction per case. This script only parses and scores
those saved outputs; it never calls a live API or loads model weights.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm_sfm_tool_deployment import (
    Action,
    EvidencePacket,
    EvidenceStatus,
    ToolStep,
    Trajectory,
)
from negbiodb_ct.model_output import ModelOutputParseError, parse_model_output_json
from negbiodb_ct.stage_a_manifest import score_stage_a_trajectory
from post_training.run_stage_a_sft_smoke_eval import (
    load_jsonl,
    load_manifest_rows,
    summarize_rows,
    trajectory_from_payload,
    write_json,
)


DATASET = "negbiodb_ct_stage_a_prediction_eval_v1"

ACTION_CLASS_DEFAULT_STATUS = {
    "ground": EvidenceStatus.SUPPORTED.value,
    "reject": EvidenceStatus.CONTRADICTED.value,
    "defer": EvidenceStatus.INSUFFICIENT.value,
    "verify": EvidenceStatus.INSUFFICIENT.value,
    "flag": EvidenceStatus.INVALID_VALUE.value,
    "self_answer": EvidenceStatus.UNKNOWN.value,
}

ACTION_TO_TERMINAL = {
    "ground": Action.GROUND_WITH_ATTRIBUTION.value,
    "reject": Action.REJECT_OR_FLAG_UNSUPPORTED_CLAIM.value,
    "defer": Action.DEFER_OR_REQUEST_MORE_EVIDENCE.value,
    "verify": Action.VERIFY_WITH_ASSAY_OR_DATABASE.value,
    "flag": Action.REJECT_OR_FLAG_UNSUPPORTED_CLAIM.value,
    "self_answer": Action.ANSWER_SELF.value,
}

ACTION_ALIASES = {
    "answer_self": "self_answer",
    "self_answer": "self_answer",
    "ground": "ground",
    "ground_with_attribution": "ground",
    "reject": "reject",
    "reject_or_flag_unsupported_claim": "reject",
    "flag": "flag",
    "defer": "defer",
    "defer_or_request_more_evidence": "defer",
    "verify": "verify",
    "verify_with_assay_or_database": "verify",
}

EVIDENCE_STATUS_ALIASES = {
    "supported": EvidenceStatus.SUPPORTED.value,
    "ground": EvidenceStatus.SUPPORTED.value,
    "contradicted": EvidenceStatus.CONTRADICTED.value,
    "reject": EvidenceStatus.CONTRADICTED.value,
    "invalid_value": EvidenceStatus.INVALID_VALUE.value,
    "flag": EvidenceStatus.INVALID_VALUE.value,
    "insufficient": EvidenceStatus.INSUFFICIENT.value,
    "insufficient_evidence": EvidenceStatus.INSUFFICIENT.value,
    "defer": EvidenceStatus.INSUFFICIENT.value,
    "verify": EvidenceStatus.INSUFFICIENT.value,
    "unknown": EvidenceStatus.UNKNOWN.value,
}

SIMPLIFIED_TOOL_ALIASES = {
    "search_failures": (
        "nullatlas_survey_prior_failures",
        "nullatlas_verify_trial_claims",
    ),
    "check_other_indications": ("nullatlas_negative_evidence_completeness",),
    "check_value_validity": ("nullatlas_check_value_validity",),
}


def prediction_case_id(row: Mapping[str, Any]) -> str:
    """Return the Stage A case id carried by a prediction or SFT row."""

    for key in ("case_id", "source_manifest_case_id", "task_id", "id"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            if key == "id" and value.startswith("stage_a_sft::"):
                return value.removeprefix("stage_a_sft::")
            return value
    raise ValueError("Prediction row is missing case_id/source_manifest_case_id/task_id.")


def index_predictions(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    """Index prediction rows and fail fast on duplicate case ids."""

    indexed: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        case_id = prediction_case_id(row)
        if case_id in indexed:
            raise ValueError(f"Duplicate prediction for case_id={case_id}")
        indexed[case_id] = row
    return indexed


def expected_case_ids_from_rows(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Extract expected case ids while preserving file order."""

    seen: set[str] = set()
    case_ids: list[str] = []
    for row in rows:
        case_id = prediction_case_id(row)
        if case_id in seen:
            raise ValueError(f"Duplicate expected case_id={case_id}")
        seen.add(case_id)
        case_ids.append(case_id)
    return case_ids


def trajectory_from_prediction_row(row: Mapping[str, Any], *, case_id: str) -> Trajectory:
    """Convert a prediction row into the shared scoreable trajectory schema."""

    payload = prediction_payload(row)
    if "trajectory" in payload:
        return trajectory_from_payload(payload["trajectory"], target_input_id=case_id)
    if "target_trajectory" in payload:
        return trajectory_from_payload(payload["target_trajectory"], target_input_id=case_id)
    if "model_output" in payload:
        return trajectory_from_compact_model_output(payload["model_output"], case_id=case_id)
    if "raw_output" in payload:
        return trajectory_from_compact_model_output(payload["raw_output"], case_id=case_id)
    return trajectory_from_compact_model_output(payload, case_id=case_id)


def prediction_payload(row: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the nested prediction object when present, otherwise the row."""

    payload = row.get("prediction")
    if payload is None:
        return row
    if not isinstance(payload, Mapping):
        raise ModelOutputParseError("prediction must be a JSON object when present.")
    return payload


def trajectory_from_compact_model_output(raw_output: Any, *, case_id: str) -> Trajectory:
    """Parse compact model output JSON into a Stage A trajectory.

    Compact rows may use `tool_calls` as strings or objects. String-only tool
    calls remain scoreable, but they fail Stage A query-completeness gates
    because no tool arguments were emitted.
    """

    obj = parse_model_output_json(raw_output)
    action_class = normalise_action(obj.get("action") or obj.get("terminal_action"))
    status = normalise_status(
        obj.get("evidence_status", obj.get("predicted_evidence_status")),
        action_class=action_class,
    )
    return Trajectory(
        input_id=case_id,
        steps=tuple(tool_steps_from_compact_output(obj)),
        evidence_packet=EvidencePacket(
            input_id=case_id,
            representation_type="drug_indication_claim",
            negative_evidence_status=status,
            claim_guard_status=claim_guard_status(obj),
        ),
        terminal_action=ACTION_TO_TERMINAL[action_class],
        cited_source_ids=tuple(normalise_citations(obj)),
        predicted_evidence_status=status,
        rationale=str(obj["rationale"]) if "rationale" in obj else None,
    )


def normalise_action(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ModelOutputParseError(
            "Model output must include an action or terminal_action string."
        )
    key = value.strip().lower()
    try:
        return ACTION_ALIASES[key]
    except KeyError as exc:
        raise ModelOutputParseError(f"Unknown model action: {value!r}") from exc


def normalise_status(value: Any, *, action_class: str) -> str:
    if value is None:
        return ACTION_CLASS_DEFAULT_STATUS[action_class]
    if not isinstance(value, str):
        raise ModelOutputParseError("evidence_status must be a string when provided.")
    key = value.strip().lower()
    try:
        return EVIDENCE_STATUS_ALIASES[key]
    except KeyError as exc:
        raise ModelOutputParseError(f"Unknown evidence_status: {value!r}") from exc


def tool_steps_from_compact_output(obj: Mapping[str, Any]) -> list[ToolStep]:
    raw_calls = obj.get("tool_calls", obj.get("called", obj.get("steps", ())))
    if raw_calls is None:
        return []
    if not isinstance(raw_calls, list):
        raise ModelOutputParseError("tool_calls/called/steps must be a list.")

    steps: list[ToolStep] = []
    for call in raw_calls:
        names, arguments, observation = parse_tool_call(call)
        for name in names:
            steps.append(ToolStep(name=name, arguments=arguments, observation=observation))
    return steps


def parse_tool_call(
    call: Any,
) -> tuple[tuple[str, ...], Mapping[str, Any], Mapping[str, Any]]:
    if isinstance(call, str):
        raw_name = call
        arguments: Mapping[str, Any] = {}
        observation: Mapping[str, Any] = {"status": "completed"}
    elif isinstance(call, Mapping):
        raw_name = call.get("name") or call.get("tool")
        arguments_obj = call.get("arguments", call.get("args", {}))
        observation_obj = call.get("observation", {"status": "completed"})
        if not isinstance(arguments_obj, Mapping):
            raise ModelOutputParseError("tool call arguments must be an object.")
        if not isinstance(observation_obj, Mapping):
            raise ModelOutputParseError("tool call observation must be an object.")
        arguments = dict(arguments_obj)
        observation = dict(observation_obj)
    else:
        raise ModelOutputParseError("Each tool call must be a string or object.")

    if not isinstance(raw_name, str) or not raw_name.strip():
        raise ModelOutputParseError("Each tool call must include a non-empty name.")
    key = raw_name.strip()
    names = SIMPLIFIED_TOOL_ALIASES.get(key, (key,))
    return names, arguments, observation


def claim_guard_status(obj: Mapping[str, Any]) -> str:
    if obj.get("tool_calls") or obj.get("steps") or obj.get("called"):
        return "checked"
    return "unchecked"


def normalise_citations(obj: Mapping[str, Any]) -> list[str]:
    raw = obj.get("cited_source_ids", obj.get("cited_sources"))
    if raw is None:
        raw = obj.get("nct") or obj.get("cited_nct")
    if raw is None:
        return []
    if isinstance(raw, str):
        values = [raw]
    elif isinstance(raw, list):
        values = raw
    else:
        raise ModelOutputParseError("Citations must be a string or list.")

    citations: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text and text not in seen:
            seen.add(text)
            citations.append(text)
    return citations


def build_report(
    *,
    manifest_rows: Sequence[Mapping[str, Any]],
    prediction_rows: Sequence[Mapping[str, Any]],
    expected_case_ids: Sequence[str] | None = None,
    run_id: str = "stage_a_predictions",
) -> dict[str, Any]:
    manifest_by_case = {str(row["case_id"]): row for row in manifest_rows}
    predictions_by_case = index_predictions(prediction_rows)
    expected = (
        list(expected_case_ids)
        if expected_case_ids is not None
        else list(predictions_by_case)
    )

    row_reports = []
    for case_id in expected:
        manifest_row = manifest_by_case.get(case_id)
        prediction_row = predictions_by_case.get(case_id)
        if manifest_row is None:
            row_reports.append(
                failed_row(
                    case_id,
                    ("unknown_case_id",),
                    prediction_row=prediction_row,
                )
            )
            continue
        if prediction_row is None:
            row_reports.append(
                failed_row(
                    case_id,
                    ("missing_prediction",),
                    manifest_row=manifest_row,
                )
            )
            continue
        row_reports.append(
            score_prediction_row(manifest_row, prediction_row, case_id=case_id)
        )

    expected_set = set(expected)
    for case_id, prediction_row in predictions_by_case.items():
        if case_id not in expected_set:
            row_reports.append(
                failed_row(
                    case_id,
                    ("unexpected_prediction_case_id",),
                    manifest_row=manifest_by_case.get(case_id),
                    prediction_row=prediction_row,
                )
            )

    parse_errors = Counter(
        row.get("parse_error")
        for row in row_reports
        if row.get("parse_error")
    )
    return {
        "dataset": DATASET,
        "run_id": run_id,
        "boundary": (
            "Offline Stage A prediction scorer. Inputs are saved model/API/local "
            "outputs; this script performs no live API calls and loads no model weights."
        ),
        "cases_expected": len(expected),
        "predictions_received": len(prediction_rows),
        "summary": summarize_rows(row_reports),
        "parse_errors": dict(sorted(parse_errors.items())),
        "rows": row_reports,
    }


def score_prediction_row(
    manifest_row: Mapping[str, Any],
    prediction_row: Mapping[str, Any],
    *,
    case_id: str,
) -> dict[str, Any]:
    try:
        trajectory = trajectory_from_prediction_row(prediction_row, case_id=case_id)
    except (KeyError, TypeError, ValueError, ModelOutputParseError) as exc:
        return failed_row(
            case_id,
            ("prediction_parse_error",),
            manifest_row=manifest_row,
            prediction_row=prediction_row,
            parse_error=str(exc),
        )

    result = score_stage_a_trajectory(manifest_row, trajectory)
    hidden = manifest_row["hidden_eval_metadata"]
    return {
        "case_id": case_id,
        "split": prediction_row.get("split"),
        "prediction_source": prediction_source(prediction_row),
        "case_family": str(hidden.get("case_family")),
        "gold_evidence_status": str(hidden.get("gold_evidence_status")),
        "expected_terminal_action": str(hidden.get("expected_terminal_action")),
        "predicted_evidence_status": str(trajectory.predicted_evidence_status),
        "predicted_terminal_action": str(trajectory.terminal_action),
        "score": round(result.score, 3),
        "passed": result.passed,
        "reward_breakdown": dict(result.reward_breakdown),
        "violations": list(result.violations),
    }


def failed_row(
    case_id: str,
    violations: Sequence[str],
    *,
    manifest_row: Mapping[str, Any] | None = None,
    prediction_row: Mapping[str, Any] | None = None,
    parse_error: str | None = None,
) -> dict[str, Any]:
    hidden = manifest_row.get("hidden_eval_metadata", {}) if manifest_row else {}
    return {
        "case_id": case_id,
        "split": prediction_row.get("split") if prediction_row else None,
        "prediction_source": prediction_source(prediction_row) if prediction_row else None,
        "case_family": str(hidden.get("case_family")) if hidden else None,
        "gold_evidence_status": str(hidden.get("gold_evidence_status")) if hidden else None,
        "expected_terminal_action": str(hidden.get("expected_terminal_action")) if hidden else None,
        "predicted_evidence_status": None,
        "predicted_terminal_action": None,
        "score": 0.0,
        "passed": False,
        "reward_breakdown": {},
        "violations": list(violations),
        "parse_error": parse_error,
    }


def prediction_source(row: Mapping[str, Any]) -> Any:
    return row.get("source", row.get("model", row.get("run_id")))


def print_summary(report: Mapping[str, Any]) -> None:
    summary = report["summary"]
    print("run_id                 expected  received  passed  mean_score")
    print("---------------------  --------  --------  ------  ----------")
    print(
        f"{str(report['run_id']):<21}  "
        f"{report['cases_expected']:<8}  "
        f"{report['predictions_received']:<8}  "
        f"{summary['passed']:<6}  "
        f"{summary['mean_score']:<10.3f}"
    )
    if summary["violations"]:
        print("violations=", json.dumps(summary["violations"], sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="negbiodb_ct/stage_a_mini_manifest.jsonl")
    parser.add_argument("--predictions", required=True)
    parser.add_argument(
        "--expected-sft",
        default=None,
        help=(
            "Optional SFT JSONL whose source_manifest_case_id/task_id values "
            "define the required evaluation cases."
        ),
    )
    parser.add_argument(
        "--require-all-manifest",
        action="store_true",
        help="Require predictions for every case in the Stage A manifest.",
    )
    parser.add_argument("--run-id", default="stage_a_predictions")
    parser.add_argument("--out", default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    manifest_rows = load_manifest_rows(args.manifest)
    expected_case_ids = None
    if args.expected_sft:
        expected_case_ids = expected_case_ids_from_rows(load_jsonl(args.expected_sft))
    if args.require_all_manifest:
        expected_case_ids = [str(row["case_id"]) for row in manifest_rows]

    report = build_report(
        manifest_rows=manifest_rows,
        prediction_rows=load_jsonl(args.predictions),
        expected_case_ids=expected_case_ids,
        run_id=args.run_id,
    )
    if args.out:
        write_json(args.out, report)
    if args.json:
        print(json.dumps(report["summary"], indent=2, sort_keys=True))
    else:
        print_summary(report)


if __name__ == "__main__":
    main()

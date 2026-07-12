"""Adapter from NegBioDB-CT pilot tasks to the generic trajectory schema."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from llm_sfm_tool_deployment import (
    Action,
    CalibrationStatus,
    EvidencePacket,
    EvidenceStatus,
    TaskSpec,
    ToolStep,
    Trajectory,
)


CT_REQUIRED_TOOL_LOOP = (
    "nullatlas_survey_prior_failures",
    "nullatlas_verify_trial_claims",
    "nullatlas_check_value_validity",
    "nullatlas_negative_evidence_completeness",
)

CT_NATIVE_TOOL_SEARCH = "search_failures"
CT_NATIVE_TOOL_OTHER_INDICATIONS = "check_other_indications"
CT_NATIVE_TOOL_LOOP = (
    CT_NATIVE_TOOL_SEARCH,
    CT_NATIVE_TOOL_OTHER_INDICATIONS,
)

CT_ACTION_TO_STATUS = {
    "ground": EvidenceStatus.SUPPORTED,
    "reject": EvidenceStatus.CONTRADICTED,
    "defer": EvidenceStatus.INSUFFICIENT,
    "verify": EvidenceStatus.INSUFFICIENT,
    "flag": EvidenceStatus.INVALID_VALUE,
}

CT_ACTION_TO_TERMINAL = {
    "ground": Action.GROUND_WITH_ATTRIBUTION,
    "reject": Action.REJECT_OR_FLAG_UNSUPPORTED_CLAIM,
    "defer": Action.DEFER_OR_REQUEST_MORE_EVIDENCE,
    "verify": Action.VERIFY_WITH_ASSAY_OR_DATABASE,
    "flag": Action.REJECT_OR_FLAG_UNSUPPORTED_CLAIM,
}


def load_task_records(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    """Load compact NegBioDB-CT task records from JSONL."""

    records: list[dict[str, Any]] = []
    with Path(path).open() as handle:
        for line in handle:
            if not line.strip():
                continue
            records.append(json.loads(line))
            if limit is not None and len(records) >= limit:
                break
    return records


def task_spec_from_record(
    record: Mapping[str, Any],
    *,
    tool_profile: str = "nullatlas_full",
) -> TaskSpec:
    """Build hidden evaluator metadata from a NegBioDB-CT task record."""

    action_class = _action_class(record)
    scoring_key = _scoring_key(record)
    gold_nct = scoring_key.get("gold_nct")
    gold_source_ids = (gold_nct,) if gold_nct else ()

    return TaskSpec(
        input_id=str(record["packet_id"]),
        claim=str(record["observation"]["claim"]),
        required_tools=required_tools_for_action(action_class, tool_profile=tool_profile),
        gold_evidence_status=CT_ACTION_TO_STATUS[action_class],
        expected_terminal_action=CT_ACTION_TO_TERMINAL[action_class],
        gold_source_ids=gold_source_ids,
        requires_attribution=action_class in {"ground", "flag"},
        requires_external_tool=True,
        web_zero=False,
    )


def required_tools_for_action(
    action_class: str,
    *,
    tool_profile: str = "nullatlas_full",
) -> tuple[str, ...]:
    """Return the required tool trace for a CT task under a tool profile.

    `nullatlas_full` is the post-training target: the model should express the
    full evidence loop. `native_ct` is the runner interface: `search_failures`
    collapses survey/attribution/value evidence into one native tool, and
    `check_other_indications` is only required when the terminal action depends
    on evidence absence for the exact indication.
    """

    action_class = _normalise_action_class(action_class)
    if tool_profile == "nullatlas_full":
        return CT_REQUIRED_TOOL_LOOP
    if tool_profile == "native_ct":
        if action_class in {"defer", "verify"}:
            return CT_NATIVE_TOOL_LOOP
        return (CT_NATIVE_TOOL_SEARCH,)
    raise ValueError(f"Unknown CT tool profile: {tool_profile!r}")


def ideal_trajectory_from_record(record: Mapping[str, Any]) -> Trajectory:
    """Create an oracle-style trajectory for evaluator smoke tests.

    This is not a model output. It is a deterministic positive control that says:
    if the task builder's gold action is followed and the full NullAtlas loop is
    present, the evaluator should pass the trajectory.
    """

    action_class = _action_class(record)
    scoring_key = _scoring_key(record)
    status = CT_ACTION_TO_STATUS[action_class]
    terminal = CT_ACTION_TO_TERMINAL[action_class]
    gold_nct = scoring_key.get("gold_nct")

    return Trajectory(
        input_id=str(record["packet_id"]),
        steps=_tool_loop_steps(record),
        evidence_packet=_evidence_packet(record, status),
        terminal_action=terminal,
        cited_source_ids=(gold_nct,) if gold_nct else (),
        predicted_evidence_status=status,
    )


def prediction_trajectory_from_record(
    record: Mapping[str, Any],
    predicted_action_class: str,
    *,
    tool_names: Sequence[str] = (),
    cited_source_ids: Sequence[str] = (),
    predicted_evidence_status: EvidenceStatus | str | None = None,
) -> Trajectory:
    """Convert a simple model/policy prediction into a scoreable trajectory."""

    if predicted_action_class == "self_answer":
        terminal = Action.ANSWER_SELF
        status = predicted_evidence_status or EvidenceStatus.UNKNOWN
    else:
        terminal = CT_ACTION_TO_TERMINAL[predicted_action_class]
        status = predicted_evidence_status or CT_ACTION_TO_STATUS[predicted_action_class]

    return Trajectory(
        input_id=str(record["packet_id"]),
        steps=tuple(ToolStep(name=name) for name in tool_names),
        evidence_packet=_evidence_packet(record, status),
        terminal_action=terminal,
        cited_source_ids=tuple(cited_source_ids),
        predicted_evidence_status=status,
    )


def _tool_loop_steps(record: Mapping[str, Any]) -> tuple[ToolStep, ...]:
    observation = record["observation"]
    args = {
        "drug_id": observation["drug_id"],
        "condition_id": observation["condition_id"],
    }
    return tuple(
        ToolStep(
            name=name,
            arguments=args,
            observation={"status": "completed"},
        )
        for name in CT_REQUIRED_TOOL_LOOP
    )


def _evidence_packet(
    record: Mapping[str, Any], evidence_status: EvidenceStatus | str
) -> EvidencePacket:
    observation = record["observation"]
    return EvidencePacket(
        input_id=str(record["packet_id"]),
        representation_type="drug_indication_claim",
        calibration_status=CalibrationStatus.NOT_APPLICABLE,
        negative_evidence_status=evidence_status,
        hidden_truth_pointer=str(record["packet_id"]),
        specialist_name="NullAtlas-CT",
        specialist_output={
            "drug_id": observation["drug_id"],
            "condition_id": observation["condition_id"],
        },
    )


def _action_class(record: Mapping[str, Any]) -> str:
    return _normalise_action_class(str(record["action_class"]))


def _normalise_action_class(action_class: str) -> str:
    if action_class not in CT_ACTION_TO_STATUS:
        raise ValueError(f"Unknown NegBioDB-CT action class: {action_class!r}")
    return action_class


def _scoring_key(record: Mapping[str, Any]) -> Mapping[str, Any]:
    scoring_key = record.get("scoring_key")
    if not isinstance(scoring_key, Mapping):
        raise ValueError("NegBioDB-CT record is missing a scoring_key mapping.")
    return scoring_key

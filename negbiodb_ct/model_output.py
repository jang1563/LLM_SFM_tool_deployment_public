"""Parse NegBioDB-CT model outputs into scoreable trajectories."""

from __future__ import annotations

import json
import re
from typing import Any, Mapping

from llm_sfm_tool_deployment import EvidenceStatus, Trajectory

from .adapter import (
    CT_ACTION_TO_STATUS,
    CT_NATIVE_TOOL_OTHER_INDICATIONS,
    CT_NATIVE_TOOL_SEARCH,
    CT_REQUIRED_TOOL_LOOP,
    prediction_trajectory_from_record,
)


class ModelOutputParseError(ValueError):
    """Raised when a model output cannot be converted to a trajectory."""


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
    "supported": EvidenceStatus.SUPPORTED,
    "ground": EvidenceStatus.SUPPORTED,
    "contradicted": EvidenceStatus.CONTRADICTED,
    "reject": EvidenceStatus.CONTRADICTED,
    "invalid_value": EvidenceStatus.INVALID_VALUE,
    "flag": EvidenceStatus.INVALID_VALUE,
    "insufficient": EvidenceStatus.INSUFFICIENT,
    "insufficient_evidence": EvidenceStatus.INSUFFICIENT,
    "defer": EvidenceStatus.INSUFFICIENT,
    "verify": EvidenceStatus.INSUFFICIENT,
    "unknown": EvidenceStatus.UNKNOWN,
}

SIMPLIFIED_TOOL_ALIASES = {
    "search_failures": (
        "nullatlas_survey_prior_failures",
        "nullatlas_verify_trial_claims",
    ),
    "check_other_indications": ("nullatlas_negative_evidence_completeness",),
    "check_value_validity": ("nullatlas_check_value_validity",),
}


NATIVE_TOOL_ALIASES = {
    "search_failures": (CT_NATIVE_TOOL_SEARCH,),
    "check_other_indications": (CT_NATIVE_TOOL_OTHER_INDICATIONS,),
    "check_value_validity": ("check_value_validity",),
}


def parse_model_output_json(raw: str | Mapping[str, Any]) -> dict[str, Any]:
    """Parse a JSON object from a model response.

    The parser accepts either an already-decoded mapping or a raw string. For raw
    strings, it tolerates Markdown code fences and extra prose around the first
    JSON object.
    """

    if isinstance(raw, Mapping):
        return dict(raw)

    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    match = re.search(r"\{.*\}", text, re.S)
    if match is None:
        raise ModelOutputParseError("No JSON object found in model output.")

    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise ModelOutputParseError(f"Invalid JSON model output: {exc}") from exc

    if not isinstance(obj, dict):
        raise ModelOutputParseError("Model output JSON must be an object.")
    return obj


def trajectory_from_model_output(
    record: Mapping[str, Any],
    raw_output: str | Mapping[str, Any],
    *,
    tool_profile: str = "nullatlas_full",
) -> Trajectory:
    """Convert prompt-only model JSON into the shared trajectory schema.

    Expected compact schema:

    ```json
    {
      "action": "ground|reject|defer|verify|flag|self_answer",
      "evidence_status": "supported|contradicted|insufficient|invalid_value|unknown",
      "tool_calls": ["nullatlas_survey_prior_failures", "..."],
      "cited_source_ids": ["NCT..."],
      "rationale": "short optional rationale"
    }
    ```

    For compatibility with the existing pilot runner, `called`, `nct`, and
    `cited_nct` are also accepted.
    """

    obj = parse_model_output_json(raw_output)
    action = _normalise_action(obj.get("action") or obj.get("terminal_action"))
    status = _normalise_status(obj.get("evidence_status"), action)
    tool_names = _normalise_tool_calls(obj, tool_profile=tool_profile)
    cited = _normalise_citations(obj)

    trajectory = prediction_trajectory_from_record(
        record,
        action,
        tool_names=tool_names,
        cited_source_ids=cited,
        predicted_evidence_status=status,
    )
    return Trajectory(
        input_id=trajectory.input_id,
        steps=trajectory.steps,
        evidence_packet=trajectory.evidence_packet,
        terminal_action=trajectory.terminal_action,
        cited_source_ids=trajectory.cited_source_ids,
        predicted_evidence_status=trajectory.predicted_evidence_status,
        rationale=str(obj["rationale"]) if "rationale" in obj else None,
    )


def _normalise_action(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ModelOutputParseError("Model output must include an action string.")
    key = value.strip().lower()
    try:
        return ACTION_ALIASES[key]
    except KeyError as exc:
        raise ModelOutputParseError(f"Unknown model action: {value!r}") from exc


def _normalise_status(value: Any, action: str) -> EvidenceStatus:
    if value is None:
        return CT_ACTION_TO_STATUS.get(action, EvidenceStatus.UNKNOWN)
    if not isinstance(value, str):
        raise ModelOutputParseError("evidence_status must be a string when provided.")
    key = value.strip().lower()
    try:
        return EVIDENCE_STATUS_ALIASES[key]
    except KeyError as exc:
        raise ModelOutputParseError(f"Unknown evidence_status: {value!r}") from exc


def _normalise_tool_calls(
    obj: Mapping[str, Any],
    *,
    tool_profile: str = "nullatlas_full",
) -> tuple[str, ...]:
    raw_calls = obj.get("tool_calls", obj.get("called", ()))
    if raw_calls is None:
        return ()
    if not isinstance(raw_calls, list):
        raise ModelOutputParseError("tool_calls/called must be a list.")

    names: list[str] = []
    for call in raw_calls:
        if isinstance(call, str):
            name = call
        elif isinstance(call, Mapping):
            name = call.get("name") or call.get("tool")
        else:
            raise ModelOutputParseError("Each tool call must be a string or object.")

        if not isinstance(name, str) or not name.strip():
            raise ModelOutputParseError("Each tool call must have a non-empty name.")

        key = name.strip()
        aliases = _tool_aliases(tool_profile).get(key)
        if aliases:
            names.extend(aliases)
        else:
            names.append(key)

    return tuple(_dedupe_preserve_order(names))


def _tool_aliases(tool_profile: str) -> Mapping[str, tuple[str, ...]]:
    if tool_profile == "nullatlas_full":
        return SIMPLIFIED_TOOL_ALIASES
    if tool_profile == "native_ct":
        return NATIVE_TOOL_ALIASES
    raise ModelOutputParseError(f"Unknown tool_profile: {tool_profile!r}")


def _normalise_citations(obj: Mapping[str, Any]) -> tuple[str, ...]:
    raw = obj.get("cited_source_ids", obj.get("cited_sources"))
    if raw is None:
        raw = obj.get("nct") or obj.get("cited_nct")

    if raw is None:
        return ()
    if isinstance(raw, str):
        values = [raw]
    elif isinstance(raw, list):
        values = raw
    else:
        raise ModelOutputParseError("Citations must be a string or list.")

    citations = []
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            citations.append(text)
    return tuple(_dedupe_preserve_order(citations))


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def required_model_output_schema() -> dict[str, Any]:
    """Return the compact schema to include in prompt templates."""

    return {
        "action": "ground|reject|defer|verify|flag|self_answer",
        "evidence_status": "supported|contradicted|insufficient|invalid_value|unknown",
        "tool_calls": list(CT_REQUIRED_TOOL_LOOP),
        "cited_source_ids": ["NCT00000000"],
        "rationale": "one short sentence; optional",
    }

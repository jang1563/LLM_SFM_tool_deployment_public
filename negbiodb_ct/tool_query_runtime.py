"""Fail-closed Stage A tool-query compilation.

The current Stage A protocol always executes the same four tools in a fixed
order and copies two typed identifiers into every call. That operation is a
runtime contract, not a model decision.
"""

from __future__ import annotations

from typing import Any, Mapping


TOOL_SEQUENCE = (
    "nullatlas_survey_prior_failures",
    "nullatlas_verify_trial_claims",
    "nullatlas_check_value_validity",
    "nullatlas_negative_evidence_completeness",
)
QUERY_FIELDS = ("drug_id", "condition_id")
QUERY_NAMESPACES = {
    "drug_id": "negbiodb_ct.intervention_id",
    "condition_id": "negbiodb_ct.condition_id",
}


class ToolQueryContractError(ValueError):
    """Raised when a model-visible task cannot be compiled safely."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    *,
    code: str,
) -> None:
    if set(value) != expected:
        raise ToolQueryContractError(code)


def compile_tool_query(task: Mapping[str, Any]) -> dict[str, Any]:
    """Compile a validated visible task into exact ordered tool calls."""

    if task.get("component") != "tool_query":
        raise ToolQueryContractError("wrong_component")

    allowed_tools = task.get("allowed_tools")
    if not isinstance(allowed_tools, list) or tuple(allowed_tools) != TOOL_SEQUENCE:
        raise ToolQueryContractError("tool_sequence_contract_mismatch")

    required_fields = task.get("required_query_fields")
    if (
        not isinstance(required_fields, list)
        or tuple(required_fields) != QUERY_FIELDS
    ):
        raise ToolQueryContractError("required_query_fields_mismatch")

    query = task.get("query")
    if not isinstance(query, Mapping):
        raise ToolQueryContractError("query_missing_or_not_object")
    _require_exact_keys(
        query,
        set(QUERY_FIELDS),
        code="query_field_set_mismatch",
    )

    arguments: dict[str, int] = {}
    for field in QUERY_FIELDS:
        payload = query.get(field)
        if not isinstance(payload, Mapping):
            raise ToolQueryContractError(f"{field}_payload_not_object")
        _require_exact_keys(
            payload,
            {"namespace", "value"},
            code=f"{field}_payload_keys_mismatch",
        )
        if payload.get("namespace") != QUERY_NAMESPACES[field]:
            raise ToolQueryContractError(f"{field}_namespace_mismatch")
        value = payload.get("value")
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ToolQueryContractError(f"{field}_value_not_positive_integer")
        arguments[field] = value

    return {
        "tool_calls": [
            {"name": name, "arguments": dict(arguments)}
            for name in TOOL_SEQUENCE
        ]
    }

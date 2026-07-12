"""Stage A manifest adapter for verifiable evidence-status trajectories.

This layer keeps model-visible prompts separate from hidden evaluator metadata
while reusing the generic TaskSpec / Trajectory / EvidencePacket schema.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from llm_sfm_tool_deployment import (
    Action,
    EvaluationResult,
    EvidencePacket,
    EvidenceStatus,
    TaskSpec,
    ToolStep,
    Trajectory,
    TrajectoryEvaluator,
)

from .adapter import (
    CT_ACTION_TO_STATUS,
    CT_ACTION_TO_TERMINAL,
    CT_NATIVE_TOOL_LOOP,
    CT_REQUIRED_TOOL_LOOP,
    load_task_records,
    required_tools_for_action,
)


STAGE_A_DATASET = "negbiodb_ct_stage_a_v1"
DEFAULT_COST_PROFILE = {
    "call_tool": 1.0,
    "verify_with_assay_or_database": 3.0,
    "defer_or_request_more_evidence": 0.5,
}
DEFAULT_REQUIRED_QUERY_FIELDS = ("drug_id", "condition_id")
ACTION_CLASS_ORDER = ("ground", "reject", "defer", "verify", "flag")
HIDDEN_LABEL_TOKENS = (
    "ground",
    "reject",
    "defer",
    "verify",
    "flag",
    "supported",
    "contradicted",
    "invalid_value",
    "insufficient",
    "unknown",
    "NCT",
)


def load_stage_a_manifest(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    """Load Stage A manifest rows from JSONL."""

    rows: list[dict[str, Any]] = []
    with Path(path).open() as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    return rows


def write_stage_a_manifest(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write Stage A manifest rows as sorted-key JSONL."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def build_stage_a_manifest_rows(
    records: Iterable[Mapping[str, Any]],
    *,
    per_action: int = 5,
    tool_profile: str = "nullatlas_full",
) -> list[dict[str, Any]]:
    """Build a balanced mini-manifest from NegBioDB-CT task records."""

    buckets: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        action_class = str(record["action_class"])
        if action_class in ACTION_CLASS_ORDER and len(buckets[action_class]) < per_action:
            buckets[action_class].append(record)
        if all(len(buckets[action]) >= per_action for action in ACTION_CLASS_ORDER):
            break

    rows: list[dict[str, Any]] = []
    for action_class in ACTION_CLASS_ORDER:
        for record in buckets[action_class]:
            rows.append(
                stage_a_row_from_task_record(
                    record,
                    case_index=len(rows),
                    tool_profile=tool_profile,
                )
            )
    return rows


def stage_a_row_from_task_record(
    record: Mapping[str, Any],
    *,
    case_index: int,
    tool_profile: str = "nullatlas_full",
) -> dict[str, Any]:
    """Project a NegBioDB-CT task into the Stage A manifest shape."""

    action_class = str(record["action_class"])
    observation = record["observation"]
    scoring_key = record["scoring_key"]
    case_id = f"stage_a::{case_index:06d}"
    input_id = case_id
    required_tools = required_tools_for_action(action_class, tool_profile=tool_profile)
    gold_nct = scoring_key.get("gold_nct")
    gold_source_ids = [str(gold_nct)] if gold_nct else []

    return {
        "case_id": case_id,
        "dataset": STAGE_A_DATASET,
        "model_visible_task": {
            "input_id": input_id,
            "claim": str(observation["claim"]),
            "allowed_tools": list(allowed_tools_for_profile(tool_profile)),
        },
        "hidden_eval_metadata": {
            "source_task_id": str(record["packet_id"]),
            "required_tools": list(required_tools),
            "gold_evidence_status": CT_ACTION_TO_STATUS[action_class].value,
            "expected_terminal_action": CT_ACTION_TO_TERMINAL[action_class].value,
            "gold_source_ids": gold_source_ids,
            "requires_attribution": action_class in {"ground", "flag"},
            "requires_external_tool": True,
            "web_zero": False,
            "case_family": case_family_for_action(action_class),
            "split_group": split_group_for_record(record),
            "required_query_fields": list(DEFAULT_REQUIRED_QUERY_FIELDS),
            "tool_profile": tool_profile,
        },
        "cost_profile": dict(DEFAULT_COST_PROFILE),
    }


def allowed_tools_for_profile(tool_profile: str) -> tuple[str, ...]:
    """Return the visible tool universe for a Stage A tool profile."""

    if tool_profile == "nullatlas_full":
        return CT_REQUIRED_TOOL_LOOP
    if tool_profile == "native_ct":
        return CT_NATIVE_TOOL_LOOP
    raise ValueError(f"Unknown CT tool profile: {tool_profile!r}")


def case_family_for_action(action_class: str) -> str:
    """Map action classes to manifest case families."""

    families = {
        "ground": "supported_negative_evidence",
        "reject": "contradicted_or_mixed_endpoint_claim",
        "defer": "insufficient_evidence",
        "verify": "related_evidence_requires_verification",
        "flag": "invalid_value_attribution_failure",
    }
    if action_class not in families:
        raise ValueError(f"Unknown action class: {action_class!r}")
    return families[action_class]


def split_group_for_record(record: Mapping[str, Any]) -> str:
    """Stable leakage-control group for a source record or claim cluster."""

    scoring_key = record["scoring_key"]
    observation = record["observation"]
    gold_nct = scoring_key.get("gold_nct")
    if gold_nct:
        return f"source::{gold_nct}"
    return f"drug_condition::{observation['drug_id']}::{observation['condition_id']}"


def task_spec_from_stage_a_row(row: Mapping[str, Any]) -> TaskSpec:
    """Project hidden Stage A metadata into the generic evaluator TaskSpec."""

    model_visible = _model_visible(row)
    hidden = _hidden_metadata(row)
    return TaskSpec(
        input_id=str(model_visible["input_id"]),
        claim=str(model_visible["claim"]),
        required_tools=tuple(str(tool) for tool in hidden.get("required_tools", ())),
        gold_evidence_status=str(hidden["gold_evidence_status"]),
        expected_terminal_action=str(hidden["expected_terminal_action"]),
        gold_source_ids=tuple(str(source) for source in hidden.get("gold_source_ids", ())),
        requires_attribution=bool(hidden.get("requires_attribution", False)),
        requires_external_tool=bool(hidden.get("requires_external_tool", True)),
        web_zero=bool(hidden.get("web_zero", False)),
    )


def ideal_trajectory_from_stage_a_row(row: Mapping[str, Any]) -> Trajectory:
    """Build a deterministic positive-control trajectory for a Stage A row."""

    model_visible = _model_visible(row)
    hidden = _hidden_metadata(row)
    input_id = str(model_visible["input_id"])
    status = str(hidden["gold_evidence_status"])
    terminal = str(hidden["expected_terminal_action"])
    required_tools = tuple(str(tool) for tool in hidden.get("required_tools", ()))
    required_query_fields = tuple(str(field) for field in hidden.get("required_query_fields", ()))
    step_args = {field: f"<{field}>" for field in required_query_fields}

    return Trajectory(
        input_id=input_id,
        steps=tuple(
            ToolStep(name=tool, arguments=step_args, observation={"status": "completed"})
            for tool in required_tools
        ),
        evidence_packet=EvidencePacket(
            input_id=input_id,
            representation_type="drug_indication_claim",
            negative_evidence_status=status,
            claim_guard_status="checked",
            hidden_truth_pointer=str(hidden.get("source_task_id", row["case_id"])),
        ),
        terminal_action=terminal,
        cited_source_ids=tuple(str(source) for source in hidden.get("gold_source_ids", ())),
        predicted_evidence_status=status,
    )


def score_stage_a_trajectory(row: Mapping[str, Any], trajectory: Trajectory) -> EvaluationResult:
    """Score a Stage A trajectory with generic and manifest-level checks."""

    base = TrajectoryEvaluator().evaluate(task_spec_from_stage_a_row(row), trajectory)
    hidden = _hidden_metadata(row)
    reward = dict(base.reward_breakdown)
    violations = list(base.violations)

    query_ok = _query_filter_complete(
        trajectory,
        required_tools=tuple(str(tool) for tool in hidden.get("required_tools", ())),
        required_fields=tuple(str(field) for field in hidden.get("required_query_fields", ())),
    )
    reward["query_filter_completeness"] = float(query_ok)
    if not query_ok:
        violations.append("query_filter_missing_required_field")

    return EvaluationResult(
        input_id=base.input_id,
        earned=float(sum(reward.values())),
        possible=float(len(reward)),
        reward_breakdown=reward,
        violations=tuple(violations),
    )


def failure_trajectories_for_stage_a_row(row: Mapping[str, Any]) -> dict[str, Trajectory]:
    """Generate rejected trajectory variants for DPO/process-supervision candidates."""

    ideal = ideal_trajectory_from_stage_a_row(row)
    hidden = _hidden_metadata(row)
    expected_terminal = str(hidden["expected_terminal_action"])
    gold_status = str(hidden["gold_evidence_status"])
    gold_sources = tuple(str(source) for source in hidden.get("gold_source_ids", ()))
    required_tools = tuple(str(tool) for tool in hidden.get("required_tools", ()))

    variants: dict[str, Trajectory] = {
        "self_answering_without_tools": replace(
            ideal,
            steps=(),
            evidence_packet=replace(
                ideal.evidence_packet,
                negative_evidence_status=EvidenceStatus.UNKNOWN,
                claim_guard_status="unchecked",
            ),
            terminal_action=Action.ANSWER_SELF,
            cited_source_ids=(),
            predicted_evidence_status=EvidenceStatus.UNKNOWN,
        ),
        "wrong_tool": replace(
            ideal,
            steps=(ToolStep(name="wrong_tool", arguments={"query": "shortcut"}),),
        ),
        "missing_tool": replace(
            ideal,
            steps=ideal.steps[: max(len(ideal.steps) - 1, 0)],
        ),
        "partial_query": replace(
            ideal,
            steps=tuple(ToolStep(name=step.name, arguments={}, observation=step.observation) for step in ideal.steps),
        ),
    }

    if gold_sources:
        variants["missing_attribution"] = replace(ideal, cited_source_ids=())

    if gold_status == EvidenceStatus.INVALID_VALUE.value:
        variants["invalid_value_missed"] = replace(
            ideal,
            terminal_action=Action.VERIFY_WITH_ASSAY_OR_DATABASE,
            evidence_packet=replace(ideal.evidence_packet, negative_evidence_status=EvidenceStatus.SUPPORTED),
            predicted_evidence_status=EvidenceStatus.SUPPORTED,
        )

    if expected_terminal != Action.TRUST_SPECIALIST_OUTPUT.value:
        variants["unsupported_trust"] = replace(
            ideal,
            terminal_action=Action.TRUST_SPECIALIST_OUTPUT,
            evidence_packet=replace(
                ideal.evidence_packet,
                negative_evidence_status=gold_status,
                calibration_status="uncalibrated",
            ),
            predicted_evidence_status=gold_status,
        )

    if gold_status == EvidenceStatus.INSUFFICIENT.value:
        variants["insufficient_as_negative"] = replace(
            ideal,
            terminal_action=Action.REJECT_OR_FLAG_UNSUPPORTED_CLAIM,
            evidence_packet=replace(ideal.evidence_packet, negative_evidence_status=EvidenceStatus.CONTRADICTED),
            predicted_evidence_status=EvidenceStatus.CONTRADICTED,
            cited_source_ids=(),
        )

    if not required_tools:
        variants.pop("missing_tool", None)
    return variants


def validate_stage_a_manifest(
    rows: Sequence[Mapping[str, Any]],
    *,
    min_rows: int = 1,
    min_status_count: int = 1,
    require_unique_split_groups: bool = True,
) -> list[str]:
    """Return manifest issues; an empty list means the manifest passes."""

    issues: list[str] = []
    if len(rows) < min_rows:
        issues.append(f"manifest_too_small:{len(rows)}<{min_rows}")

    seen_case_ids: set[str] = set()
    split_groups: list[str] = []
    status_counts: Counter[str] = Counter()
    attribution_required = 0
    for index, row in enumerate(rows):
        prefix = f"row[{index}]"
        issues.extend(_validate_stage_a_row(row, prefix=prefix))
        case_id = str(row.get("case_id", ""))
        if case_id in seen_case_ids:
            issues.append(f"{prefix}:duplicate_case_id:{case_id}")
        seen_case_ids.add(case_id)

        hidden = row.get("hidden_eval_metadata")
        if isinstance(hidden, Mapping):
            status_counts[str(hidden.get("gold_evidence_status", ""))] += 1
            if hidden.get("requires_attribution"):
                attribution_required += 1
            split_group = hidden.get("split_group")
            if split_group:
                split_groups.append(str(split_group))

    for status in (
        EvidenceStatus.SUPPORTED.value,
        EvidenceStatus.CONTRADICTED.value,
        EvidenceStatus.INVALID_VALUE.value,
        EvidenceStatus.INSUFFICIENT.value,
    ):
        if status_counts[status] < min_status_count:
            issues.append(f"status_underrepresented:{status}:{status_counts[status]}<{min_status_count}")

    if attribution_required < min_status_count:
        issues.append(f"attribution_cases_underrepresented:{attribution_required}<{min_status_count}")

    if require_unique_split_groups:
        duplicates = sorted(group for group, count in Counter(split_groups).items() if count > 1)
        if duplicates:
            issues.append(f"split_group_overlap:{duplicates[:5]}")

    return issues


def _validate_stage_a_row(row: Mapping[str, Any], *, prefix: str) -> list[str]:
    issues: list[str] = []
    for key in ("case_id", "dataset", "model_visible_task", "hidden_eval_metadata", "cost_profile"):
        if key not in row:
            issues.append(f"{prefix}:missing_key:{key}")

    model_visible = row.get("model_visible_task")
    hidden = row.get("hidden_eval_metadata")
    if not isinstance(model_visible, Mapping):
        issues.append(f"{prefix}:model_visible_task_not_mapping")
        return issues
    if not isinstance(hidden, Mapping):
        issues.append(f"{prefix}:hidden_eval_metadata_not_mapping")
        return issues

    for key in ("input_id", "claim", "allowed_tools"):
        if key not in model_visible:
            issues.append(f"{prefix}:model_visible_missing:{key}")
    for key in (
        "required_tools",
        "gold_evidence_status",
        "expected_terminal_action",
        "gold_source_ids",
        "requires_attribution",
        "requires_external_tool",
        "case_family",
        "split_group",
    ):
        if key not in hidden:
            issues.append(f"{prefix}:hidden_missing:{key}")

    visible_text = json.dumps(model_visible, sort_keys=True).lower()
    for source_id in hidden.get("gold_source_ids", ()) or ():
        if source_id and str(source_id).lower() in visible_text:
            issues.append(f"{prefix}:hidden_source_id_leak:{source_id}")
    for hidden_value in (
        hidden.get("gold_evidence_status"),
        hidden.get("expected_terminal_action"),
        hidden.get("case_family"),
        hidden.get("source_task_id"),
    ):
        if hidden_value and str(hidden_value).lower() in visible_text:
            issues.append(f"{prefix}:hidden_value_leak:{hidden_value}")

    input_id = str(model_visible.get("input_id", ""))
    for token in HIDDEN_LABEL_TOKENS:
        if token.lower() in input_id.lower():
            issues.append(f"{prefix}:input_id_leaks_label_token:{token}")

    try:
        task_spec_from_stage_a_row(row)
    except (KeyError, TypeError, ValueError) as exc:
        issues.append(f"{prefix}:task_spec_projection_failed:{exc}")
    return issues


def _query_filter_complete(
    trajectory: Trajectory,
    *,
    required_tools: Sequence[str],
    required_fields: Sequence[str],
) -> bool:
    if not required_tools or not required_fields:
        return True
    observed = {step.name: step for step in trajectory.steps}
    for tool in required_tools:
        step = observed.get(tool)
        if step is None:
            return False
        if not all(field in step.arguments for field in required_fields):
            return False
    return True


def _model_visible(row: Mapping[str, Any]) -> Mapping[str, Any]:
    model_visible = row.get("model_visible_task")
    if not isinstance(model_visible, Mapping):
        raise ValueError("Stage A row is missing model_visible_task mapping.")
    return model_visible


def _hidden_metadata(row: Mapping[str, Any]) -> Mapping[str, Any]:
    hidden = row.get("hidden_eval_metadata")
    if not isinstance(hidden, Mapping):
        raise ValueError("Stage A row is missing hidden_eval_metadata mapping.")
    return hidden


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default="negbiodb_ct/tasks_pilot.jsonl")
    parser.add_argument("--out", default="negbiodb_ct/stage_a_mini_manifest.jsonl")
    parser.add_argument("--per-action", type=int, default=5)
    parser.add_argument("--tool-profile", default="nullatlas_full")
    parser.add_argument("--min-rows", type=int, default=20)
    args = parser.parse_args()

    rows = build_stage_a_manifest_rows(
        load_task_records(args.tasks),
        per_action=args.per_action,
        tool_profile=args.tool_profile,
    )
    issues = validate_stage_a_manifest(rows, min_rows=args.min_rows)
    if issues:
        raise SystemExit("Stage A manifest validation failed:\n- " + "\n- ".join(issues))
    write_stage_a_manifest(args.out, rows)
    summary = {
        "out": args.out,
        "rows": len(rows),
        "status_counts": dict(Counter(row["hidden_eval_metadata"]["gold_evidence_status"] for row in rows)),
        "case_families": dict(Counter(row["hidden_eval_metadata"]["case_family"] for row in rows)),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

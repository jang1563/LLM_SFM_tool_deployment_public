from llm_sfm_tool_deployment import (
    Action,
    EvidenceStatus,
    TrajectoryEvaluator,
)
from negbiodb_ct import (
    CT_NATIVE_TOOL_LOOP,
    CT_NATIVE_TOOL_SEARCH,
    CT_REQUIRED_TOOL_LOOP,
    ideal_trajectory_from_record,
    prediction_trajectory_from_record,
    required_tools_for_action,
    task_spec_from_record,
)


def ground_record() -> dict:
    return {
        "packet_id": "ct::ground::27::1486",
        "action_class": "ground",
        "available_actions": ["ground", "reject", "defer", "verify", "flag"],
        "observation": {
            "claim": "Has Gemcitabine been tested and failed for Osteosarcoma?",
            "drug_id": 27,
            "condition_id": 1486,
        },
        "scoring_key": {
            "gold_action": "ground",
            "gold_nct": "NCT00673179",
            "gold_failure_category": "enrollment",
            "inject_impossible_value": False,
            "note": None,
        },
    }


def flag_record() -> dict:
    record = ground_record()
    record["packet_id"] = "ct::flag::433::25250"
    record["action_class"] = "flag"
    record["scoring_key"] = {
        "gold_action": "flag",
        "gold_nct": "NCT03396861",
        "gold_failure_category": "enrollment",
        "inject_impossible_value": True,
        "note": "adapter injects an impossible p-value into the retrieved record",
    }
    return record


def test_ground_record_maps_to_ground_with_attribution_and_passes() -> None:
    task = task_spec_from_record(ground_record())
    trajectory = ideal_trajectory_from_record(ground_record())

    result = TrajectoryEvaluator().evaluate(task, trajectory)

    assert task.expected_terminal_action == Action.GROUND_WITH_ATTRIBUTION
    assert task.gold_evidence_status == EvidenceStatus.SUPPORTED
    assert task.required_tools == CT_REQUIRED_TOOL_LOOP
    assert result.passed


def test_flag_record_maps_to_invalid_value_and_reject_or_flag() -> None:
    task = task_spec_from_record(flag_record())
    trajectory = ideal_trajectory_from_record(flag_record())

    result = TrajectoryEvaluator().evaluate(task, trajectory)

    assert task.expected_terminal_action == Action.REJECT_OR_FLAG_UNSUPPORTED_CLAIM
    assert task.gold_evidence_status == EvidenceStatus.INVALID_VALUE
    assert result.passed


def test_self_answer_prediction_fails_external_tool_loop() -> None:
    task = task_spec_from_record(ground_record())
    trajectory = prediction_trajectory_from_record(ground_record(), "self_answer")

    result = TrajectoryEvaluator().evaluate(task, trajectory)

    assert not result.passed
    assert "missing_required_tool_sequence" in result.violations
    assert "external_tool_required" in result.violations
    assert "terminal_action_mismatch" in result.violations


def test_partial_tool_prediction_still_requires_full_loop_and_attribution() -> None:
    task = task_spec_from_record(ground_record())
    trajectory = prediction_trajectory_from_record(
        ground_record(),
        "ground",
        tool_names=("nullatlas_survey_prior_failures",),
        cited_source_ids=(),
        predicted_evidence_status=EvidenceStatus.SUPPORTED,
    )

    result = TrajectoryEvaluator().evaluate(task, trajectory)

    assert not result.passed
    assert "missing_required_tool_sequence" in result.violations
    assert "missing_required_attribution" in result.violations


def test_native_profile_requires_minimal_action_specific_tools() -> None:
    assert required_tools_for_action("ground", tool_profile="native_ct") == (
        CT_NATIVE_TOOL_SEARCH,
    )
    assert required_tools_for_action("reject", tool_profile="native_ct") == (
        CT_NATIVE_TOOL_SEARCH,
    )
    assert required_tools_for_action("flag", tool_profile="native_ct") == (
        CT_NATIVE_TOOL_SEARCH,
    )
    assert required_tools_for_action("defer", tool_profile="native_ct") == CT_NATIVE_TOOL_LOOP
    assert required_tools_for_action("verify", tool_profile="native_ct") == CT_NATIVE_TOOL_LOOP


def test_native_profile_task_spec_uses_collapsed_runner_tools() -> None:
    task = task_spec_from_record(ground_record(), tool_profile="native_ct")

    assert task.required_tools == (CT_NATIVE_TOOL_SEARCH,)

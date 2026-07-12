import pytest

from llm_sfm_tool_deployment import EvidenceStatus, TrajectoryEvaluator
from negbiodb_ct import (
    CT_REQUIRED_TOOL_LOOP,
    ModelOutputParseError,
    parse_model_output_json,
    task_spec_from_record,
    trajectory_from_model_output,
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


def defer_record() -> dict:
    record = ground_record()
    record["packet_id"] = "ct::defer::74096::41728"
    record["action_class"] = "defer"
    record["scoring_key"] = {
        "gold_action": "defer",
        "gold_nct": None,
        "gold_failure_category": None,
        "inject_impossible_value": False,
        "note": "drug has no recorded failures -> insufficient evidence",
    }
    return record


def flag_record() -> dict:
    record = ground_record()
    record["packet_id"] = "ct::flag::383::5105"
    record["action_class"] = "flag"
    record["scoring_key"] = {
        "gold_action": "flag",
        "gold_nct": "NCT00844805",
        "gold_failure_category": "efficacy",
        "inject_impossible_value": True,
        "note": "adapter injects an impossible p-value into the gold efficacy record",
    }
    return record


def test_parse_model_output_json_tolerates_markdown_fence() -> None:
    parsed = parse_model_output_json(
        '```json\n{"action":"defer","evidence_status":"insufficient"}\n```'
    )

    assert parsed == {"action": "defer", "evidence_status": "insufficient"}


def test_ground_model_output_becomes_passing_trajectory() -> None:
    record = ground_record()
    output = {
        "action": "ground",
        "evidence_status": "supported",
        "tool_calls": list(CT_REQUIRED_TOOL_LOOP),
        "cited_source_ids": ["NCT00673179"],
        "rationale": "Recorded failure with attribution.",
    }

    trajectory = trajectory_from_model_output(record, output)
    result = TrajectoryEvaluator().evaluate(task_spec_from_record(record), trajectory)

    assert trajectory.predicted_evidence_status == EvidenceStatus.SUPPORTED
    assert trajectory.rationale == "Recorded failure with attribution."
    assert result.passed


def test_ground_output_missing_nct_fails_attribution() -> None:
    record = ground_record()
    output = {
        "action": "ground",
        "evidence_status": "supported",
        "tool_calls": list(CT_REQUIRED_TOOL_LOOP),
    }

    trajectory = trajectory_from_model_output(record, output)
    result = TrajectoryEvaluator().evaluate(task_spec_from_record(record), trajectory)

    assert not result.passed
    assert "missing_required_attribution" in result.violations


def test_defer_model_output_defaults_status_from_action() -> None:
    record = defer_record()
    output = {
        "action": "defer",
        "tool_calls": list(CT_REQUIRED_TOOL_LOOP),
    }

    trajectory = trajectory_from_model_output(record, output)
    result = TrajectoryEvaluator().evaluate(task_spec_from_record(record), trajectory)

    assert trajectory.predicted_evidence_status == EvidenceStatus.INSUFFICIENT
    assert result.passed


def test_simplified_runner_tool_names_are_mapped_but_still_partial() -> None:
    record = ground_record()
    output = {
        "action": "ground",
        "evidence_status": "supported",
        "called": ["search_failures"],
        "nct": "NCT00673179",
    }

    trajectory = trajectory_from_model_output(record, output)
    result = TrajectoryEvaluator().evaluate(task_spec_from_record(record), trajectory)

    assert [step.name for step in trajectory.steps] == [
        "nullatlas_survey_prior_failures",
        "nullatlas_verify_trial_claims",
    ]
    assert not result.passed
    assert "missing_required_tool_sequence" in result.violations


def test_native_profile_runner_tool_names_pass_ground_trace() -> None:
    record = ground_record()
    output = {
        "action": "ground",
        "evidence_status": "supported",
        "called": ["search_failures"],
        "nct": "NCT00673179",
    }

    trajectory = trajectory_from_model_output(record, output, tool_profile="native_ct")
    result = TrajectoryEvaluator().evaluate(
        task_spec_from_record(record, tool_profile="native_ct"),
        trajectory,
    )

    assert [step.name for step in trajectory.steps] == ["search_failures"]
    assert result.passed


def test_native_profile_defer_requires_other_indications_tool() -> None:
    record = defer_record()
    output = {
        "action": "defer",
        "called": ["search_failures"],
    }

    trajectory = trajectory_from_model_output(record, output, tool_profile="native_ct")
    result = TrajectoryEvaluator().evaluate(
        task_spec_from_record(record, tool_profile="native_ct"),
        trajectory,
    )

    assert not result.passed
    assert "missing_required_tool_sequence" in result.violations


def test_native_profile_defer_passes_with_absence_and_other_check() -> None:
    record = defer_record()
    output = {
        "action": "defer",
        "called": ["search_failures", "check_other_indications"],
    }

    trajectory = trajectory_from_model_output(record, output, tool_profile="native_ct")
    result = TrajectoryEvaluator().evaluate(
        task_spec_from_record(record, tool_profile="native_ct"),
        trajectory,
    )

    assert result.passed


def test_native_profile_flag_requires_invalid_record_attribution() -> None:
    record = flag_record()
    output = {
        "action": "flag",
        "called": ["search_failures"],
    }

    trajectory = trajectory_from_model_output(record, output, tool_profile="native_ct")
    result = TrajectoryEvaluator().evaluate(
        task_spec_from_record(record, tool_profile="native_ct"),
        trajectory,
    )

    assert not result.passed
    assert "missing_required_attribution" in result.violations


def test_native_profile_flag_passes_with_invalid_record_nct() -> None:
    record = flag_record()
    output = {
        "action": "flag",
        "called": ["search_failures"],
        "nct": "NCT00844805",
    }

    trajectory = trajectory_from_model_output(record, output, tool_profile="native_ct")
    result = TrajectoryEvaluator().evaluate(
        task_spec_from_record(record, tool_profile="native_ct"),
        trajectory,
    )

    assert result.passed


def test_invalid_json_raises_parse_error() -> None:
    with pytest.raises(ModelOutputParseError):
        parse_model_output_json("not json")


def test_unknown_action_raises_parse_error() -> None:
    with pytest.raises(ModelOutputParseError):
        trajectory_from_model_output(ground_record(), {"action": "maybe"})

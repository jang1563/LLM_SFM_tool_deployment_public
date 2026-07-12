from negbiodb_ct.export_post_training_data import (
    preference_pairs,
    supervised_example,
)


def base_record(action_class: str, gold_nct: str | None = None) -> dict:
    return {
        "packet_id": f"ct::{action_class}::1::2",
        "action_class": action_class,
        "available_actions": ["ground", "reject", "defer", "verify", "flag"],
        "observation": {
            "claim": "Has Drug X been tested and failed for Condition Y?",
            "drug_id": 1,
            "condition_id": 2,
        },
        "scoring_key": {
            "gold_action": action_class,
            "gold_nct": gold_nct,
            "gold_failure_category": "efficacy" if gold_nct else None,
            "inject_impossible_value": action_class == "flag",
            "note": None,
        },
    }


def tool_results(nct: str = "NCT12345678") -> dict:
    return {
        "search_failures": [
            {
                "nct": nct,
                "endpoint_met": 0,
                "p_value": 0.42,
                "year": 2024,
                "failure_category": "efficacy",
            }
        ],
        "check_other_indications": {"failures_for_other_indications": 3},
    }


def runner_row(action: str, nct: str | None = None) -> dict:
    model_output = {"action": action, "called": ["search_failures", "check_other_indications"]}
    if nct:
        model_output["cited_source_ids"] = [nct]
    return {
        "packet_id": f"ct::{action}::1::2",
        "class": action,
        "gold": action,
        "pred": {"action": action, "cited_nct": nct},
        "called": ["search_failures", "check_other_indications"],
        "model_output": model_output,
        "correct": True,
        "reward": 1.0,
        "generic_score": 1.0,
        "generic_violations": [],
    }


def test_supervised_example_preserves_flag_nct_in_final_tool_call() -> None:
    record = base_record("flag", gold_nct="NCT12345678")
    row = runner_row("flag", nct="NCT12345678")

    example = supervised_example(
        record,
        row,
        tool_results(),
        source_runner="claude-sonnet-4-6",
    )

    final_message = example["messages"][-1]
    assert final_message["tool_call"] == {
        "name": "submit_decision",
        "arguments": {"action": "flag", "nct": "NCT12345678"},
    }


def test_preference_pairs_score_missing_attribution_and_self_answer() -> None:
    record = base_record("ground", gold_nct="NCT12345678")
    row = runner_row("ground", nct="NCT12345678")

    pairs = preference_pairs(
        record,
        row,
        tool_results(),
        source_runner="claude-sonnet-4-6",
    )

    by_mode = {pair["failure_mode"]: pair for pair in pairs}
    assert by_mode["missing_attribution"]["chosen_score"]["passed"]
    assert not by_mode["missing_attribution"]["rejected_score"]["passed"]
    assert "missing_required_attribution" in by_mode["missing_attribution"]["rejected_score"]["violations"]
    assert "self_answering_without_tools" in by_mode
    assert "external_tool_required" in by_mode["self_answering_without_tools"]["rejected_score"]["violations"]
    assert by_mode["self_answering_without_tools"]["rejected_messages"] == [
        {"role": "assistant", "content": '{"action": "self_answer"}'}
    ]


def test_preference_pairs_capture_mixed_endpoint_over_grounding() -> None:
    record = base_record("reject")
    row = runner_row("reject")
    observations = {
        "search_failures": [
            {
                "nct": "NCT00000001",
                "endpoint_met": 1,
                "p_value": 0.001,
                "year": None,
                "failure_category": "strategic",
            },
            {
                "nct": "NCT00000001",
                "endpoint_met": 0,
                "p_value": 0.74,
                "year": None,
                "failure_category": "efficacy",
            },
        ],
        "check_other_indications": {"failures_for_other_indications": 10},
    }

    pairs = preference_pairs(
        record,
        row,
        observations,
        source_runner="claude-sonnet-4-6",
    )

    pair = next(pair for pair in pairs if pair["failure_mode"] == "mixed_endpoint_over_grounding")
    assert pair["chosen"] == '{"action": "reject"}'
    assert pair["rejected"] == '{"action": "ground", "nct": "NCT00000001"}'
    assert "contradicted_claim_requires_reject_or_flag" in pair["rejected_score"]["violations"]

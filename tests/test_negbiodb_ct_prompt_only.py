import json

from examples.analyze_negbiodb_ct_prompt_only import classify_prompt_only_failure
from negbiodb_ct.run_prompt_only import (
    load_packet_ids,
    parse_prompt_only_decision,
    select_tasks,
)


def test_parse_prompt_only_decision_extracts_nct() -> None:
    decision = parse_prompt_only_decision('{"action":"ground","nct":"NCT01234567"}')

    assert decision == {"action": "ground", "cited_nct": "NCT01234567"}


def test_parse_prompt_only_decision_ignores_nct_for_non_citing_actions() -> None:
    decision = parse_prompt_only_decision('{"action":"verify","nct":"NCT01234567"}')

    assert decision == {"action": "verify", "cited_nct": None}


def test_parse_prompt_only_decision_rejects_unknown_action() -> None:
    decision = parse_prompt_only_decision('{"action":"trust_me"}')

    assert decision == {"action": None, "cited_nct": None}


def test_load_packet_ids_accepts_post_training_task_id(tmp_path) -> None:
    path = tmp_path / "sft.jsonl"
    path.write_text(json.dumps({"task_id": "ct::ground::1::2"}) + "\n")

    assert load_packet_ids(path) == ["ct::ground::1::2"]


def test_select_tasks_preserves_packet_id_order() -> None:
    tasks = [
        {"packet_id": "a", "action_class": "ground"},
        {"packet_id": "b", "action_class": "defer"},
    ]

    assert [task["packet_id"] for task in select_tasks(tasks, n=2, packet_ids=["b", "a"])] == ["b", "a"]


def test_prompt_only_failure_classifier_names_conservative_defer() -> None:
    row = {"gold": "ground", "pred": {"action": "defer"}, "correct": False}

    assert classify_prompt_only_failure(
        row,
        ("missing_required_tool_sequence", "terminal_action_mismatch"),
    ) == "conservative_defer_wrong_without_tools"


def test_prompt_only_failure_classifier_names_correct_action_missing_tools() -> None:
    row = {"gold": "defer", "pred": {"action": "defer"}, "correct": True}

    assert classify_prompt_only_failure(row, ("missing_required_tool_sequence",)) == (
        "correct_action_but_no_tool_trace"
    )


def test_prompt_only_failure_classifier_names_oververify() -> None:
    row = {"gold": "defer", "pred": {"action": "verify"}, "correct": False}

    assert classify_prompt_only_failure(
        row,
        ("missing_required_tool_sequence", "terminal_action_mismatch"),
    ) == "oververify_without_evidence"

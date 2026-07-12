from post_training.run_sft_decision_eval import (
    parse_final_decision,
    tool_calls_from_example,
)


def test_parse_final_decision_normalizes_ground_citation() -> None:
    decision = parse_final_decision('```json\n{"action":"ground","nct":"NCT01234567"}\n```')

    assert decision == {"action": "ground", "cited_nct": "NCT01234567"}


def test_parse_final_decision_rejects_bad_action() -> None:
    decision = parse_final_decision('{"action":"self_answer"}')

    assert decision == {"action": None, "cited_nct": None}


def test_tool_calls_from_example_excludes_final_submit_decision() -> None:
    example = {
        "messages": [
            {"role": "assistant", "tool_call": {"name": "search_failures", "arguments": {}}},
            {"role": "tool", "name": "search_failures", "content": []},
            {"role": "assistant", "tool_call": {"name": "check_other_indications", "arguments": {}}},
            {"role": "assistant", "tool_call": {"name": "submit_decision", "arguments": {"action": "defer"}}},
        ]
    }

    assert tool_calls_from_example(example) == ["search_failures", "check_other_indications"]

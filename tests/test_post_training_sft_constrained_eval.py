from post_training.run_sft_constrained_eval import (
    candidate_decisions,
    candidate_to_decision,
    choose_scored_candidate,
    returned_ncts,
    target_json,
)


def example_with_duplicate_ncts() -> dict:
    return {
        "messages": [
            {"role": "tool", "name": "search_failures", "content": [
                {"nct": "NCT00000001"},
                {"nct": "NCT00000001"},
                {"nct": "NCT00000002"},
            ]},
            {"role": "tool", "name": "check_other_indications", "content": {"failures_for_other_indications": 1}},
            {"role": "assistant", "tool_call": {"name": "submit_decision", "arguments": {"action": "reject"}}},
        ]
    }


def test_returned_ncts_dedupes_in_order() -> None:
    assert returned_ncts(example_with_duplicate_ncts()) == ["NCT00000001", "NCT00000002"]


def test_candidate_decisions_include_base_actions_and_returned_nct_actions() -> None:
    assert candidate_decisions(example_with_duplicate_ncts()) == [
        {"action": "defer"},
        {"action": "verify"},
        {"action": "reject"},
        {"action": "ground", "nct": "NCT00000001"},
        {"action": "flag", "nct": "NCT00000001"},
        {"action": "ground", "nct": "NCT00000002"},
        {"action": "flag", "nct": "NCT00000002"},
    ]


def test_candidate_to_decision_maps_citation_only_for_ground_or_flag() -> None:
    assert candidate_to_decision({"action": "ground", "nct": "NCT00000001"}) == {
        "action": "ground",
        "cited_nct": "NCT00000001",
    }
    assert candidate_to_decision({"action": "reject", "nct": "NCT00000001"}) == {
        "action": "reject",
        "cited_nct": None,
    }


def test_choose_scored_candidate_supports_mean_and_sum_modes() -> None:
    scores = [
        {"candidate": {"action": "defer"}, "mean_nll": 1.0, "sum_nll": 10.0},
        {"candidate": {"action": "reject"}, "mean_nll": 1.1, "sum_nll": 2.0},
    ]

    assert choose_scored_candidate(scores, score_mode="mean")["candidate"] == {"action": "defer"}
    assert choose_scored_candidate(scores, score_mode="sum")["candidate"] == {"action": "reject"}


def test_target_json_is_sorted_and_eos_terminated() -> None:
    assert target_json({"nct": "NCT00000001", "action": "ground"}, "<eos>") == (
        '{"action": "ground", "nct": "NCT00000001"}<eos>'
    )

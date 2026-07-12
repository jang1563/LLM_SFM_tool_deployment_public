from post_training.run_boundary_preference_candidate_eval import (
    candidate_actions_match,
    candidates_for_pair,
    candidates_match,
    expected_rank,
    summarize_group,
)


def pair() -> dict:
    return {
        "id": "pref-1",
        "task_id": "packet-1",
        "failure_mode": "boundary_flag_over_ground",
        "evidence_derived_action": "flag",
        "rejected_action": "ground",
        "prompt_messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "claim"},
            {"role": "assistant", "tool_call": {"name": "search_failures", "arguments": {}}},
            {
                "role": "tool",
                "name": "search_failures",
                "content": [{"nct": "NCT00000001"}, {"nct": "NCT00000001"}],
            },
            {"role": "assistant", "tool_call": {"name": "check_other_indications", "arguments": {}}},
            {"role": "tool", "name": "check_other_indications", "content": {"failures_for_other_indications": 1}},
        ],
        "chosen_messages": [
            {
                "role": "assistant",
                "tool_call": {
                    "name": "submit_decision",
                    "arguments": {"action": "flag", "nct": "NCT00000001"},
                },
            }
        ],
        "rejected_messages": [
            {
                "role": "assistant",
                "tool_call": {
                    "name": "submit_decision",
                    "arguments": {"action": "ground", "nct": "NCT00000001"},
                },
            }
        ],
    }


def test_candidates_for_pair_includes_expected_from_visible_ncts() -> None:
    candidates, expected_in_candidates = candidates_for_pair(pair())

    assert expected_in_candidates is True
    assert {"action": "flag", "nct": "NCT00000001"} in candidates
    assert candidates.count({"action": "flag", "nct": "NCT00000001"}) == 1


def test_candidates_match_requires_exact_nct_but_action_match_does_not() -> None:
    expected = {"action": "flag", "nct": "NCT00000001"}
    same_action = {"action": "flag", "nct": "NCT00000002"}

    assert not candidates_match(same_action, expected)
    assert candidate_actions_match(same_action, expected)


def test_expected_rank_reports_margin_from_winner() -> None:
    scores = [
        {"candidate": {"action": "verify"}, "mean_nll": 1.0, "sum_nll": 10.0},
        {"candidate": {"action": "flag", "nct": "NCT00000001"}, "mean_nll": 1.5, "sum_nll": 8.0},
    ]

    assert expected_rank(
        scores,
        {"action": "flag", "nct": "NCT00000001"},
        score_mode="mean",
    ) == {
        "rank": 2,
        "score": 1.5,
        "margin_from_winner": 0.5,
    }
    assert expected_rank(
        scores,
        {"action": "flag", "nct": "NCT00000001"},
        score_mode="sum",
    ) == {
        "rank": 1,
        "score": 8.0,
        "margin_from_winner": 0.0,
    }


def test_summarize_group_reports_accuracy_rank_and_actions() -> None:
    rows = [
        {
            "failure_mode": "boundary_flag_over_ground",
            "action_correct": True,
            "exact_candidate_correct": True,
            "expected_rank": 1,
            "expected_margin_from_winner": 0.0,
            "pred": {"action": "flag", "nct": "NCT00000001"},
        },
        {
            "failure_mode": "boundary_flag_over_ground",
            "action_correct": False,
            "exact_candidate_correct": False,
            "expected_rank": 3,
            "expected_margin_from_winner": 0.25,
            "pred": {"action": "ground", "nct": "NCT00000001"},
        },
    ]

    summary = summarize_group(rows, key="failure_mode")

    assert summary["boundary_flag_over_ground"]["action_accuracy"] == 0.5
    assert summary["boundary_flag_over_ground"]["exact_candidate_accuracy"] == 0.5
    assert summary["boundary_flag_over_ground"]["expected_rank_counts"] == {1: 1, 3: 1}
    assert summary["boundary_flag_over_ground"]["pred_actions"] == {"flag": 1, "ground": 1}

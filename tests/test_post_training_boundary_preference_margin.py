from post_training.run_boundary_preference_margin import (
    final_candidate,
    margin_stats,
    prompt_from_pair,
    summarize_group,
)


def pair() -> dict:
    return {
        "id": "pref-1",
        "task_id": "packet-1",
        "failure_mode": "boundary_defer_over_verify",
        "evidence_derived_action": "defer",
        "rejected_action": "verify",
        "prompt_messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "claim"},
            {"role": "assistant", "tool_call": {"name": "search_failures", "arguments": {}}},
            {"role": "tool", "name": "search_failures", "content": []},
        ],
        "chosen_messages": [
            {"role": "assistant", "tool_call": {"name": "submit_decision", "arguments": {"action": "defer"}}}
        ],
        "rejected_messages": [
            {"role": "assistant", "tool_call": {"name": "submit_decision", "arguments": {"action": "verify"}}}
        ],
    }


def test_final_candidate_extracts_action_and_nct() -> None:
    assert final_candidate([
        {"role": "assistant", "tool_call": {"name": "submit_decision", "arguments": {"action": "defer"}}}
    ]) == {"action": "defer"}
    assert final_candidate([
        {
            "role": "assistant",
            "tool_call": {
                "name": "submit_decision",
                "arguments": {"action": "ground", "nct": "NCT00000001"},
            },
        }
    ]) == {"action": "ground", "nct": "NCT00000001"}


def test_prompt_from_pair_uses_prompt_messages_only() -> None:
    prompt = prompt_from_pair(pair())

    assert "SYSTEM: system" in prompt
    assert "TOOL_RESULT search_failures: []" in prompt
    assert '{"action": "defer"}' not in prompt
    assert prompt.endswith("FINAL_SUBMIT_DECISION_JSON:")


def test_margin_stats_handles_empty_and_values() -> None:
    assert margin_stats([]) == {"mean": None, "median": None, "min": None, "max": None}
    assert margin_stats([1.0, 2.0, 4.0]) == {
        "mean": 2.3333,
        "median": 2.0,
        "min": 1.0,
        "max": 4.0,
    }


def test_summarize_group_reports_win_rate_and_margin() -> None:
    rows = [
        {"failure_mode": "a", "mean_margin": 1.0, "mean_chosen_wins": True},
        {"failure_mode": "a", "mean_margin": -1.0, "mean_chosen_wins": False},
        {"failure_mode": "b", "mean_margin": 2.0, "mean_chosen_wins": True},
    ]

    summary = summarize_group(rows, key="failure_mode")

    assert summary["a"]["n"] == 2
    assert summary["a"]["mean_win_rate"] == 0.5
    assert summary["a"]["mean_margin"]["mean"] == 0.0
    assert summary["b"]["mean_win_rate"] == 1.0

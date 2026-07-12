from post_training.build_boundary_preference_data import (
    boundary_preference_pairs_for_row,
    build_boundary_preference_pairs,
)


def record(action: str) -> dict:
    return {
        "packet_id": f"ct::{action}::1::2",
        "action_class": action,
        "available_actions": ["ground", "reject", "defer", "verify", "flag"],
        "observation": {
            "claim": "Has Drug X been tested and failed for Condition Y?",
            "drug_id": 1,
            "condition_id": 2,
        },
        "scoring_key": {
            "gold_action": action,
            "gold_nct": "NCT00000001" if action in {"ground", "flag"} else None,
            "gold_failure_category": "efficacy" if action == "ground" else None,
            "inject_impossible_value": action == "flag",
            "note": None,
        },
    }


def sft_row(action: str, search_failures: list[dict], other_failures: int) -> dict:
    return {
        "id": f"sft::ct::{action}::1::2",
        "task_id": f"ct::{action}::1::2",
        "dataset": "source",
        "tool_profile": "native_ct",
        "action_class": action,
        "messages": [
            {"role": "system", "content": "Use tools."},
            {"role": "user", "content": "Has Drug X been tested and failed for Condition Y?"},
            {"role": "assistant", "tool_call": {"name": "search_failures", "arguments": {}}},
            {"role": "tool", "name": "search_failures", "content": search_failures},
            {"role": "assistant", "tool_call": {"name": "check_other_indications", "arguments": {}}},
            {
                "role": "tool",
                "name": "check_other_indications",
                "content": {"failures_for_other_indications": other_failures},
            },
            {"role": "assistant", "tool_call": {"name": "submit_decision", "arguments": {"action": action}}},
        ],
        "metadata": {"gold_action": action},
    }


def test_boundary_preference_pair_targets_defer_over_verify() -> None:
    row = sft_row("defer", [], 0)
    pairs = boundary_preference_pairs_for_row(
        row,
        record("defer"),
        dataset="boundary_pref",
        strategy="test",
        pair_index=0,
    )

    assert len(pairs) == 1
    pair = pairs[0]
    assert pair["failure_mode"] == "boundary_defer_over_verify"
    assert pair["chosen"] == '{"action": "defer"}'
    assert pair["rejected"] == '{"action": "verify"}'
    assert pair["chosen_score"]["passed"]
    assert not pair["rejected_score"]["passed"]
    assert "terminal_action_mismatch" in pair["rejected_score"]["violations"]
    assert pair["prompt_messages"][-1]["name"] == "check_other_indications"
    assert pair["chosen_messages"][0]["tool_call"]["name"] == "submit_decision"


def test_boundary_preference_pair_targets_reject_over_ground_and_flag() -> None:
    row = sft_row(
        "reject",
        [
            {"nct": "NCT00000001", "endpoint_met": 1, "p_value": 0.01, "failure_category": "strategic"},
            {"nct": "NCT00000001", "endpoint_met": 0, "p_value": 0.42, "failure_category": "efficacy"},
        ],
        3,
    )
    pairs = boundary_preference_pairs_for_row(
        row,
        record("reject"),
        dataset="boundary_pref",
        strategy="test",
        pair_index=0,
    )

    by_mode = {pair["failure_mode"]: pair for pair in pairs}
    assert sorted(by_mode) == ["boundary_reject_over_flag", "boundary_reject_over_ground"]
    assert by_mode["boundary_reject_over_ground"]["rejected"] == '{"action": "ground", "nct": "NCT00000001"}'
    assert not by_mode["boundary_reject_over_ground"]["rejected_score"]["passed"]
    assert "contradicted_claim_requires_reject_or_flag" in by_mode["boundary_reject_over_ground"]["rejected_score"]["violations"]
    assert not by_mode["boundary_reject_over_flag"]["rejected_score"]["passed"]


def test_build_boundary_preference_pairs_counts_all_negatives() -> None:
    rows = [
        sft_row("defer", [], 0),
        sft_row(
            "ground",
            [{"nct": "NCT00000001", "endpoint_met": 0, "p_value": 0.42, "failure_category": "efficacy"}],
            0,
        ),
    ]
    tasks = {
        "ct::defer::1::2": record("defer"),
        "ct::ground::1::2": record("ground"),
    }

    pairs = build_boundary_preference_pairs(rows, tasks, dataset="boundary_pref")

    assert [pair["failure_mode"] for pair in pairs] == [
        "boundary_defer_over_verify",
        "boundary_ground_over_flag",
        "boundary_ground_over_reject",
    ]

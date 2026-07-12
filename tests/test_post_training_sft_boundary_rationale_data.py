from post_training.build_sft_boundary_rationale_data import (
    BOUNDARY_NEGATIVES,
    build_boundary_rows,
    rationale_copy,
)


def example(action: str = "reject") -> dict:
    return {
        "id": "row-1",
        "task_id": "packet-1",
        "dataset": "source",
        "action_class": action,
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "claim"},
            {"role": "assistant", "tool_call": {"name": "search_failures", "arguments": {}}},
            {"role": "tool", "name": "search_failures", "content": []},
            {"role": "assistant", "tool_call": {"name": "submit_decision", "arguments": {"action": action}}},
        ],
    }


def test_rationale_copy_inserts_boundary_message_before_final_decision() -> None:
    row = rationale_copy(example("reject"), dataset="dataset", pair_index=0)

    assert row["dataset"] == "dataset"
    assert row["boundary_pair_role"] == "rationale"
    assert row["boundary_negative_actions"] == BOUNDARY_NEGATIVES["reject"]
    assert row["messages"][-2]["role"] == "user"
    assert "BOUNDARY_RATIONALE" in row["messages"][-2]["content"]
    assert row["messages"][-1]["tool_call"]["name"] == "submit_decision"


def test_build_boundary_rows_pairs_base_and_rationale_rows() -> None:
    rows = build_boundary_rows([example("defer"), example("flag")], dataset="dataset")

    assert [row["boundary_pair_role"] for row in rows] == ["base", "rationale", "base", "rationale"]
    assert [row["action_class"] for row in rows] == ["defer", "defer", "flag", "flag"]
    assert all(row["dataset"] == "dataset" for row in rows)

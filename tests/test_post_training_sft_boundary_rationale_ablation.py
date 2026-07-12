from post_training.apply_evidence_rationale import build_evidence_rationale_rows, manifest_for_rows
from post_training.evidence_rationale import evidence_action
from post_training.run_sft_boundary_rationale_ablation import build_heldout_rationale_rows


def example(action: str = "defer") -> dict:
    return {
        "id": "sft::packet-1",
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


def example_with_tool_content(
    *,
    action: str,
    search_failures: list[dict],
    other_failures: int = 0,
) -> dict:
    row = example(action)
    row["messages"].insert(
        -1,
        {
            "role": "tool",
            "name": "check_other_indications",
            "content": {"failures_for_other_indications": other_failures},
        },
    )
    row["messages"][3]["content"] = search_failures
    return row


def test_build_heldout_rationale_rows_inserts_oracle_rationale() -> None:
    rows = build_heldout_rationale_rows(
        [example("defer"), example("verify")],
        dataset="ablation",
        rationale_mode="oracle",
    )

    assert [row["boundary_pair_role"] for row in rows] == ["rationale", "rationale"]
    assert [row["dataset"] for row in rows] == ["ablation", "ablation"]
    assert rows[0]["messages"][-2]["role"] == "user"
    assert "BOUNDARY_RATIONALE" in rows[0]["messages"][-2]["content"]
    assert "Correct final action: defer." in rows[0]["messages"][-2]["content"]
    assert rows[0]["messages"][-1]["tool_call"]["name"] == "submit_decision"


def test_evidence_action_uses_empty_evidence_for_defer() -> None:
    assert evidence_action(example("defer")) == "defer"


def test_evidence_action_uses_related_failures_for_verify() -> None:
    row = example_with_tool_content(action="verify", search_failures=[], other_failures=2)

    assert evidence_action(row) == "verify"


def test_evidence_action_uses_clean_efficacy_failure_for_ground() -> None:
    row = example_with_tool_content(
        action="ground",
        search_failures=[
            {
                "nct": "NCT123",
                "failure_category": "efficacy",
                "endpoint_met": 0,
                "p_value": 0.41,
            }
        ],
    )

    assert evidence_action(row) == "ground"


def test_evidence_action_uses_impossible_p_value_for_flag() -> None:
    row = example_with_tool_content(
        action="flag",
        search_failures=[
            {
                "nct": "NCT123",
                "failure_category": "efficacy",
                "endpoint_met": 0,
                "p_value": 14.73,
            }
        ],
    )

    assert evidence_action(row) == "flag"


def test_evidence_action_uses_mixed_endpoint_override_for_reject() -> None:
    row = example_with_tool_content(
        action="reject",
        search_failures=[
            {"failure_category": "efficacy", "endpoint_met": 0, "p_value": 0.5},
            {"failure_category": "efficacy", "endpoint_met": 1, "p_value": 0.01},
        ],
    )

    assert evidence_action(row) == "reject"


def test_build_heldout_rationale_rows_can_use_evidence_mode() -> None:
    rows = build_heldout_rationale_rows([example("defer")], dataset="ablation", rationale_mode="evidence")

    assert rows[0]["boundary_pair_role"] == "evidence_rationale"
    assert rows[0]["boundary_strategy"] == "heldout_evidence_boundary_rationale_ablation_v1"
    assert rows[0]["evidence_derived_action"] == "defer"
    assert rows[0]["evidence_matches_action_class"] is True
    assert "Correct final action: defer." in rows[0]["messages"][-2]["content"]


def test_build_evidence_rationale_rows_uses_deployable_strategy() -> None:
    rows = build_evidence_rationale_rows([example("defer")], dataset="deployable")

    assert rows[0]["boundary_strategy"] == "deployable_evidence_boundary_rationale_v1"
    assert rows[0]["evidence_derived_action"] == "defer"
    assert "Evidence-derived final action: defer." in rows[0]["messages"][-2]["content"]


def test_manifest_for_rows_counts_evidence_matches() -> None:
    rows = build_evidence_rationale_rows(
        [
            example("defer"),
            example_with_tool_content(action="verify", search_failures=[], other_failures=2),
        ],
        dataset="deployable",
    )
    manifest = manifest_for_rows(
        source="source.jsonl",
        out="out.jsonl",
        dataset="deployable",
        strategy="deployable_evidence_boundary_rationale_v1",
        rows=rows,
    )

    assert manifest["examples"] == 2
    assert manifest["by_evidence_action"] == {"defer": 1, "verify": 1}
    assert manifest["by_role"] == {"evidence_rationale": 2}
    assert manifest["evidence_action_matches"] == 2
    assert manifest["evidence_action_mismatches"] == 0
    assert manifest["evidence_action_unlabeled"] == 0

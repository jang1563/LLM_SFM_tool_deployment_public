from post_training.build_boundary_preference_hard_modes import (
    DEFAULT_HARD_MODES,
    manifest_for_hard_pairs,
    parse_modes,
    select_hard_pairs,
)


def pair(mode: str, index: int) -> dict:
    chosen, rejected = mode.removeprefix("boundary_").split("_over_")
    return {
        "id": f"pref-{index}",
        "dataset": "full",
        "task_id": f"ct::{chosen}::{index}",
        "tool_profile": "native_ct",
        "failure_mode": mode,
        "evidence_derived_action": chosen,
        "rejected_action": rejected,
        "chosen_score": {"passed": True},
        "rejected_score": {"passed": False},
    }


def test_parse_modes_defaults_and_custom_list() -> None:
    assert parse_modes(None) == DEFAULT_HARD_MODES
    assert parse_modes("a,b") == ("a", "b")


def test_select_hard_pairs_rewrites_dataset_and_preserves_source_id() -> None:
    rows = [
        pair("boundary_defer_over_verify", 0),
        pair("boundary_ground_over_reject", 1),
        pair("boundary_reject_over_flag", 2),
    ]

    selected = select_hard_pairs(rows, modes=("boundary_defer_over_verify", "boundary_reject_over_flag"))

    assert [row["failure_mode"] for row in selected] == [
        "boundary_defer_over_verify",
        "boundary_reject_over_flag",
    ]
    assert selected[0]["dataset"] == "negbiodb_ct_oracle_boundary_preferences_hard_v1"
    assert selected[0]["source_preference_id"] == "pref-0"
    assert selected[0]["source_dataset"] == "full"
    assert selected[0]["id"].startswith("prefhard::ct::defer::0::boundary_defer_over_verify::")


def test_manifest_for_hard_pairs_counts_selected_modes() -> None:
    rows = [
        pair("boundary_defer_over_verify", 0),
        pair("boundary_reject_over_ground", 1),
    ]
    selected = select_hard_pairs(rows, modes=("boundary_defer_over_verify",))
    manifest = manifest_for_hard_pairs(
        source="full.jsonl",
        out="hard.jsonl",
        dataset="hard",
        strategy="strategy",
        modes=("boundary_defer_over_verify",),
        rows=rows,
        selected=selected,
        selection_source="summary.json",
    )

    assert manifest["source_preference_pairs"] == 2
    assert manifest["preference_pairs"] == 1
    assert manifest["pairs_by_failure_mode"] == {"boundary_defer_over_verify": 1}
    assert manifest["pairs_by_chosen_action"] == {"defer": 1}
    assert manifest["pairs_by_rejected_action"] == {"verify": 1}
    assert manifest["chosen_passed"] == 1
    assert manifest["rejected_passed"] == 0
    assert manifest["missing_selected_failure_modes"] == []

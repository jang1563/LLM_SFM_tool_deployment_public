from post_training.split_boundary_preference_hard_modes import (
    manifest_for_split,
    split_by_failure_mode,
)


def row(mode: str, index: int) -> dict:
    chosen, rejected = mode.removeprefix("boundary_").split("_over_")
    return {
        "id": f"prefhard::{mode}::{index}",
        "dataset": "hard",
        "task_id": f"ct::{chosen}::{index}",
        "failure_mode": mode,
        "evidence_derived_action": chosen,
        "rejected_action": rejected,
        "tool_profile": "native_ct",
    }


def test_split_by_failure_mode_is_stratified_and_rewrites_dataset() -> None:
    rows = [
        *(row("boundary_defer_over_verify", index) for index in range(4)),
        *(row("boundary_reject_over_ground", index) for index in range(4, 8)),
    ]

    train, heldout = split_by_failure_mode(rows, heldout_per_mode=1, seed=7)

    assert len(train) == 6
    assert len(heldout) == 2
    assert {item["failure_mode"] for item in heldout} == {
        "boundary_defer_over_verify",
        "boundary_reject_over_ground",
    }
    assert {item["dataset"] for item in train} == {
        "negbiodb_ct_oracle_boundary_preferences_hard_train_v1"
    }
    assert {item["dataset"] for item in heldout} == {
        "negbiodb_ct_oracle_boundary_preferences_hard_heldout_v1"
    }
    assert all(item["id"].endswith("::train") for item in train)
    assert all(item["id"].endswith("::heldout") for item in heldout)


def test_manifest_for_split_reports_counts_and_no_overlap() -> None:
    rows = [row("boundary_defer_over_verify", index) for index in range(4)]
    train, heldout = split_by_failure_mode(rows, heldout_per_mode=1, seed=7)
    manifest = manifest_for_split(
        source="hard.jsonl",
        train_out="train.jsonl",
        heldout_out="heldout.jsonl",
        seed=7,
        heldout_per_mode=1,
        train=train,
        heldout=heldout,
        train_dataset="train_dataset",
        heldout_dataset="heldout_dataset",
    )

    assert manifest["train_pairs"] == 3
    assert manifest["heldout_pairs"] == 1
    assert manifest["train_by_failure_mode"] == {"boundary_defer_over_verify": 3}
    assert manifest["heldout_by_failure_mode"] == {"boundary_defer_over_verify": 1}
    assert manifest["overlap_source_ids"] == []

from collections import Counter

import pytest

from post_training.split_sft_data import manifest_for_split, split_rows


def rows() -> list[dict]:
    out = []
    for action_class in ["defer", "ground", "flag"]:
        for i in range(4):
            out.append({
                "id": f"sft::ct::{action_class}::{i}",
                "task_id": f"ct::{action_class}::{i}",
                "action_class": action_class,
            })
    return out


def test_split_rows_is_stratified_and_disjoint() -> None:
    train, heldout = split_rows(rows(), heldout_per_class=1, seed=123)

    assert len(train) == 9
    assert len(heldout) == 3
    assert Counter(row["action_class"] for row in train) == {"defer": 3, "flag": 3, "ground": 3}
    assert Counter(row["action_class"] for row in heldout) == {"defer": 1, "flag": 1, "ground": 1}
    assert {row["id"] for row in train}.isdisjoint({row["id"] for row in heldout})


def test_split_rows_is_deterministic_for_same_seed() -> None:
    train_a, heldout_a = split_rows(rows(), heldout_per_class=1, seed=123)
    train_b, heldout_b = split_rows(rows(), heldout_per_class=1, seed=123)

    assert [row["id"] for row in train_a] == [row["id"] for row in train_b]
    assert [row["id"] for row in heldout_a] == [row["id"] for row in heldout_b]


def test_split_rows_rejects_too_large_heldout() -> None:
    with pytest.raises(ValueError, match="cannot hold out"):
        split_rows(rows(), heldout_per_class=4, seed=123)


def test_manifest_for_split_records_counts_and_ids() -> None:
    train, heldout = split_rows(rows(), heldout_per_class=1, seed=123)

    manifest = manifest_for_split(
        "source.jsonl",
        "train.jsonl",
        "heldout.jsonl",
        train,
        heldout,
        seed=123,
        heldout_per_class=1,
    )

    assert manifest["train_examples"] == 9
    assert manifest["heldout_examples"] == 3
    assert manifest["train_by_class"] == {"defer": 3, "flag": 3, "ground": 3}
    assert manifest["heldout_by_class"] == {"defer": 1, "flag": 1, "ground": 1}
    assert len(manifest["train_task_ids"]) == 9
    assert len(manifest["heldout_task_ids"]) == 3

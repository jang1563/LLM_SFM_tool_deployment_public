from collections import Counter

import pytest

from post_training.build_sft_cv_splits import build_folds, fold_chunks, split_for_fold


def rows() -> list[dict]:
    out = []
    for action_class in ["defer", "ground"]:
        for i in range(4):
            out.append({
                "id": f"sft::ct::{action_class}::{i}",
                "task_id": f"ct::{action_class}::{i}",
                "action_class": action_class,
            })
    return out


def test_fold_chunks_cover_each_example_once() -> None:
    chunks = fold_chunks(rows(), folds=2, seed=123)
    counts = Counter(row_id for ids in chunks.values() for row_id in ids)

    assert len(chunks) == 2
    assert set(counts.values()) == {1}
    assert len(counts) == len(rows())


def test_fold_chunks_requires_divisible_class_counts() -> None:
    with pytest.raises(ValueError, match="not divisible"):
        fold_chunks(rows(), folds=3, seed=123)


def test_split_for_fold_is_disjoint() -> None:
    chunks = fold_chunks(rows(), folds=2, seed=123)
    train, heldout = split_for_fold(rows(), chunks[0])

    assert len(train) == 4
    assert len(heldout) == 4
    assert {row["id"] for row in train}.isdisjoint({row["id"] for row in heldout})


def test_build_folds_writes_manifest_and_files(tmp_path) -> None:
    manifest = build_folds(
        rows(),
        folds=2,
        seed=123,
        out_dir=tmp_path,
        prefix="toy",
    )

    assert manifest["heldout_coverage_unique_examples"] == 8
    assert manifest["heldout_coverage_min_count"] == 1
    assert manifest["heldout_coverage_max_count"] == 1
    assert len(manifest["fold_manifests"]) == 2
    for fold in manifest["fold_manifests"]:
        assert fold["train_examples"] == 4
        assert fold["heldout_examples"] == 4
        assert fold["train_by_class"] == {"defer": 2, "ground": 2}
        assert fold["heldout_by_class"] == {"defer": 2, "ground": 2}

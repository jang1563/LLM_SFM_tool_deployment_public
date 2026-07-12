from post_training.build_sft_curriculum_data import (
    build_curriculum_rows,
    class_counts,
    family_counts,
)


def row(row_id: str, action_class: str) -> dict:
    return {
        "id": row_id,
        "task_id": row_id.replace("sft::", ""),
        "dataset": "source",
        "action_class": action_class,
        "messages": [],
    }


def balanced_rows() -> list[dict]:
    rows = []
    for action_class in ["defer", "flag", "ground", "reject", "verify"]:
        for i in range(2):
            rows.append(row(f"sft::{action_class}::{i}", action_class))
    return rows


def test_build_curriculum_rows_adds_contrast_blocks() -> None:
    out = build_curriculum_rows(balanced_rows(), dataset="curriculum")

    assert len(out) == 24
    assert class_counts(out) == {
        "defer": 4,
        "flag": 6,
        "ground": 6,
        "reject": 4,
        "verify": 4,
    }
    assert family_counts(out) == {
        "base": 10,
        "ground_flag": 4,
        "reject_override": 6,
        "verify_defer": 4,
    }


def test_curriculum_rows_keep_source_ids_and_unique_ids() -> None:
    out = build_curriculum_rows(balanced_rows(), dataset="curriculum")

    assert len({item["id"] for item in out}) == len(out)
    assert all(item["dataset"] == "curriculum" for item in out)
    assert all("source_example_id" in item for item in out)
    assert out[0]["curriculum_family"] == "base"
    assert out[10]["curriculum_family"] == "ground_flag"

import pytest

from post_training.build_sft_pressure_data import (
    apply_class_pressure,
    balance_to_max_class_count,
    build_native_cv_pressure,
    parse_multipliers,
)


def row(row_id: str, action_class: str) -> dict:
    return {
        "id": row_id,
        "task_id": row_id.replace("sft::", ""),
        "dataset": "source",
        "action_class": action_class,
        "messages": [],
    }


def test_parse_multipliers_requires_positive_class_counts() -> None:
    assert parse_multipliers("flag=3, verify=2") == {"flag": 3, "verify": 2}
    with pytest.raises(ValueError, match=">= 1"):
        parse_multipliers("flag=0")


def test_apply_class_pressure_duplicates_target_classes_with_unique_ids() -> None:
    rows = [row("sft::flag::1", "flag"), row("sft::ground::1", "ground")]

    out = apply_class_pressure(
        rows,
        multipliers={"flag": 3},
        dataset="pressure",
    )

    assert [item["action_class"] for item in out] == ["flag", "flag", "flag", "ground"]
    assert len({item["id"] for item in out}) == 4
    assert out[0]["dataset"] == "pressure"
    assert out[1]["source_example_id"] == "sft::flag::1"
    assert out[2]["pressure_replicate_index"] == 2


def test_balance_to_max_class_count_equalizes_classes() -> None:
    rows = [
        row("sft::ground::1", "ground"),
        row("sft::ground::2", "ground"),
        row("sft::flag::1", "flag"),
    ]

    out = balance_to_max_class_count(rows, dataset="balanced", seed=123)

    assert len(out) == 4
    assert sum(item["action_class"] == "ground" for item in out) == 2
    assert sum(item["action_class"] == "flag" for item in out) == 2
    assert all(item["dataset"] == "balanced" for item in out)


def test_build_native_cv_pressure_writes_fold_train_files(tmp_path) -> None:
    source_train = tmp_path / "fold0_train.jsonl"
    source_train.write_text(
        '{"id":"sft::flag::1","task_id":"ct::flag::1","action_class":"flag"}\n'
        '{"id":"sft::verify::1","task_id":"ct::verify::1","action_class":"verify"}\n'
        '{"id":"sft::ground::1","task_id":"ct::ground::1","action_class":"ground"}\n'
    )
    manifest = {
        "source": "source_sft.jsonl",
        "folds": 1,
        "fold_manifests": [{
            "fold": 0,
            "train": str(source_train),
            "heldout": "heldout.jsonl",
            "heldout_examples": 3,
            "heldout_by_class": {"flag": 1, "verify": 1, "ground": 1},
        }],
    }

    pressure_manifest = build_native_cv_pressure(
        manifest,
        multipliers={"flag": 2, "verify": 3},
        out_dir=tmp_path / "pressure",
        prefix="toy",
        dataset="pressure",
    )

    fold = pressure_manifest["fold_manifests"][0]
    assert fold["source_train_examples"] == 3
    assert fold["train_examples"] == 6
    assert fold["train_by_class"] == {"flag": 2, "ground": 1, "verify": 3}

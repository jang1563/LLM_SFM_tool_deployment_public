from post_training.build_sft_curriculum_v2_data import (
    build_targeted_rows,
    target_family,
)


def test_target_family_names_boundary_groups() -> None:
    assert target_family("defer", "verify") == "target_defer_verify"
    assert target_family("verify", "defer") == "target_defer_verify"
    assert target_family("ground", "flag") == "target_clean_ground"
    assert target_family("reject", "flag") == "target_reject_override"
    assert target_family("flag", "reject") == "target_flag_preserve"


def test_build_targeted_rows_uses_gold_class_weights() -> None:
    rows = [
        {"id": "row-a", "task_id": "packet-a", "action_class": "reject"},
        {"id": "row-b", "task_id": "packet-b", "action_class": "ground"},
        {"id": "row-c", "task_id": "packet-c", "action_class": "flag"},
    ]
    failures = {
        "packet-a": {
            "gold": "reject",
            "pred": "flag",
            "failure_pair": "reject->flag",
            "failure_conditions": 2,
            "note": "mixed endpoints",
        },
        "packet-b": {
            "gold": "ground",
            "pred": "flag",
            "failure_pair": "ground->flag",
            "failure_conditions": 2,
            "note": "efficacy",
        },
    }

    targeted = build_targeted_rows(rows, dataset="dataset", persistent_failures=failures)

    assert [row["task_id"] for row in targeted] == [
        "packet-a",
        "packet-a",
        "packet-a",
        "packet-b",
        "packet-b",
    ]
    assert [row["curriculum_family"] for row in targeted] == [
        "target_reject_override",
        "target_reject_override",
        "target_reject_override",
        "target_clean_ground",
        "target_clean_ground",
    ]
    assert {row["dataset"] for row in targeted} == {"dataset"}

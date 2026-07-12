import json
import subprocess
import sys
from pathlib import Path

from post_training.export_stage_a_saved_output_evidence_candidate_routing_rows import (
    CANDIDATE_PAIRS,
    DATASET,
    build_rows,
    bridge_case_reasons,
    load_json,
    manifest_for_rows,
)
from post_training.validate_post_training_data import (
    load_jsonl,
    validate_stage_a_saved_output_evidence_candidate_routing,
)


ROOT = Path(__file__).resolve().parents[1]


def tracked_artifacts() -> tuple[list[dict], list[dict], list[dict], dict]:
    rows = load_jsonl(
        ROOT / "post_training" / "stage_a_saved_output_evidence_candidate_routing_rows_v1.jsonl"
    )
    train_rows = load_jsonl(
        ROOT / "post_training" / "stage_a_saved_output_evidence_candidate_routing_train_v1.jsonl"
    )
    heldout_rows = load_jsonl(
        ROOT / "post_training" / "stage_a_saved_output_evidence_candidate_routing_heldout_v1.jsonl"
    )
    manifest = json.loads(
        (
            ROOT
            / "post_training"
            / "stage_a_saved_output_evidence_candidate_routing_manifest.json"
        ).read_text()
    )
    return rows, train_rows, heldout_rows, manifest


def test_saved_output_evidence_candidate_routing_rows_validate_tracked_artifacts() -> None:
    rows, train_rows, heldout_rows, manifest = tracked_artifacts()

    assert (
        validate_stage_a_saved_output_evidence_candidate_routing(
            rows, train_rows, heldout_rows, manifest
        )
        == []
    )
    assert manifest["row_count"] == 25
    assert manifest["train_rows"] == 20
    assert manifest["heldout_rows"] == 5
    assert manifest["bridge_focus_rows"] == 4
    assert manifest["candidate_pairs"] == list(CANDIDATE_PAIRS)
    assert manifest["overlap_case_ids"] == []
    assert manifest["overlap_split_groups"] == []
    assert manifest["overlap_source_task_ids"] == []


def test_saved_output_evidence_candidate_routing_balances_pairs_and_focus_cases() -> None:
    rows, train_rows, heldout_rows, manifest = tracked_artifacts()

    assert manifest["by_target_pair"] == {
        "defer/insufficient": 5,
        "flag/invalid_value": 5,
        "ground/supported": 5,
        "reject/contradicted": 5,
        "verify/insufficient": 5,
    }
    assert manifest["train_by_target_pair"] == {
        "defer/insufficient": 4,
        "flag/invalid_value": 4,
        "ground/supported": 4,
        "reject/contradicted": 4,
        "verify/insufficient": 4,
    }
    assert manifest["heldout_by_target_pair"] == {
        "defer/insufficient": 1,
        "flag/invalid_value": 1,
        "ground/supported": 1,
        "reject/contradicted": 1,
        "verify/insufficient": 1,
    }
    assert manifest["bridge_focus_by_target_pair"] == {
        "defer/insufficient": 1,
        "flag/invalid_value": 1,
        "reject/contradicted": 1,
        "verify/insufficient": 1,
    }
    assert not any(row["bridge_focus_case"] for row in train_rows)
    assert sum(1 for row in heldout_rows if row["bridge_focus_case"]) == 4
    assert all(row["dataset"] == DATASET for row in rows)


def test_saved_output_evidence_candidate_routing_model_visible_task_is_label_isolated() -> None:
    rows, train_rows, heldout_rows, _ = tracked_artifacts()

    for row in rows:
        task = row["model_visible_task"]
        text = json.dumps(task, sort_keys=True)
        assert task["candidate_pairs"] == list(CANDIDATE_PAIRS)
        assert row["target_pair"] in task["candidate_pairs"]
        assert "target_pair" not in text
        assert "target_output" not in text
        assert "selected_pair" not in text
        assert "hidden_eval_metadata" not in text
        assert str(row["source_task_id"]) not in text
        assert str(row["split_group"]) not in text
        assert str(row["case_family"]) not in text
        assert row["runtime_evidence_pair"] == row["target_pair"]
        assert row["runtime_evidence_exact"] is True
    assert all(row["training_allowed"] and not row["evaluation_only"] for row in train_rows)
    assert all((not row["training_allowed"]) and row["evaluation_only"] for row in heldout_rows)


def test_saved_output_evidence_candidate_routing_cli_recreates_artifacts(tmp_path: Path) -> None:
    rows_out = tmp_path / "rows.jsonl"
    train_out = tmp_path / "train.jsonl"
    heldout_out = tmp_path / "heldout.jsonl"
    manifest_out = tmp_path / "manifest.json"
    completed = subprocess.run(
        [
            sys.executable,
            "post_training/export_stage_a_saved_output_evidence_candidate_routing_rows.py",
            "--rows-out",
            str(rows_out),
            "--train-out",
            str(train_out),
            "--heldout-out",
            str(heldout_out),
            "--manifest-out",
            str(manifest_out),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    stdout = json.loads(completed.stdout)
    rows = load_jsonl(rows_out)
    train_rows = load_jsonl(train_out)
    heldout_rows = load_jsonl(heldout_out)
    manifest = json.loads(manifest_out.read_text())
    assert stdout == manifest
    assert (
        validate_stage_a_saved_output_evidence_candidate_routing(
            rows, train_rows, heldout_rows, manifest
        )
        == []
    )


def test_saved_output_evidence_candidate_routing_build_rows_from_bridge() -> None:
    source_rows = load_jsonl(
        ROOT / "post_training" / "stage_a_evidence_conditioned_component_targets_v1.jsonl"
    )
    train_source_rows = load_jsonl(
        ROOT / "post_training" / "stage_a_evidence_conditioned_component_targets_train_v1.jsonl"
    )
    heldout_source_rows = load_jsonl(
        ROOT / "post_training" / "stage_a_evidence_conditioned_component_targets_heldout_v1.jsonl"
    )
    bridge = load_json(ROOT / "post_training" / "stage_a_saved_output_evidence_bridge_2026-07-10.json")
    bridge_reasons = bridge_case_reasons(bridge)

    train_rows = build_rows(
        source_rows=source_rows,
        split_rows=train_source_rows,
        split="train",
        bridge_reasons_by_case=bridge_reasons,
    )
    heldout_rows = build_rows(
        source_rows=source_rows,
        split_rows=heldout_source_rows,
        split="heldout",
        bridge_reasons_by_case=bridge_reasons,
    )
    rows = train_rows + heldout_rows
    manifest = manifest_for_rows(
        source_targets="post_training/stage_a_evidence_conditioned_component_targets_v1.jsonl",
        source_train_targets="post_training/stage_a_evidence_conditioned_component_targets_train_v1.jsonl",
        source_heldout_targets="post_training/stage_a_evidence_conditioned_component_targets_heldout_v1.jsonl",
        evidence_bridge="post_training/stage_a_saved_output_evidence_bridge_2026-07-10.json",
        rows_out="post_training/stage_a_saved_output_evidence_candidate_routing_rows_v1.jsonl",
        train_out="post_training/stage_a_saved_output_evidence_candidate_routing_train_v1.jsonl",
        heldout_out="post_training/stage_a_saved_output_evidence_candidate_routing_heldout_v1.jsonl",
        manifest_out="post_training/stage_a_saved_output_evidence_candidate_routing_manifest.json",
        rows=rows,
        train_rows=train_rows,
        heldout_rows=heldout_rows,
        bridge_reasons_by_case=bridge_reasons,
    )

    assert len(rows) == 25
    assert manifest["bridge_focus_case_ids"] == [
        "stage_a::000007",
        "stage_a::000012",
        "stage_a::000019",
        "stage_a::000021",
    ]

import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

from post_training.export_stage_a_enum_action_contrast_pairs import (
    build_pairs,
    manifest_for_pairs,
)
from post_training.validate_post_training_data import (
    load_jsonl,
    validate_stage_a_enum_action_contrast_pairs,
)


ROOT = Path(__file__).resolve().parents[1]


def tracked_artifacts() -> tuple[list[dict], list[dict], list[dict], dict]:
    rows = load_jsonl(ROOT / "post_training" / "stage_a_enum_action_contrast_pairs_v1.jsonl")
    train_rows = load_jsonl(ROOT / "post_training" / "stage_a_enum_action_contrast_pairs_train_v1.jsonl")
    heldout_rows = load_jsonl(ROOT / "post_training" / "stage_a_enum_action_contrast_pairs_heldout_v1.jsonl")
    manifest = json.loads((ROOT / "post_training" / "stage_a_enum_action_contrast_pairs_manifest.json").read_text())
    return rows, train_rows, heldout_rows, manifest


def test_stage_a_enum_action_contrast_pairs_validate_tracked_artifacts() -> None:
    rows, train_rows, heldout_rows, manifest = tracked_artifacts()

    assert validate_stage_a_enum_action_contrast_pairs(rows, train_rows, heldout_rows, manifest) == []
    assert manifest["pair_examples"] == 20
    assert manifest["train_pairs"] == 16
    assert manifest["heldout_pairs"] == 4
    assert manifest["skipped_examples"] == 5
    assert manifest["contrast_axis"] == "action"
    assert manifest["candidate_policy"] == "same_status_action_contrast"
    assert manifest["overlap_case_ids"] == []
    assert manifest["overlap_split_groups"] == []
    assert manifest["overlap_source_task_ids"] == []


def test_stage_a_enum_action_contrast_pairs_preserve_status_and_change_action() -> None:
    rows, _, _, _ = tracked_artifacts()

    for row in rows:
        chosen = row["chosen_output"]
        rejected = row["rejected_output"]
        assert row["component"] == "enum_action"
        assert row["failure_mode"] == "same_status_wrong_action_contrast"
        assert row["contrast_axis"] == "action"
        assert chosen["action"] != "ground"
        assert rejected["action"] == "ground"
        assert rejected["evidence_status"] == chosen["evidence_status"]
        assert row["chosen_pair"] != row["rejected_pair"]
        assert row["chosen_score"]["passed"] is True
        assert row["rejected_score"]["passed"] is False
        assert "target_mismatch" in row["rejected_score"]["violations"]


def test_stage_a_enum_action_contrast_targets_flag_action_for_invalid_value() -> None:
    rows, _, heldout_rows, manifest = tracked_artifacts()
    flag_rows = [row for row in rows if row["chosen_pair"] == "flag/invalid_value"]
    heldout_flag = [row for row in heldout_rows if row["chosen_pair"] == "flag/invalid_value"]

    assert len(flag_rows) == 5
    assert len(heldout_flag) == 1
    assert {row["rejected_pair"] for row in flag_rows} == {"ground/invalid_value"}
    assert manifest["priority"]["chosen_pair"] == "flag/invalid_value"


def test_stage_a_enum_action_contrast_prompts_keep_hidden_fields_out() -> None:
    rows, _, _, _ = tracked_artifacts()

    for row in rows:
        prompt_text = json.dumps(row["prompt_messages"], sort_keys=True)
        assert "hidden_eval_metadata" not in prompt_text
        assert "gold_evidence_status" not in prompt_text
        assert "expected_terminal_action" not in prompt_text
        assert str(row["source_task_id"]) not in prompt_text
        assert str(row["split_group"]) not in prompt_text


def test_stage_a_enum_action_contrast_validator_fails_if_status_not_preserved() -> None:
    rows, train_rows, heldout_rows, manifest = tracked_artifacts()
    broken = deepcopy(rows)
    broken[0]["rejected_output"]["evidence_status"] = "unknown"

    issues = validate_stage_a_enum_action_contrast_pairs(broken, train_rows, heldout_rows, manifest)

    assert any(issue.endswith("status_not_preserved") for issue in issues)


def test_stage_a_enum_action_contrast_export_cli_recreates_artifacts(tmp_path: Path) -> None:
    pairs = tmp_path / "pairs.jsonl"
    train = tmp_path / "pairs_train.jsonl"
    heldout = tmp_path / "pairs_heldout.jsonl"
    manifest_path = tmp_path / "manifest.json"
    result = subprocess.run(
        [
            sys.executable,
            "post_training/export_stage_a_enum_action_contrast_pairs.py",
            "--pairs-out",
            str(pairs),
            "--train-out",
            str(train),
            "--heldout-out",
            str(heldout),
            "--manifest-out",
            str(manifest_path),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    rows = load_jsonl(pairs)
    train_rows = load_jsonl(train)
    heldout_rows = load_jsonl(heldout)
    manifest = json.loads(manifest_path.read_text())
    assert validate_stage_a_enum_action_contrast_pairs(rows, train_rows, heldout_rows, manifest) == []


def test_stage_a_enum_action_contrast_build_pairs_skips_ground_targets() -> None:
    source_rows = load_jsonl(ROOT / "post_training" / "stage_a_strict_component_targets_v1.jsonl")
    pairs = build_pairs(source_rows)
    manifest = manifest_for_pairs(
        source_targets="post_training/stage_a_strict_component_targets_v1.jsonl",
        source_train_targets="post_training/stage_a_strict_component_targets_train_v1.jsonl",
        source_heldout_targets="post_training/stage_a_strict_component_targets_heldout_v1.jsonl",
        pairs_out="post_training/stage_a_enum_action_contrast_pairs_v1.jsonl",
        train_out="post_training/stage_a_enum_action_contrast_pairs_train_v1.jsonl",
        heldout_out="post_training/stage_a_enum_action_contrast_pairs_heldout_v1.jsonl",
        rows=pairs,
        train_rows=[row for row in pairs if row["split"] == "train"],
        heldout_rows=[row for row in pairs if row["split"] == "heldout"],
        skipped_rows=[],
    )

    assert len(pairs) == 20
    assert "ground/supported" not in manifest["by_chosen_pair"]
    assert manifest["by_rejected_pair"]["ground/invalid_value"] == 5

import json
import subprocess
import sys
from pathlib import Path

from post_training.export_stage_a_enum_corrective_pairs import (
    COLLAPSE_OUTPUT,
    build_pairs,
    manifest_for_pairs,
)
from post_training.validate_post_training_data import (
    load_jsonl,
    validate_stage_a_enum_corrective_pairs,
)


ROOT = Path(__file__).resolve().parents[1]


def tracked_artifacts() -> tuple[list[dict], list[dict], list[dict], dict]:
    rows = load_jsonl(ROOT / "post_training" / "stage_a_enum_corrective_pairs_v1.jsonl")
    train_rows = load_jsonl(ROOT / "post_training" / "stage_a_enum_corrective_pairs_train_v1.jsonl")
    heldout_rows = load_jsonl(ROOT / "post_training" / "stage_a_enum_corrective_pairs_heldout_v1.jsonl")
    manifest = json.loads((ROOT / "post_training" / "stage_a_enum_corrective_pairs_manifest.json").read_text())
    return rows, train_rows, heldout_rows, manifest


def test_stage_a_enum_corrective_pairs_validate_tracked_artifacts() -> None:
    rows, train_rows, heldout_rows, manifest = tracked_artifacts()

    assert validate_stage_a_enum_corrective_pairs(rows, train_rows, heldout_rows, manifest) == []
    assert manifest["pair_examples"] == 20
    assert manifest["train_pairs"] == 16
    assert manifest["heldout_pairs"] == 4
    assert manifest["skipped_examples"] == 5
    assert manifest["rejected_output"] == COLLAPSE_OUTPUT
    assert manifest["overlap_case_ids"] == []
    assert manifest["overlap_split_groups"] == []
    assert manifest["overlap_source_task_ids"] == []


def test_stage_a_enum_corrective_pairs_are_ground_supported_contrasts() -> None:
    rows, _, _, _ = tracked_artifacts()

    for row in rows:
        assert row["component"] == "enum_action"
        assert row["failure_mode"] == "ground_supported_collapse"
        assert row["rejected_output"] == COLLAPSE_OUTPUT
        assert row["chosen_output"] != row["rejected_output"]
        assert row["chosen_score"]["passed"] is True
        assert row["rejected_score"]["passed"] is False
        assert "target_mismatch" in row["rejected_score"]["violations"]
        assert row["rejected_pair"] == "ground/supported"


def test_stage_a_enum_corrective_prompts_keep_hidden_fields_out() -> None:
    rows, _, _, _ = tracked_artifacts()

    for row in rows:
        prompt_text = json.dumps(row["prompt_messages"], sort_keys=True)
        assert "hidden_eval_metadata" not in prompt_text
        assert "gold_evidence_status" not in prompt_text
        assert "expected_terminal_action" not in prompt_text
        assert str(row["source_task_id"]) not in prompt_text
        assert str(row["split_group"]) not in prompt_text


def test_stage_a_enum_corrective_export_cli_recreates_artifacts(tmp_path: Path) -> None:
    pairs = tmp_path / "pairs.jsonl"
    train = tmp_path / "pairs_train.jsonl"
    heldout = tmp_path / "pairs_heldout.jsonl"
    manifest_path = tmp_path / "manifest.json"
    result = subprocess.run(
        [
            sys.executable,
            "post_training/export_stage_a_enum_corrective_pairs.py",
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
    assert validate_stage_a_enum_corrective_pairs(rows, train_rows, heldout_rows, manifest) == []


def test_stage_a_enum_corrective_build_pairs_skips_ground_targets() -> None:
    source_rows = load_jsonl(ROOT / "post_training" / "stage_a_strict_component_targets_v1.jsonl")
    pairs = build_pairs(source_rows)
    manifest = manifest_for_pairs(
        source_targets="post_training/stage_a_strict_component_targets_v1.jsonl",
        source_train_targets="post_training/stage_a_strict_component_targets_train_v1.jsonl",
        source_heldout_targets="post_training/stage_a_strict_component_targets_heldout_v1.jsonl",
        pairs_out="post_training/stage_a_enum_corrective_pairs_v1.jsonl",
        train_out="post_training/stage_a_enum_corrective_pairs_train_v1.jsonl",
        heldout_out="post_training/stage_a_enum_corrective_pairs_heldout_v1.jsonl",
        rows=pairs,
        train_rows=[row for row in pairs if row["split"] == "train"],
        heldout_rows=[row for row in pairs if row["split"] == "heldout"],
        skipped_rows=[],
    )

    assert len(pairs) == 20
    assert "ground/supported" not in manifest["by_chosen_pair"]

from pathlib import Path
import json
import subprocess
import sys

from post_training.split_stage_a_data import (
    build_stage_a_split,
    load_jsonl,
    manifest_for_stage_a_split,
    split_case_ids,
)


ROOT = Path(__file__).resolve().parents[1]


def stage_a_rows():
    return (
        load_jsonl(ROOT / "post_training" / "stage_a_sft_v1.jsonl"),
        load_jsonl(ROOT / "post_training" / "stage_a_preferences_v1.jsonl"),
        load_jsonl(ROOT / "post_training" / "stage_a_process_supervision_v1.jsonl"),
    )


def test_stage_a_split_balances_heldout_by_family() -> None:
    sft, prefs, process = stage_a_rows()
    splits = build_stage_a_split(sft, prefs, process, heldout_per_family=1, seed=20260704)
    manifest = manifest_for_stage_a_split(
        source_export_manifest="post_training/stage_a_export_manifest.json",
        source_sft="post_training/stage_a_sft_v1.jsonl",
        source_preferences="post_training/stage_a_preferences_v1.jsonl",
        source_process="post_training/stage_a_process_supervision_v1.jsonl",
        train_sft_path="post_training/stage_a_sft_train_v1.jsonl",
        heldout_sft_path="post_training/stage_a_sft_heldout_v1.jsonl",
        train_preferences_path="post_training/stage_a_preferences_train_v1.jsonl",
        heldout_preferences_path="post_training/stage_a_preferences_heldout_v1.jsonl",
        train_process_path="post_training/stage_a_process_train_v1.jsonl",
        heldout_process_path="post_training/stage_a_process_heldout_v1.jsonl",
        splits=splits,
        seed=20260704,
        heldout_per_family=1,
    )

    assert manifest["train_sft_examples"] == 20
    assert manifest["heldout_sft_examples"] == 5
    assert manifest["train_preference_pairs"] == 120
    assert manifest["heldout_preference_pairs"] == 30
    assert set(manifest["heldout_by_case_family"].values()) == {1}
    assert manifest["overlap_case_ids"] == []
    assert manifest["overlap_split_groups"] == []
    assert manifest["overlap_source_task_ids"] == []


def test_stage_a_split_case_ids_are_deterministic() -> None:
    sft, _, _ = stage_a_rows()

    first = split_case_ids(sft, heldout_per_family=1, seed=20260704)
    second = split_case_ids(sft, heldout_per_family=1, seed=20260704)

    assert first == second


def test_stage_a_split_rejects_impossible_family_holdout() -> None:
    sft, _, _ = stage_a_rows()

    try:
        split_case_ids(sft, heldout_per_family=5, seed=20260704)
    except ValueError as exc:
        assert "cannot hold out" in str(exc)
    else:
        raise AssertionError("Expected split_case_ids to reject all-row heldout families")


def test_stage_a_split_script_writes_valid_files(tmp_path: Path) -> None:
    train_sft = tmp_path / "train_sft.jsonl"
    heldout_sft = tmp_path / "heldout_sft.jsonl"
    train_prefs = tmp_path / "train_prefs.jsonl"
    heldout_prefs = tmp_path / "heldout_prefs.jsonl"
    train_process = tmp_path / "train_process.jsonl"
    heldout_process = tmp_path / "heldout_process.jsonl"
    manifest = tmp_path / "manifest.json"

    result = subprocess.run(
        [
            sys.executable,
            "post_training/split_stage_a_data.py",
            "--train-sft-out",
            str(train_sft),
            "--heldout-sft-out",
            str(heldout_sft),
            "--train-preferences-out",
            str(train_prefs),
            "--heldout-preferences-out",
            str(heldout_prefs),
            "--train-process-out",
            str(train_process),
            "--heldout-process-out",
            str(heldout_process),
            "--manifest-out",
            str(manifest),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(manifest.read_text())
    assert payload["train_cases"] == 20
    assert payload["heldout_cases"] == 5
    assert payload["overlap_split_groups"] == []

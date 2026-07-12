import json
import subprocess
import sys
from pathlib import Path

from post_training.validate_post_training_data import (
    load_jsonl,
    validate_stage_a_strict_component_targets,
)


ROOT = Path(__file__).resolve().parents[1]


def test_stage_a_strict_component_targets_validate_tracked_artifacts() -> None:
    rows = load_jsonl(ROOT / "post_training" / "stage_a_strict_component_targets_v1.jsonl")
    train_rows = load_jsonl(ROOT / "post_training" / "stage_a_strict_component_targets_train_v1.jsonl")
    heldout_rows = load_jsonl(ROOT / "post_training" / "stage_a_strict_component_targets_heldout_v1.jsonl")
    manifest = json.loads((ROOT / "post_training" / "stage_a_strict_component_targets_manifest.json").read_text())

    assert validate_stage_a_strict_component_targets(rows, train_rows, heldout_rows, manifest) == []
    assert manifest["target_examples"] == 75
    assert manifest["train_target_examples"] == 60
    assert manifest["heldout_target_examples"] == 15
    assert manifest["overlap_case_ids"] == []
    assert manifest["by_component"] == {
        "enum_action": 25,
        "routing_after_loop": 25,
        "tool_query": 25,
    }


def test_stage_a_strict_component_targets_keep_prompts_model_visible_only() -> None:
    rows = load_jsonl(ROOT / "post_training" / "stage_a_strict_component_targets_v1.jsonl")

    for row in rows:
        prompt_text = json.dumps(row["prompt_messages"], sort_keys=True)
        assert "hidden_eval_metadata" not in prompt_text
        assert "gold_evidence_status" not in prompt_text
        assert "expected_terminal_action" not in prompt_text
        assert str(row["source_task_id"]) not in prompt_text
        assert str(row["split_group"]) not in prompt_text


def test_stage_a_strict_component_targets_have_slice_specific_schema() -> None:
    rows = load_jsonl(ROOT / "post_training" / "stage_a_strict_component_targets_v1.jsonl")
    by_component = {row["component"]: row for row in rows[:3]}

    assert set(by_component["enum_action"]["target_output"]) == {"action", "evidence_status"}
    assert set(by_component["tool_query"]["target_output"]) == {"tool_calls"}
    assert set(by_component["routing_after_loop"]["target_output"]) == {
        "action",
        "evidence_status",
        "cited_source_ids",
    }
    tool_call = by_component["tool_query"]["target_output"]["tool_calls"][0]
    assert {"drug_id", "condition_id"}.issubset(tool_call["arguments"])
    routing_prompt = json.loads(by_component["routing_after_loop"]["prompt_messages"][1]["content"])
    assert "observed_tool_loop" in routing_prompt
    assert "observed_tool_loop" not in json.loads(by_component["enum_action"]["prompt_messages"][1]["content"])


def test_stage_a_strict_component_export_cli_recreates_artifacts(tmp_path: Path) -> None:
    targets = tmp_path / "component_targets.jsonl"
    train = tmp_path / "component_targets_train.jsonl"
    heldout = tmp_path / "component_targets_heldout.jsonl"
    manifest_path = tmp_path / "component_manifest.json"
    result = subprocess.run(
        [
            sys.executable,
            "post_training/export_stage_a_strict_component_targets.py",
            "--targets-out",
            str(targets),
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
    rows = load_jsonl(targets)
    train_rows = load_jsonl(train)
    heldout_rows = load_jsonl(heldout)
    manifest = json.loads(manifest_path.read_text())
    assert validate_stage_a_strict_component_targets(rows, train_rows, heldout_rows, manifest) == []

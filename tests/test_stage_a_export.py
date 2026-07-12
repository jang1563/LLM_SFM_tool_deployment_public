from pathlib import Path
import json
import subprocess
import sys

from negbiodb_ct.stage_a_manifest import load_stage_a_manifest
from post_training.export_stage_a_data import (
    build_stage_a_exports,
    manifest_for_exports,
    preference_rows,
    prompt_messages,
    sft_row,
)


ROOT = Path(__file__).resolve().parents[1]


def first_row():
    return load_stage_a_manifest(ROOT / "negbiodb_ct" / "stage_a_mini_manifest.jsonl", limit=1)[0]


def test_stage_a_sft_row_preserves_prompt_target_boundary() -> None:
    row = first_row()
    out = sft_row(row)
    prompt_text = json.dumps(out["messages"][:2], sort_keys=True)

    assert out["dataset"] == "negbiodb_ct_stage_a_sft_v1"
    assert out["oracle_target"] is True
    assert out["score"]["passed"] is True
    assert out["messages"][-1]["tool_call"]["name"] == "submit_decision"
    assert row["hidden_eval_metadata"]["source_task_id"] not in prompt_text
    assert "hidden_eval_metadata" not in prompt_text
    assert "gold_evidence_status" not in prompt_text


def test_stage_a_preference_rows_have_passing_chosen_and_failing_rejected() -> None:
    row = first_row()
    pairs = preference_rows(row)

    assert pairs
    assert all(pair["chosen_score"]["passed"] for pair in pairs)
    assert not any(pair["rejected_score"]["passed"] for pair in pairs)
    assert any(pair["failure_mode"] == "self_answering_without_tools" for pair in pairs)
    assert all(pair["prompt_messages"] == prompt_messages(row) for pair in pairs)


def test_stage_a_export_manifest_counts_match_rows() -> None:
    rows = load_stage_a_manifest(ROOT / "negbiodb_ct" / "stage_a_mini_manifest.jsonl")
    sft, prefs, process = build_stage_a_exports(rows)
    manifest = manifest_for_exports(
        source_manifest="negbiodb_ct/stage_a_mini_manifest.jsonl",
        sft_out="post_training/stage_a_sft_v1.jsonl",
        preference_out="post_training/stage_a_preferences_v1.jsonl",
        process_out="post_training/stage_a_process_supervision_v1.jsonl",
        rows=rows,
        sft_rows=sft,
        preference_rows_out=prefs,
        process_rows=process,
    )

    assert manifest["source_cases"] == 25
    assert manifest["sft_examples"] == 25
    assert manifest["process_examples"] == 25
    assert manifest["preference_pairs"] == len(prefs)
    assert manifest["chosen_passed"] == len(prefs)
    assert manifest["rejected_passed"] == 0
    assert manifest["split_group_overlap"] == []


def test_stage_a_export_script_writes_valid_files(tmp_path: Path) -> None:
    sft = tmp_path / "sft.jsonl"
    prefs = tmp_path / "prefs.jsonl"
    process = tmp_path / "process.jsonl"
    manifest = tmp_path / "manifest.json"
    result = subprocess.run(
        [
            sys.executable,
            "post_training/export_stage_a_data.py",
            "--sft-out",
            str(sft),
            "--preference-out",
            str(prefs),
            "--process-out",
            str(process),
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
    assert payload["sft_examples"] == 25
    assert payload["process_examples"] == 25
    assert payload["preference_pairs"] > 25

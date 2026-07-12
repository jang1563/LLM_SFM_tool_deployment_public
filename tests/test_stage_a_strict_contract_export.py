import json
import subprocess
import sys
from pathlib import Path

from post_training.export_stage_a_strict_contract_data import (
    STRICT_FAILURE_MODES,
    apply_split_manifest,
    build_strict_contract_exports,
    manifest_for_exports,
    preference_rows,
    sft_row,
)
from post_training.run_stage_a_sft_smoke_eval import load_jsonl, load_manifest_rows
from post_training.validate_post_training_data import validate_stage_a_strict_contract


ROOT = Path(__file__).resolve().parents[1]


def source_sft_rows() -> list[dict]:
    return load_jsonl(ROOT / "post_training" / "stage_a_sft_v1.jsonl")


def manifest_rows() -> list[dict]:
    return load_manifest_rows(ROOT / "negbiodb_ct" / "stage_a_mini_manifest.jsonl")


def test_strict_contract_sft_target_roundtrips_through_evaluator() -> None:
    row = source_sft_rows()[0]
    out = sft_row(row, manifest_rows=manifest_rows())
    prompt_text = json.dumps(out["messages"][:-1], sort_keys=True)

    assert out["dataset"] == "negbiodb_ct_stage_a_strict_contract_sft_v1"
    assert out["oracle_target"] is True
    assert out["prompt_contract"] == "stage_a_v2_strict"
    assert out["score"]["passed"] is True
    assert out["target_output"]["tool_calls"]
    assert out["messages"][-1]["role"] == "assistant"
    assert "Strict Stage A output contract" in prompt_text
    assert "gold_evidence_status" not in prompt_text
    assert row["source_task_id"] not in prompt_text


def test_strict_contract_preferences_have_passing_chosen_and_failing_observed_collapses() -> None:
    row = source_sft_rows()[0]
    pairs = preference_rows(row, manifest_rows=manifest_rows())

    assert [pair["failure_mode"] for pair in pairs] == list(STRICT_FAILURE_MODES)
    assert all(pair["chosen_score"]["passed"] for pair in pairs)
    assert not any(pair["rejected_score"]["passed"] for pair in pairs)
    assert all(pair["rejected_score"]["violations"] for pair in pairs)
    assert pairs[0]["rejected_output"]["action"] == "verify"
    assert pairs[0]["rejected_output"]["evidence_status"] == "supported"


def test_strict_contract_validator_accepts_generated_rows() -> None:
    sft, prefs, process = build_strict_contract_exports(source_sft_rows(), manifest_rows())
    split_manifest = json.loads((ROOT / "post_training" / "stage_a_split_manifest.json").read_text())
    splits = apply_split_manifest(sft, prefs, process, split_manifest)
    manifest = manifest_for_exports(
        source_sft="post_training/stage_a_sft_v1.jsonl",
        source_manifest="negbiodb_ct/stage_a_mini_manifest.jsonl",
        source_split_manifest="post_training/stage_a_split_manifest.json",
        sft_out="post_training/stage_a_strict_contract_sft_v1.jsonl",
        preference_out="post_training/stage_a_strict_contract_preferences_v1.jsonl",
        process_out="post_training/stage_a_strict_contract_process_v1.jsonl",
        split_paths={
            "train_sft": "post_training/stage_a_strict_contract_sft_train_v1.jsonl",
            "heldout_sft": "post_training/stage_a_strict_contract_sft_heldout_v1.jsonl",
            "train_preferences": "post_training/stage_a_strict_contract_preferences_train_v1.jsonl",
            "heldout_preferences": "post_training/stage_a_strict_contract_preferences_heldout_v1.jsonl",
            "train_process": "post_training/stage_a_strict_contract_process_train_v1.jsonl",
            "heldout_process": "post_training/stage_a_strict_contract_process_heldout_v1.jsonl",
        },
        sft_rows=sft,
        preference_rows_out=prefs,
        process_rows=process,
        splits=splits,
    )

    issues = validate_stage_a_strict_contract(
        sft,
        prefs,
        process,
        splits["train_sft"],
        splits["heldout_sft"],
        splits["train_preferences"],
        splits["heldout_preferences"],
        splits["train_process"],
        splits["heldout_process"],
        manifest,
    )

    assert issues == []


def test_strict_contract_export_script_writes_valid_files(tmp_path: Path) -> None:
    paths = {
        "sft": tmp_path / "strict_sft.jsonl",
        "prefs": tmp_path / "strict_prefs.jsonl",
        "process": tmp_path / "strict_process.jsonl",
        "train_sft": tmp_path / "strict_sft_train.jsonl",
        "heldout_sft": tmp_path / "strict_sft_heldout.jsonl",
        "train_prefs": tmp_path / "strict_prefs_train.jsonl",
        "heldout_prefs": tmp_path / "strict_prefs_heldout.jsonl",
        "train_process": tmp_path / "strict_process_train.jsonl",
        "heldout_process": tmp_path / "strict_process_heldout.jsonl",
        "manifest": tmp_path / "strict_manifest.json",
    }
    result = subprocess.run(
        [
            sys.executable,
            "post_training/export_stage_a_strict_contract_data.py",
            "--sft-out",
            str(paths["sft"]),
            "--preference-out",
            str(paths["prefs"]),
            "--process-out",
            str(paths["process"]),
            "--train-sft-out",
            str(paths["train_sft"]),
            "--heldout-sft-out",
            str(paths["heldout_sft"]),
            "--train-preferences-out",
            str(paths["train_prefs"]),
            "--heldout-preferences-out",
            str(paths["heldout_prefs"]),
            "--train-process-out",
            str(paths["train_process"]),
            "--heldout-process-out",
            str(paths["heldout_process"]),
            "--manifest-out",
            str(paths["manifest"]),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(paths["manifest"].read_text())
    assert payload["sft_examples"] == 25
    assert payload["preference_pairs"] == 50
    assert payload["process_examples"] == 25
    assert payload["train_sft_examples"] == 20
    assert payload["heldout_sft_examples"] == 5
    assert payload["rejected_passed"] == 0

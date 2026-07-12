import json
import subprocess
import sys
from pathlib import Path

from post_training.export_stage_a_saved_output_calibration_probe import (
    NEXT_DECISION_PATH,
    PROMPT_CONTRACT,
    build_pairs,
    load_json,
    manifest_for_pairs,
    pair_label,
    target_failure_pairs,
)
from post_training.run_stage_a_sft_smoke_eval import load_jsonl, load_manifest_rows


ROOT = Path(__file__).resolve().parents[1]


def manifest_rows() -> list[dict]:
    return load_manifest_rows(ROOT / "negbiodb_ct" / "stage_a_mini_manifest.jsonl")


def train_sft_rows() -> list[dict]:
    return load_jsonl(ROOT / "post_training" / "stage_a_sft_train_v1.jsonl")


def heldout_sft_rows() -> list[dict]:
    return load_jsonl(ROOT / "post_training" / "stage_a_sft_heldout_v1.jsonl")


def selected_pairs() -> list[dict[str, str]]:
    return target_failure_pairs(load_json(ROOT / NEXT_DECISION_PATH))


def build_probe_rows() -> tuple[list[dict], list[dict], list[dict]]:
    labels = {pair_label(pair) for pair in selected_pairs()}
    train_rows = build_pairs(
        train_sft_rows(),
        manifest_rows=manifest_rows(),
        selected_pairs=labels,
        source_next_decision=NEXT_DECISION_PATH,
        prompt_contract=PROMPT_CONTRACT,
    )
    heldout_rows = build_pairs(
        heldout_sft_rows(),
        manifest_rows=manifest_rows(),
        selected_pairs=labels,
        source_next_decision=NEXT_DECISION_PATH,
        prompt_contract=PROMPT_CONTRACT,
    )
    return train_rows + heldout_rows, train_rows, heldout_rows


def test_saved_output_calibration_probe_uses_next_decision_targets() -> None:
    assert [pair_label(pair) for pair in selected_pairs()] == [
        "defer/insufficient",
        "flag/invalid_value",
        "reject/contradicted",
        "verify/insufficient",
    ]


def test_saved_output_calibration_probe_is_split_safe_and_evaluable() -> None:
    rows, train_rows, heldout_rows = build_probe_rows()
    report = manifest_for_pairs(
        next_decision_path=NEXT_DECISION_PATH,
        train_sft="post_training/stage_a_sft_train_v1.jsonl",
        heldout_sft="post_training/stage_a_sft_heldout_v1.jsonl",
        pairs_out="post_training/stage_a_saved_output_calibration_probe_v1.jsonl",
        train_out="post_training/stage_a_saved_output_calibration_probe_train_v1.jsonl",
        heldout_out="post_training/stage_a_saved_output_calibration_probe_heldout_v1.jsonl",
        rows=rows,
        train_rows=train_rows,
        heldout_rows=heldout_rows,
        selected_pairs=selected_pairs(),
        prompt_contract=PROMPT_CONTRACT,
    )

    assert len(rows) == 20
    assert len(train_rows) == 16
    assert len(heldout_rows) == 4
    assert report["train_by_chosen_pair"] == {
        "defer/insufficient": 4,
        "flag/invalid_value": 4,
        "reject/contradicted": 4,
        "verify/insufficient": 4,
    }
    assert report["heldout_by_chosen_pair"] == {
        "defer/insufficient": 1,
        "flag/invalid_value": 1,
        "reject/contradicted": 1,
        "verify/insufficient": 1,
    }
    assert report["overlap_case_ids"] == []
    assert report["overlap_split_groups"] == []
    assert report["overlap_source_task_ids"] == []
    assert all(row["training_allowed"] is True for row in train_rows)
    assert all(row["evaluation_only"] is True for row in heldout_rows)
    assert all(row["chosen_score"]["passed"] is True for row in rows)
    assert all(row["rejected_score"]["passed"] is False for row in rows)

    invalid_value_rows = [row for row in rows if row["chosen_pair"] == "flag/invalid_value"]
    assert invalid_value_rows
    assert all(row["chosen_output"]["cited_source_ids"] for row in invalid_value_rows)


def test_saved_output_calibration_probe_cli_writes_public_safe_artifacts(tmp_path: Path) -> None:
    pairs_out = tmp_path / "pairs.jsonl"
    train_out = tmp_path / "train.jsonl"
    heldout_out = tmp_path / "heldout.jsonl"
    manifest_out = tmp_path / "manifest.json"
    result = subprocess.run(
        [
            sys.executable,
            "post_training/export_stage_a_saved_output_calibration_probe.py",
            "--pairs-out",
            str(pairs_out),
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
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    stdout = json.loads(result.stdout)
    manifest = json.loads(manifest_out.read_text())
    assert stdout["pair_examples"] == 20
    assert manifest["train_pairs"] == 16
    assert manifest["heldout_probe_pairs"] == 4
    assert len(load_jsonl(pairs_out)) == 20
    assert len(load_jsonl(train_out)) == 16
    assert len(load_jsonl(heldout_out)) == 4
    assert "raw_output" not in manifest_out.read_text()
    assert "hidden_eval_metadata" not in manifest_out.read_text()

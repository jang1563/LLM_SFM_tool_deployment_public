import json
import subprocess
import sys
from pathlib import Path

from post_training.run_stage_a_strict_contract_sft_smoke import (
    load_jsonl,
    prompt_messages_from_row,
    target_output_from_row,
    validate_strict_rows,
)


ROOT = Path(__file__).resolve().parents[1]


def train_rows() -> list[dict]:
    return load_jsonl(ROOT / "post_training" / "stage_a_strict_contract_sft_train_v1.jsonl")


def heldout_rows() -> list[dict]:
    return load_jsonl(ROOT / "post_training" / "stage_a_strict_contract_sft_heldout_v1.jsonl")


def test_strict_contract_sft_smoke_validates_split_and_prompt_boundary() -> None:
    train = train_rows()
    heldout = heldout_rows()
    row = train[0]
    prompt_text = json.dumps(prompt_messages_from_row(row), sort_keys=True)
    target = target_output_from_row(row)

    assert validate_strict_rows(train, heldout) == []
    assert "Strict Stage A output contract" in prompt_text
    assert "gold_evidence_status" not in prompt_text
    assert row["source_task_id"] not in prompt_text
    assert set(target) == {"action", "evidence_status", "tool_calls", "cited_source_ids", "rationale"}
    assert target["tool_calls"][0]["arguments"]["drug_id"] == "<drug_id>"


def test_strict_contract_sft_smoke_flags_case_overlap() -> None:
    train = train_rows()
    heldout = heldout_rows()
    heldout[0] = dict(train[0])

    issues = validate_strict_rows(train, heldout)

    assert "train_heldout_case_overlap" in issues


def test_strict_contract_sft_smoke_cli_dry_run_writes_report(tmp_path: Path) -> None:
    out_dir = tmp_path / "strict_sft_dry_run"
    result = subprocess.run(
        [
            sys.executable,
            "post_training/run_stage_a_strict_contract_sft_smoke.py",
            "--dry-run",
            "--out-dir",
            str(out_dir),
            "--run-id",
            "unit_strict_sft_dry_run",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((out_dir / "report.json").read_text())
    assert report["dry_run"] is True
    assert report["train_examples"] == 20
    assert report["heldout_examples"] == 5
    assert report["issues"] == []
    assert report["prompt_contract"] == "stage_a_v2_strict"


def test_strict_contract_sft_smoke_requires_explicit_model_load_for_full_mode(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "post_training/run_stage_a_strict_contract_sft_smoke.py",
            "--out-dir",
            str(tmp_path / "strict_sft_blocked_full_run"),
            "--limit-train",
            "1",
            "--limit-heldout",
            "1",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "--allow-model-load" in result.stderr


def test_strict_contract_cluster_templates_call_sft_smoke_runner() -> None:
    for rel in (
        "post_training/run_stage_a_strict_sft_cayuga.sbatch",
        "post_training/run_stage_a_strict_sft_expanse.sbatch",
    ):
        text = (ROOT / rel).read_text()
        assert "run_stage_a_strict_contract_sft_smoke.py" in text
        assert "--allow-model-load" in text
        assert "stage_a_strict_contract_sft_train_v1.jsonl" in text
        assert "stage_a_strict_contract_sft_heldout_v1.jsonl" in text
        assert "post_training/runs/" in text
        assert "<allocation>" in text

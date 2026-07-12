import json
import subprocess
import sys
from pathlib import Path

from post_training.evaluate_stage_a_saved_output_evidence_candidate_routing_dry_run_checkpoint import (
    DATASET,
    build_report,
    render_markdown,
)
from post_training.run_stage_a_saved_output_evidence_candidate_routing_smoke import (
    DATASET as SMOKE_DATASET,
)


ROOT = Path(__file__).resolve().parents[1]


def dry_run_payload(**overrides: object) -> dict:
    payload = {
        "dataset": SMOKE_DATASET,
        "dry_run": True,
        "rows": "post_training/stage_a_saved_output_evidence_candidate_routing_rows_v1.jsonl",
        "train_rows": "post_training/stage_a_saved_output_evidence_candidate_routing_train_v1.jsonl",
        "heldout_rows": "post_training/stage_a_saved_output_evidence_candidate_routing_heldout_v1.jsonl",
        "manifest": "post_training/stage_a_saved_output_evidence_candidate_routing_manifest.json",
        "train_examples": 20,
        "heldout_examples": 5,
        "bridge_focus_heldout_examples": 4,
        "candidate_space_size": 5,
        "ready_for_full_mode": True,
        "issues": [],
        "public_safety_contract": {
            "raw_prediction_jsonl_read": False,
            "raw_candidate_score_jsonl_read": False,
            "scheduler_logs_read": False,
            "model_state_read": False,
            "ignored_run_folder_read": False,
            "hidden_labels_used_for_training": False,
        },
    }
    payload.update(overrides)
    return payload


def write_dry_run(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "dry_run_report.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return path


def test_dry_run_checkpoint_passes_clean_report(tmp_path: Path) -> None:
    report = build_report(
        dry_run_report_path=write_dry_run(tmp_path, dry_run_payload()),
        execution_surface="unit",
        mirror_commit="abc1234",
        public_release_check_passed=True,
    )

    assert report["dataset"] == DATASET
    assert report["passes_checkpoint"] is True
    assert report["checkpoint_violations"] == []
    assert report["dry_run_summary"]["train_examples"] == 20
    assert report["dry_run_summary"]["heldout_examples"] == 5
    assert report["dry_run_summary"]["bridge_focus_heldout_examples"] == 4
    assert report["public_safety_contract"]["raw_fields_in_dry_run_report"] is False
    assert report["next_decision"] == (
        "explicitly_approve_full_cayuga_smoke_or_keep_no_submit_boundary"
    )


def test_dry_run_checkpoint_fails_on_counts_and_release_gate(tmp_path: Path) -> None:
    report = build_report(
        dry_run_report_path=write_dry_run(
            tmp_path,
            dry_run_payload(heldout_examples=4, ready_for_full_mode=False),
        ),
        execution_surface="unit",
        mirror_commit="abc1234",
        public_release_check_passed=False,
    )

    assert report["passes_checkpoint"] is False
    assert report["checkpoint_violations"] == [
        "heldout_examples_mismatch",
        "not_ready_for_full_mode",
        "public_release_check_not_passed",
    ]


def test_dry_run_checkpoint_fails_closed_on_raw_fields(tmp_path: Path) -> None:
    payload = dry_run_payload(candidate_scores=[{"score": 0.0}])
    report = build_report(
        dry_run_report_path=write_dry_run(tmp_path, payload),
        execution_surface="unit",
        mirror_commit="abc1234",
        public_release_check_passed=True,
    )

    assert report["passes_checkpoint"] is False
    assert report["checkpoint_violations"] == ["raw_fields_present"]
    assert report["public_safety_contract"]["raw_fields_in_dry_run_report"] is True


def test_dry_run_checkpoint_markdown(tmp_path: Path) -> None:
    report = build_report(
        dry_run_report_path=write_dry_run(tmp_path, dry_run_payload()),
        execution_surface="unit",
        mirror_commit="abc1234",
        public_release_check_passed=True,
    )
    markdown = render_markdown(report)

    assert "Train rows: 20" in markdown
    assert "Held-out rows: 5" in markdown
    assert "Bridge-focus held-out rows: 4" in markdown
    assert "Passes checkpoint: `True`" in markdown


def test_dry_run_checkpoint_cli_writes_outputs(tmp_path: Path) -> None:
    dry_run = write_dry_run(tmp_path, dry_run_payload())
    out_json = tmp_path / "checkpoint.json"
    out_md = tmp_path / "CHECKPOINT.md"
    completed = subprocess.run(
        [
            sys.executable,
            "post_training/evaluate_stage_a_saved_output_evidence_candidate_routing_dry_run_checkpoint.py",
            "--dry-run-report",
            str(dry_run),
            "--execution-surface",
            "unit",
            "--mirror-commit",
            "abc1234",
            "--public-release-check-passed",
            "--out-json",
            str(out_json),
            "--out-md",
            str(out_md),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    stdout = json.loads(completed.stdout)
    payload = json.loads(out_json.read_text())
    assert stdout == payload
    assert payload["passes_checkpoint"] is True
    assert "Passes checkpoint: `True`" in out_md.read_text()

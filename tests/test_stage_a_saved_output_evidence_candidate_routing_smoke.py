import json
import subprocess
import sys
from pathlib import Path

import post_training.run_stage_a_saved_output_evidence_candidate_routing_smoke as smoke
from post_training.export_stage_a_saved_output_evidence_candidate_routing_rows import (
    CANDIDATE_PAIRS,
)
from post_training.run_stage_a_saved_output_evidence_candidate_routing_smoke import (
    build_eval_report,
    candidate_outputs_from_row,
    load_json,
    load_jsonl,
    prompt_messages_from_row,
    score_candidate_outputs,
    target_output_from_row,
    validate_candidate_routing_rows,
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
    manifest = load_json(
        ROOT / "post_training" / "stage_a_saved_output_evidence_candidate_routing_manifest.json"
    )
    return rows, train_rows, heldout_rows, manifest


def test_evidence_candidate_routing_smoke_validates_tracked_rows() -> None:
    rows, train_rows, heldout_rows, manifest = tracked_artifacts()

    assert validate_candidate_routing_rows(rows, train_rows, heldout_rows, manifest) == []
    assert len(train_rows) == 20
    assert len(heldout_rows) == 5
    assert sum(1 for row in heldout_rows if row["bridge_focus_case"]) == 4
    assert not any(row["bridge_focus_case"] for row in train_rows)


def test_evidence_candidate_routing_smoke_prompt_is_label_isolated() -> None:
    rows, _, _, _ = tracked_artifacts()

    for row in rows:
        prompt_text = json.dumps(prompt_messages_from_row(row), sort_keys=True)
        target = target_output_from_row(row)
        assert target["selected_pair"] in row["model_visible_task"]["candidate_pairs"]
        assert "target_output" not in prompt_text
        assert "target_pair" not in prompt_text
        assert "hidden_eval_metadata" not in prompt_text
        assert str(row["source_task_id"]) not in prompt_text
        assert str(row["split_group"]) not in prompt_text
        assert str(row["case_family"]) not in prompt_text


def test_evidence_candidate_routing_smoke_candidate_space_matches_contract() -> None:
    rows, _, _, _ = tracked_artifacts()

    for row in rows:
        candidates = candidate_outputs_from_row(row)
        assert [candidate["selected_pair"] for candidate in candidates] == list(CANDIDATE_PAIRS)
        for candidate in candidates:
            assert set(candidate) == {"action", "evidence_status", "selected_pair"}
            assert candidate["selected_pair"] == (
                f"{candidate['action']}/{candidate['evidence_status']}"
            )


def test_evidence_candidate_routing_smoke_cli_dry_run_writes_report(tmp_path: Path) -> None:
    out_dir = tmp_path / "candidate_routing_smoke_dry"
    result = subprocess.run(
        [
            sys.executable,
            "post_training/run_stage_a_saved_output_evidence_candidate_routing_smoke.py",
            "--dry-run",
            "--out-dir",
            str(out_dir),
            "--run-id",
            "unit_candidate_routing_smoke_dry",
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
    assert report["bridge_focus_heldout_examples"] == 4
    assert report["candidate_space_size"] == 5
    assert report["issues"] == []
    assert report["ready_for_full_mode"] is True


def test_evidence_candidate_routing_smoke_requires_model_load_for_full_mode(
    tmp_path: Path,
) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "post_training/run_stage_a_saved_output_evidence_candidate_routing_smoke.py",
            "--out-dir",
            str(tmp_path / "blocked_full_run"),
            "--run-id",
            "blocked_full_run",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "--allow-model-load" in result.stderr


def test_evidence_candidate_routing_smoke_scores_finite_candidates(monkeypatch) -> None:
    _, _, heldout_rows, _ = tracked_artifacts()
    row = heldout_rows[1]
    target = target_output_from_row(row)

    def fake_score_candidate_target(*args, **kwargs) -> float:
        candidate = json.loads(args[3])
        return 1.0 if candidate == target else 0.0

    class FakeTokenizer:
        chat_template = None
        eos_token = "<eos>"

    monkeypatch.setattr(smoke, "score_candidate_target", fake_score_candidate_target)
    result = score_candidate_outputs(
        model=None,
        tokenizer=FakeTokenizer(),
        row=row,
        device="cpu",
        max_length=64,
    )

    assert result["prediction"] == target
    assert len(result["candidate_scores"]) == 5
    assert result["candidate_scores"][0]["candidate"] == target


def test_evidence_candidate_routing_smoke_eval_report_oracle_passes() -> None:
    _, _, heldout_rows, _ = tracked_artifacts()
    predictions = [
        {
            "id": f"oracle::{row['id']}",
            "source_candidate_routing_id": row["id"],
            "split": "heldout",
            "bridge_focus_case": row["bridge_focus_case"],
            "target_pair": row["target_pair"],
            "prediction": target_output_from_row(row),
        }
        for row in heldout_rows
    ]

    report = build_eval_report(
        expected_rows=heldout_rows,
        prediction_rows=predictions,
        split="heldout",
        run_id="oracle_candidate_routing_smoke",
    )

    assert report["summary"]["exact"] == 5
    assert report["summary"]["bridge_focus_exact"] == 4
    assert report["summary"]["violations"] == {}


def test_evidence_candidate_routing_smoke_eval_report_flags_static_prior() -> None:
    _, _, heldout_rows, _ = tracked_artifacts()
    predictions = [
        {
            "id": f"static::{row['id']}",
            "source_candidate_routing_id": row["id"],
            "split": "heldout",
            "bridge_focus_case": row["bridge_focus_case"],
            "target_pair": row["target_pair"],
            "prediction": {
                "action": "ground",
                "evidence_status": "supported",
                "selected_pair": "ground/supported",
            },
        }
        for row in heldout_rows
    ]

    report = build_eval_report(
        expected_rows=heldout_rows,
        prediction_rows=predictions,
        split="heldout",
        run_id="static_ground_candidate_routing_smoke",
    )

    assert report["summary"]["exact"] == 1
    assert report["summary"]["bridge_focus_exact"] == 0
    assert report["summary"]["violations"]["selected_pair_mismatch"] == 4


def test_evidence_candidate_routing_smoke_cayuga_template_calls_runner() -> None:
    text = (
        ROOT
        / "post_training"
        / "run_stage_a_saved_output_evidence_candidate_routing_smoke_cayuga.sbatch"
    ).read_text()

    assert "run_stage_a_saved_output_evidence_candidate_routing_smoke.py" in text
    assert "--allow-model-load" in text
    assert "stage_a_saved_output_evidence_candidate_routing_train_v1.jsonl" in text
    assert "stage_a_saved_output_evidence_candidate_routing_heldout_v1.jsonl" in text
    assert "post_training/runs/" in text
    assert "<allocation>" in text
    assert "<gpu-partition>" in text

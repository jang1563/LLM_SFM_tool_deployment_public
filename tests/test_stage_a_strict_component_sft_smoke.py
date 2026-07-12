import json
import subprocess
import sys
from pathlib import Path

import post_training.run_stage_a_strict_component_sft_smoke as component_sft
from post_training.run_stage_a_strict_component_sft_smoke import (
    build_component_eval_report,
    enum_action_candidate_outputs,
    enum_action_observed_pair_outputs,
    filter_component,
    load_jsonl,
    score_enum_action_candidates,
    validate_component_rows,
)
from post_training.export_stage_a_strict_component_targets import (
    ALLOWED_ACTIONS,
    ALLOWED_EVIDENCE_STATUSES,
)


ROOT = Path(__file__).resolve().parents[1]


def all_rows() -> list[dict]:
    return load_jsonl(ROOT / "post_training" / "stage_a_strict_component_targets_v1.jsonl")


def train_rows() -> list[dict]:
    return load_jsonl(ROOT / "post_training" / "stage_a_strict_component_targets_train_v1.jsonl")


def heldout_rows() -> list[dict]:
    return load_jsonl(ROOT / "post_training" / "stage_a_strict_component_targets_heldout_v1.jsonl")


def test_component_sft_smoke_validates_all_components() -> None:
    rows = all_rows()
    train = train_rows()
    heldout = heldout_rows()

    for component in ("enum_action", "tool_query", "routing_after_loop"):
        issues = validate_component_rows(
            rows,
            filter_component(train, component),
            filter_component(heldout, component),
            component=component,
        )
        assert issues == []


def test_component_sft_smoke_cli_dry_run_writes_report(tmp_path: Path) -> None:
    out_dir = tmp_path / "component_sft_dry_run"
    result = subprocess.run(
        [
            sys.executable,
            "post_training/run_stage_a_strict_component_sft_smoke.py",
            "--dry-run",
            "--component",
            "enum_action",
            "--out-dir",
            str(out_dir),
            "--run-id",
            "unit_component_sft_dry_run",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((out_dir / "report.json").read_text())
    assert report["dry_run"] is True
    assert report["component"] == "enum_action"
    assert report["decode_mode"] == "freeform"
    assert report["train_examples"] == 20
    assert report["heldout_examples"] == 5
    assert report["issues"] == []


def test_component_sft_smoke_requires_explicit_model_load_for_full_mode(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "post_training/run_stage_a_strict_component_sft_smoke.py",
            "--component",
            "tool_query",
            "--out-dir",
            str(tmp_path / "component_sft_blocked_full_run"),
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


def test_enum_action_candidate_outputs_are_valid_contract() -> None:
    candidates = enum_action_candidate_outputs()

    assert len(candidates) == len(ALLOWED_ACTIONS) * len(ALLOWED_EVIDENCE_STATUSES)
    assert {"action": "verify", "evidence_status": "supported"} in candidates
    for candidate in candidates:
        assert set(candidate) == {"action", "evidence_status"}
        assert candidate["action"] in ALLOWED_ACTIONS
        assert candidate["evidence_status"] in ALLOWED_EVIDENCE_STATUSES


def test_enum_action_observed_pair_outputs_use_train_targets_only() -> None:
    candidates = enum_action_observed_pair_outputs(filter_component(train_rows(), "enum_action"))

    assert candidates == [
        {"action": "ground", "evidence_status": "supported"},
        {"action": "reject", "evidence_status": "contradicted"},
        {"action": "defer", "evidence_status": "insufficient"},
        {"action": "verify", "evidence_status": "insufficient"},
        {"action": "flag", "evidence_status": "invalid_value"},
    ]


def test_enum_candidate_scoring_keeps_full_ranked_scores(monkeypatch) -> None:
    preferred = {"action": "defer", "evidence_status": "insufficient"}

    def fake_score_candidate_target(*args, **kwargs) -> float:
        candidate = json.loads(args[3])
        return 1.0 if candidate == preferred else 0.0

    monkeypatch.setattr(component_sft, "score_candidate_target", fake_score_candidate_target)
    result = score_enum_action_candidates(
        model=None,
        tokenizer=None,
        prompt="prompt",
        device="cpu",
        max_length=32,
    )

    assert result["prediction"] == preferred
    assert len(result["candidate_scores"]) == len(enum_action_candidate_outputs())
    assert result["candidate_scores"][0]["candidate"] == preferred


def test_enum_candidate_decode_mode_is_enum_only(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "post_training/run_stage_a_strict_component_sft_smoke.py",
            "--dry-run",
            "--component",
            "tool_query",
            "--decode-mode",
            "enum_candidate_score",
            "--out-dir",
            str(tmp_path / "bad_decode_mode"),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "enum_candidate_score/enum_observed_pair_score is only valid" in result.stderr


def test_routing_candidate_decode_mode_is_routing_only(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "post_training/run_stage_a_strict_component_sft_smoke.py",
            "--dry-run",
            "--component",
            "enum_action",
            "--decode-mode",
            "routing_observed_pair_score",
            "--out-dir",
            str(tmp_path / "bad_routing_decode_mode"),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "routing_observed_pair_score is only valid" in result.stderr


def test_enum_observed_pair_decode_mode_dry_run_reports_candidate_space(tmp_path: Path) -> None:
    out_dir = tmp_path / "observed_pair_dry_run"
    result = subprocess.run(
        [
            sys.executable,
            "post_training/run_stage_a_strict_component_sft_smoke.py",
            "--dry-run",
            "--component",
            "enum_action",
            "--decode-mode",
            "enum_observed_pair_score",
            "--out-dir",
            str(out_dir),
            "--run-id",
            "unit_observed_pair_dry_run",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((out_dir / "report.json").read_text())
    assert report["decode_mode"] == "enum_observed_pair_score"
    assert report["candidate_space_size"] == 5
    assert report["candidate_outputs"] == enum_action_observed_pair_outputs(
        filter_component(train_rows(), "enum_action")
    )


def test_component_sft_smoke_oracle_predictions_pass() -> None:
    expected = filter_component(heldout_rows(), "routing_after_loop")
    predictions = [
        {
            "id": f"oracle::{row['id']}",
            "source_component_target_id": row["id"],
            "split": row["split"],
            "prediction": row["target_output"],
        }
        for row in expected
    ]

    report = build_component_eval_report(
        expected_rows=expected,
        prediction_rows=predictions,
        component="routing_after_loop",
        run_id="oracle_component_smoke",
    )

    assert report["summary"]["passed"] == 5
    assert report["summary"]["mean_score"] == 1.0
    assert report["summary"]["violations"] == {}


def test_component_sft_smoke_flags_bad_component_outputs() -> None:
    enum_expected = filter_component(heldout_rows(), "enum_action")
    enum_predictions = [
        {
            "source_component_target_id": row["id"],
            "split": row["split"],
            "prediction": {"action": "ground", "evidence_status": "verified"},
        }
        for row in enum_expected
    ]
    enum_report = build_component_eval_report(
        expected_rows=enum_expected,
        prediction_rows=enum_predictions,
        component="enum_action",
        run_id="bad_enum_component_smoke",
    )

    assert enum_report["summary"]["passed"] == 0
    assert enum_report["summary"]["violations"]["enum_value_invalid"] == 5
    assert enum_report["summary"]["violations"]["target_mismatch"] == 5

    tool_expected = filter_component(heldout_rows(), "tool_query")
    tool_predictions = [
        {
            "source_component_target_id": row["id"],
            "split": row["split"],
            "prediction": {"tool_calls": ["nullatlas_survey_prior_failures"]},
        }
        for row in tool_expected
    ]
    tool_report = build_component_eval_report(
        expected_rows=tool_expected,
        prediction_rows=tool_predictions,
        component="tool_query",
        run_id="bad_tool_component_smoke",
    )

    assert tool_report["summary"]["passed"] == 0
    assert tool_report["summary"]["violations"]["tool_query_shape_invalid"] == 5
    assert tool_report["summary"]["violations"]["target_mismatch"] == 5


def test_component_sft_cluster_templates_call_component_runner() -> None:
    for rel in (
        "post_training/run_stage_a_strict_component_sft_cayuga.sbatch",
        "post_training/run_stage_a_strict_component_sft_expanse.sbatch",
    ):
        text = (ROOT / rel).read_text()
        assert "run_stage_a_strict_component_sft_smoke.py" in text
        assert "--allow-model-load" in text
        assert "--decode-mode" in text
        assert "DECODE_MODE" in text
        assert "enum_observed_pair_score" in text
        assert "routing_observed_pair_score" in text
        assert "COMPONENT" in text
        assert "stage_a_strict_component_targets_train_v1.jsonl" in text
        assert "stage_a_strict_component_targets_heldout_v1.jsonl" in text
        assert "post_training/runs/" in text
        assert "<allocation>" in text

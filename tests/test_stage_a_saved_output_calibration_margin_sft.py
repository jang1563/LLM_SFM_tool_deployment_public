import json
import subprocess
import sys
from pathlib import Path

import pytest

from post_training.run_stage_a_saved_output_calibration_margin_sft import (
    build_margin_delta_report,
    build_margin_report,
    candidate_ce_loss_from_logps,
    candidate_field_ce_loss_from_logps,
    candidate_field_values,
    candidate_index_for_row,
    candidate_output_for_pair,
    candidate_pairs_for_policy,
    expand_training_rows,
    load_jsonl,
    load_manifest,
    margin_row_for_pair,
    output_text,
    pairwise_margin_loss_from_logps,
    parse_focus_chosen_pairs,
    parse_score_target_formats,
    project_output,
    summarize_margin_delta_rows,
    summarize_margin_rows,
    target_format_output_path,
    validate_saved_output_calibration_margin_artifacts,
)


ROOT = Path(__file__).resolve().parents[1]


def all_pairs() -> list[dict]:
    return load_jsonl(ROOT / "post_training" / "stage_a_saved_output_calibration_probe_v1.jsonl")


def train_pairs() -> list[dict]:
    return load_jsonl(ROOT / "post_training" / "stage_a_saved_output_calibration_probe_train_v1.jsonl")


def heldout_pairs() -> list[dict]:
    return load_jsonl(ROOT / "post_training" / "stage_a_saved_output_calibration_probe_heldout_v1.jsonl")


def probe_manifest() -> dict:
    return load_manifest(ROOT / "post_training" / "stage_a_saved_output_calibration_probe_manifest.json")


def test_saved_output_calibration_margin_sft_validates_tracked_artifacts() -> None:
    issues = validate_saved_output_calibration_margin_artifacts(
        all_pairs(),
        train_pairs(),
        heldout_pairs(),
        probe_manifest(),
    )

    assert issues == []


def test_saved_output_calibration_margin_sft_cli_dry_run_is_split_safe(tmp_path: Path) -> None:
    out_dir = tmp_path / "saved_output_calibration_margin_sft_dry_run"
    result = subprocess.run(
        [
            sys.executable,
            "post_training/run_stage_a_saved_output_calibration_margin_sft.py",
            "--dry-run",
            "--out-dir",
            str(out_dir),
            "--run-id",
            "unit_saved_output_calibration_margin_sft_dry_run",
            "--pairwise-margin-weight",
            "1",
            "--pairwise-margin",
            "0.05",
            "--score-base-margins",
            "--score-train-margins",
            "--score-base-candidates",
            "--score-train-candidates",
            "--score-trained-candidates",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((out_dir / "report.json").read_text())
    assert report["dry_run"] is True
    assert report["failure_mode"] == "saved_output_ground_supported_collapse"
    assert report["calibration_axis"] == "target_pair_vs_ground_supported"
    assert report["train_examples"] == 16
    assert report["training_examples"] == 16
    assert report["heldout_examples"] == 4
    assert report["heldout_used_for_training"] is False
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
    assert set(report["train_case_ids"]).isdisjoint(report["heldout_case_ids"])
    assert report["pairwise_margin_weight"] == 1.0
    assert report["pairwise_margin"] == 0.05
    assert report["candidate_ce_weight"] == 0.0
    assert report["candidate_ce_mode"] == "pair"
    assert report["candidate_ce_logprob_mode"] == "mean"
    assert report["target_format"] == "full"
    assert report["score_base_margins"] is True
    assert report["score_train_margins"] is True
    assert report["score_base_candidates"] is True
    assert report["score_train_candidates"] is True
    assert report["score_trained_candidates"] is True
    assert report["candidate_policy"] == "train_observed_plus_rejected"
    assert report["candidate_target_format"] == "full"
    assert report["candidate_space_size"] == 5
    assert report["artifact_policy"]["model_state_committed"] is False
    assert report["artifact_policy"]["candidate_jsonl_committed"] is False
    assert report["issues"] == []


def test_saved_output_calibration_margin_sft_focus_expansion() -> None:
    focus = parse_focus_chosen_pairs("flag/invalid_value,defer/insufficient")
    rows = expand_training_rows(
        train_pairs(),
        focus_chosen_pairs=focus,
        focus_repeat=3,
        focus_only=False,
    )

    counts: dict[str, int] = {}
    for row in rows:
        counts[row["chosen_pair"]] = counts.get(row["chosen_pair"], 0) + 1

    assert focus == ("flag/invalid_value", "defer/insufficient")
    assert len(rows) == 32
    assert counts == {
        "defer/insufficient": 12,
        "flag/invalid_value": 12,
        "reject/contradicted": 4,
        "verify/insufficient": 4,
    }


def test_saved_output_calibration_margin_sft_target_format_projection() -> None:
    flag_row = [row for row in heldout_pairs() if row["chosen_pair"] == "flag/invalid_value"][0]
    output = flag_row["chosen_output"]

    assert project_output(output, target_format="action_only") == {"action": "flag"}
    assert project_output(output, target_format="status_only") == {
        "evidence_status": "invalid_value",
    }
    assert project_output(output, target_format="action_status_only") == {
        "action": "flag",
        "evidence_status": "invalid_value",
    }
    assert "tool_calls" in json.loads(output_text(output, target_format="full"))
    assert json.loads(output_text(output, target_format="action_only")) == {"action": "flag"}


def test_saved_output_calibration_candidate_policy_includes_collapse_boundary() -> None:
    pairs = candidate_pairs_for_policy(train_pairs(), policy="train_observed_plus_rejected")

    assert pairs == [
        {"action": "defer", "evidence_status": "insufficient"},
        {"action": "flag", "evidence_status": "invalid_value"},
        {"action": "ground", "evidence_status": "supported"},
        {"action": "reject", "evidence_status": "contradicted"},
        {"action": "verify", "evidence_status": "insufficient"},
    ]

    heldout_flag = [row for row in heldout_pairs() if row["chosen_pair"] == "flag/invalid_value"][0]
    ground_candidate = candidate_output_for_pair(
        heldout_flag,
        {"action": "ground", "evidence_status": "supported"},
    )
    flag_candidate = candidate_output_for_pair(
        heldout_flag,
        {"action": "flag", "evidence_status": "invalid_value"},
    )

    assert ground_candidate["action"] == "ground"
    assert ground_candidate["evidence_status"] == "supported"
    assert flag_candidate["action"] == "flag"
    assert flag_candidate["evidence_status"] == "invalid_value"
    assert "tool_calls" in ground_candidate
    assert "tool_calls" in flag_candidate


def test_saved_output_calibration_margin_sft_flag_action_only_dry_run(tmp_path: Path) -> None:
    out_dir = tmp_path / "saved_output_calibration_margin_sft_flag_action_only"
    result = subprocess.run(
        [
            sys.executable,
            "post_training/run_stage_a_saved_output_calibration_margin_sft.py",
            "--dry-run",
            "--out-dir",
            str(out_dir),
            "--run-id",
            "unit_saved_output_calibration_margin_sft_flag_action_only",
            "--focus-chosen-pairs",
            "flag/invalid_value",
            "--focus-repeat",
            "4",
            "--focus-only",
            "--target-format",
            "action_only",
            "--score-target-formats",
            "action_status_only:status_only",
            "--pairwise-margin-weight",
            "1",
            "--pairwise-margin",
            "0.05",
            "--score-base-margins",
            "--score-train-margins",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((out_dir / "report.json").read_text())
    assert report["target_format"] == "action_only"
    assert report["score_target_formats"] == [
        "action_only",
        "action_status_only",
        "status_only",
    ]
    assert report["focus_chosen_pairs"] == ["flag/invalid_value"]
    assert report["focus_only"] is True
    assert report["train_examples"] == 16
    assert report["training_examples"] == 16
    assert report["training_by_chosen_pair"] == {"flag/invalid_value": 16}
    assert report["heldout_by_chosen_pair"]["flag/invalid_value"] == 1
    assert report["heldout_used_for_training"] is False
    assert set(report["train_case_ids"]).isdisjoint(report["heldout_case_ids"])
    assert report["issues"] == []


def test_saved_output_calibration_margin_sft_score_target_format_parser() -> None:
    assert parse_score_target_formats(
        "action_status_only:status_only",
        training_target_format="full",
    ) == ("full", "action_status_only", "status_only")
    assert parse_score_target_formats("", training_target_format="action_only") == ("action_only",)
    with pytest.raises(ValueError, match="unknown score_target_formats"):
        parse_score_target_formats("bad_mode", training_target_format="full")


def test_saved_output_candidate_ce_dry_run_records_candidate_routing_objective(
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "saved_output_candidate_ce_dry_run"
    result = subprocess.run(
        [
            sys.executable,
            "post_training/run_stage_a_saved_output_calibration_margin_sft.py",
            "--dry-run",
            "--out-dir",
            str(out_dir),
            "--run-id",
            "unit_saved_output_candidate_ce_dry_run",
            "--candidate-ce-weight",
            "1",
            "--candidate-ce-mode",
            "pair_plus_field",
            "--candidate-ce-logprob-mode",
            "mean",
            "--candidate-policy",
            "train_observed_plus_rejected",
            "--candidate-target-format",
            "action_status_only",
            "--score-base-candidates",
            "--score-train-candidates",
            "--score-trained-candidates",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((out_dir / "report.json").read_text())
    assert report["candidate_ce_weight"] == 1.0
    assert report["candidate_ce_mode"] == "pair_plus_field"
    assert report["candidate_ce_logprob_mode"] == "mean"
    assert report["candidate_policy"] == "train_observed_plus_rejected"
    assert report["candidate_target_format"] == "action_status_only"
    assert report["candidate_space_size"] == 5
    assert report["heldout_used_for_training"] is False
    assert report["artifact_policy"]["candidate_jsonl_committed"] is False
    assert report["issues"] == []


def test_saved_output_candidate_index_uses_target_pair() -> None:
    pairs = candidate_pairs_for_policy(train_pairs(), policy="train_observed_plus_rejected")
    verify_row = [row for row in train_pairs() if row["chosen_pair"] == "verify/insufficient"][0]

    assert candidate_index_for_row(verify_row, pairs) == pairs.index(
        {"action": "verify", "evidence_status": "insufficient"}
    )


def test_saved_output_candidate_ce_loss_prefers_expected_high_logp() -> None:
    torch = pytest.importorskip("torch")

    good = candidate_ce_loss_from_logps(torch.tensor([3.0, 1.0, 0.0]), 0)
    bad = candidate_ce_loss_from_logps(torch.tensor([0.0, 3.0, 1.0]), 0)

    assert float(good) < float(bad)


def test_saved_output_candidate_field_ce_loss_prefers_expected_fields() -> None:
    torch = pytest.importorskip("torch")
    candidates = [
        {"action": "reject", "evidence_status": "contradicted"},
        {"action": "defer", "evidence_status": "insufficient"},
        {"action": "reject", "evidence_status": "insufficient"},
    ]
    expected = {"action": "defer", "evidence_status": "insufficient"}

    good = candidate_field_ce_loss_from_logps(
        torch.tensor([0.0, 3.0, 1.0]),
        candidates,
        expected,
    )
    bad = candidate_field_ce_loss_from_logps(
        torch.tensor([3.0, 0.0, 1.0]),
        candidates,
        expected,
    )

    assert candidate_field_values(candidates, "action") == ["reject", "defer"]
    assert float(good) < float(bad)


def test_saved_output_candidate_ce_negative_weight_fails_closed(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "post_training/run_stage_a_saved_output_calibration_margin_sft.py",
            "--dry-run",
            "--out-dir",
            str(tmp_path / "bad_candidate_ce_weight"),
            "--candidate-ce-weight",
            "-1",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "candidate_ce_weight_negative" in result.stderr


def test_saved_output_calibration_margin_sft_target_format_output_paths() -> None:
    base = Path("post_training/runs/unit/margin_report.json")
    assert target_format_output_path(base, "full", "full") == base
    assert target_format_output_path(base, "action_only", "full") == Path(
        "post_training/runs/unit/margin_report_action_only.json"
    )
    jsonl = Path("post_training/runs/unit/margins.jsonl")
    assert target_format_output_path(jsonl, "status_only", "full") == Path(
        "post_training/runs/unit/margins_status_only.jsonl"
    )


def test_saved_output_calibration_margin_sft_unknown_focus_pair_fails() -> None:
    with pytest.raises(ValueError, match="unknown focus_chosen_pairs"):
        expand_training_rows(
            train_pairs(),
            focus_chosen_pairs=("ground/supported",),
            focus_repeat=2,
            focus_only=False,
        )


def test_saved_output_calibration_margin_summary_tracks_wins_and_failures() -> None:
    rows = [
        margin_row_for_pair(
            heldout_pairs()[0],
            run_id="unit",
            model_id="dummy",
            chosen_score=-1.0,
            rejected_score=-2.0,
            score_label="trained_heldout",
        ),
        margin_row_for_pair(
            heldout_pairs()[1],
            run_id="unit",
            model_id="dummy",
            chosen_score=-3.0,
            rejected_score=-2.0,
            score_label="trained_heldout",
        ),
    ]
    report = build_margin_report(
        run_id="unit",
        model_id="dummy",
        rows=rows,
        score_label="trained_heldout",
    )

    assert summarize_margin_rows(rows)["margin_wins"] == 1
    assert report["summary"]["mean_margin"] == 0.0
    assert report["summary"]["violations"] == {"chosen_not_above_rejected": 1}
    assert report["target_format"] == "full"
    assert "not DPO/RLVR" in report["boundary"]


def test_saved_output_calibration_margin_delta_report_tracks_movement() -> None:
    base_rows = [
        margin_row_for_pair(
            heldout_pairs()[0],
            run_id="unit",
            model_id="dummy",
            chosen_score=-2.0,
            rejected_score=-1.0,
            score_label="base_heldout",
        ),
        margin_row_for_pair(
            heldout_pairs()[1],
            run_id="unit",
            model_id="dummy",
            chosen_score=-1.0,
            rejected_score=-2.0,
            score_label="base_heldout",
        ),
    ]
    trained_rows = [
        margin_row_for_pair(
            heldout_pairs()[0],
            run_id="unit",
            model_id="dummy",
            chosen_score=-0.5,
            rejected_score=-1.0,
            score_label="trained_heldout",
        ),
        margin_row_for_pair(
            heldout_pairs()[1],
            run_id="unit",
            model_id="dummy",
            chosen_score=-3.0,
            rejected_score=-2.0,
            score_label="trained_heldout",
        ),
    ]

    report = build_margin_delta_report(
        run_id="unit",
        model_id="dummy",
        base_rows=base_rows,
        trained_rows=trained_rows,
    )

    assert summarize_margin_delta_rows(report["rows"])["outcomes"] == {
        "newly_lost": 1,
        "newly_won": 1,
    }
    assert report["summary"]["base_margin_wins"] == 1
    assert report["summary"]["trained_margin_wins"] == 1
    assert "not DPO/RLVR" in report["boundary"]


def test_saved_output_calibration_margin_sft_requires_explicit_model_load() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "post_training/run_stage_a_saved_output_calibration_margin_sft.py",
            "--out-dir",
            "/tmp/stage_a_probe_margin_sft_should_not_run",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "--allow-model-load" in result.stderr


def test_saved_output_calibration_margin_loss_rejects_negative_margin() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        pairwise_margin_loss_from_logps(None, None, margin=-0.1)


def test_saved_output_calibration_margin_cluster_templates_keep_outputs_ignored() -> None:
    for name in (
        "run_stage_a_saved_output_calibration_margin_sft_cayuga.sbatch",
        "run_stage_a_saved_output_calibration_margin_sft_expanse.sbatch",
    ):
        text = (ROOT / "post_training" / name).read_text()
        assert "run_stage_a_saved_output_calibration_margin_sft.py" in text
        assert "--allow-model-load" in text
        assert "post_training/runs/${RUN_ID}" in text
        assert "<allocation>" in text
        assert "scheduler logs" in text
        assert "should not be committed" in text
        assert "--score-base-margins" in text
        assert "--score-train-margins" in text
        assert "TARGET_FORMAT" in text
        assert "--target-format" in text
        assert "SCORE_TARGET_FORMATS" in text
        assert "--score-target-formats" in text
        assert "CANDIDATE_CE_WEIGHT" in text
        assert "--candidate-ce-weight" in text
        assert "CANDIDATE_CE_MODE" in text
        assert "--candidate-ce-mode" in text
        assert "SCORE_TRAINED_CANDIDATES" in text
        assert "SCORE_TRAIN_CANDIDATES" in text
        assert "--score-trained-candidates" in text
        assert "--score-train-candidates" in text
        assert "CANDIDATE_POLICY" in text
        assert "--candidate-policy" in text

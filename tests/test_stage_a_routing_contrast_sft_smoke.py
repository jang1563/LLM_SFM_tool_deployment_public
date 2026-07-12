import json
import subprocess
import sys
from pathlib import Path

import pytest

from post_training.run_stage_a_routing_contrast_sft_smoke import (
    build_candidate_rank_report,
    build_margin_delta_report,
    build_margin_report,
    candidate_rank_row,
    expand_training_rows,
    load_jsonl,
    load_manifest,
    margin_row_for_pair,
    pairwise_margin_loss_from_logps,
    parse_focus_chosen_pairs,
    routing_action_status_candidates_from_pairs,
    summarize_margin_delta_rows,
    summarize_candidate_rank_rows,
    summarize_margin_rows,
    validate_routing_contrast_artifacts,
)


ROOT = Path(__file__).resolve().parents[1]


def all_pairs() -> list[dict]:
    return load_jsonl(ROOT / "post_training" / "stage_a_routing_action_status_contrast_pairs_v1.jsonl")


def train_pairs() -> list[dict]:
    return load_jsonl(ROOT / "post_training" / "stage_a_routing_action_status_contrast_pairs_train_v1.jsonl")


def heldout_pairs() -> list[dict]:
    return load_jsonl(ROOT / "post_training" / "stage_a_routing_action_status_contrast_pairs_heldout_v1.jsonl")


def contrast_manifest() -> dict:
    return load_manifest(ROOT / "post_training" / "stage_a_routing_action_status_contrast_pairs_manifest.json")


def test_routing_contrast_sft_smoke_validates_tracked_artifacts() -> None:
    issues = validate_routing_contrast_artifacts(
        all_pairs(),
        train_pairs(),
        heldout_pairs(),
        contrast_manifest(),
    )

    assert issues == []


def test_routing_contrast_sft_smoke_cli_dry_run_writes_report(tmp_path: Path) -> None:
    out_dir = tmp_path / "routing_contrast_sft_dry_run"
    result = subprocess.run(
        [
            sys.executable,
            "post_training/run_stage_a_routing_contrast_sft_smoke.py",
            "--dry-run",
            "--out-dir",
            str(out_dir),
            "--run-id",
            "unit_routing_contrast_sft_dry_run",
            "--pairwise-margin-weight",
            "1",
            "--pairwise-margin",
            "0.05",
            "--score-base-margins",
            "--score-train-margins",
            "--score-base-routing-candidates",
            "--score-trained-routing-candidates",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((out_dir / "report.json").read_text())
    assert report["dry_run"] is True
    assert report["component"] == "routing_after_loop"
    assert report["failure_mode"] == "routing_action_status_confusion"
    assert report["train_examples"] == 12
    assert report["heldout_examples"] == 3
    assert report["heldout_by_chosen_pair"] == {
        "defer/insufficient": 1,
        "flag/invalid_value": 1,
        "verify/insufficient": 1,
    }
    assert report["pairwise_margin_weight"] == 1.0
    assert report["pairwise_margin"] == 0.05
    assert report["score_base_margins"] is True
    assert report["score_train_margins"] is True
    assert report["score_base_routing_candidates"] is True
    assert report["score_trained_routing_candidates"] is True
    assert report["routing_candidate_space_size"] == 5
    assert report["routing_candidate_outputs"] == [
        {"action": "defer", "evidence_status": "insufficient"},
        {"action": "reject", "evidence_status": "contradicted"},
        {"action": "verify", "evidence_status": "insufficient"},
        {"action": "flag", "evidence_status": "invalid_value"},
        {"action": "ground", "evidence_status": "supported"},
    ]
    assert report["issues"] == []


def test_routing_contrast_sft_smoke_cli_accepts_defer_verify_boundary(tmp_path: Path) -> None:
    out_dir = tmp_path / "routing_defer_verify_dry_run"
    result = subprocess.run(
        [
            sys.executable,
            "post_training/run_stage_a_routing_contrast_sft_smoke.py",
            "--dry-run",
            "--pairs",
            "post_training/stage_a_routing_defer_verify_contrast_pairs_v1.jsonl",
            "--train-pairs",
            "post_training/stage_a_routing_defer_verify_contrast_pairs_train_v1.jsonl",
            "--heldout-pairs",
            "post_training/stage_a_routing_defer_verify_contrast_pairs_heldout_v1.jsonl",
            "--manifest",
            "post_training/stage_a_routing_defer_verify_contrast_pairs_manifest.json",
            "--out-dir",
            str(out_dir),
            "--run-id",
            "unit_routing_defer_verify_dry_run",
            "--pairwise-margin-weight",
            "1",
            "--pairwise-margin",
            "0.05",
            "--score-base-routing-candidates",
            "--score-trained-routing-candidates",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((out_dir / "report.json").read_text())
    assert report["failure_mode"] == "routing_defer_verify_boundary_confusion"
    assert report["contrast_axes"] == {"defer_verify_boundary": 10}
    assert report["train_examples"] == 8
    assert report["heldout_examples"] == 2
    assert report["routing_candidate_space_size"] == 2
    assert report["routing_candidate_outputs"] == [
        {"action": "defer", "evidence_status": "insufficient"},
        {"action": "verify", "evidence_status": "insufficient"},
    ]
    assert report["issues"] == []


def test_routing_contrast_sft_smoke_focus_expansion() -> None:
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
    assert len(rows) == 28
    assert counts == {
        "defer/insufficient": 12,
        "flag/invalid_value": 12,
        "verify/insufficient": 4,
    }


def test_routing_contrast_sft_smoke_unknown_focus_pair_fails() -> None:
    with pytest.raises(ValueError, match="unknown focus_chosen_pairs"):
        expand_training_rows(
            train_pairs(),
            focus_chosen_pairs=("ground/supported",),
            focus_repeat=2,
            focus_only=False,
        )


def test_routing_contrast_margin_summary_tracks_wins_and_failures() -> None:
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
    assert "not DPO/RLVR" in report["boundary"]


def test_routing_contrast_margin_delta_report_tracks_movement() -> None:
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
            chosen_score=-2.0,
            rejected_score=-1.0,
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


def test_routing_contrast_candidate_rank_report_tracks_top1() -> None:
    row = heldout_pairs()[0]
    target = row["chosen_output"]
    wrong = row["rejected_output"]
    prediction_row = {
        "source_routing_contrast_pair_id": row["id"],
        "case_id": "stage_a::unit",
        "case_family": row["case_family"],
        "chosen_pair": row["chosen_pair"],
        "rejected_pair": row["rejected_pair"],
        "candidate_space_size": 2,
        "target_output": target,
        "candidate_scores": [
            {"candidate": wrong, "score": 2.0},
            {"candidate": target, "score": 1.25},
        ],
    }

    rank_row = candidate_rank_row(prediction_row)
    summary = summarize_candidate_rank_rows([rank_row])
    report = build_candidate_rank_report(
        run_id="unit",
        model_id="dummy",
        rows=[prediction_row],
        score_label="unit_candidates",
        action_status_candidates=routing_action_status_candidates_from_pairs(train_pairs())[:2],
    )

    assert rank_row["exact_top1"] is False
    assert rank_row["action_status_top1"] is False
    assert rank_row["gold_rank"] == 2
    assert rank_row["top_gold_margin"] == 0.75
    assert summary["exact_top1"] == 0
    assert summary["mean_gold_rank_observed"] == 2.0
    assert report["summary"]["top_pair_counts"] == {"reject/contradicted": 1}
    assert "not free-form generation" in report["boundary"]


def test_routing_contrast_pairwise_margin_loss_rejects_negative_margin() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        pairwise_margin_loss_from_logps(None, None, margin=-0.1)


def test_routing_contrast_cluster_templates_call_margin_runner() -> None:
    for rel in (
        "post_training/run_stage_a_routing_contrast_sft_cayuga.sbatch",
        "post_training/run_stage_a_routing_contrast_sft_expanse.sbatch",
    ):
        text = (ROOT / rel).read_text()
        assert "run_stage_a_routing_contrast_sft_smoke.py" in text
        assert "--allow-model-load" in text
        assert "stage_a_routing_action_status_contrast_pairs_train_v1.jsonl" in text
        assert "stage_a_routing_action_status_contrast_pairs_heldout_v1.jsonl" in text
        assert "post_training/runs/" in text
        assert "SCORE_BASE_MARGINS" in text
        assert "--score-base-margins" in text
        assert "SCORE_TRAIN_MARGINS" in text
        assert "--score-train-margins" in text
        assert "FOCUS_CHOSEN_PAIRS" in text
        assert "--focus-chosen-pairs" in text
        assert "PAIRWISE_MARGIN_WEIGHT" in text
        assert "--pairwise-margin-weight" in text
        assert "PAIRWISE_MARGIN" in text
        assert "--pairwise-margin" in text
        assert "MARGIN_LOGPROB_MODE" in text
        assert "--margin-logprob-mode" in text
        assert "SCORE_BASE_ROUTING_CANDIDATES" in text
        assert "--score-base-routing-candidates" in text
        assert "SCORE_TRAINED_ROUTING_CANDIDATES" in text
        assert "--score-trained-routing-candidates" in text
        assert "margins.jsonl" in text
        assert "routing_candidate_report.json" in text
        assert "<allocation>" in text

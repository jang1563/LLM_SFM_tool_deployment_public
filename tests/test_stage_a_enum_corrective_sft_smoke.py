import json
import subprocess
import sys
from pathlib import Path

import pytest

from post_training.run_stage_a_enum_corrective_sft_smoke import (
    build_candidate_selection_report,
    build_margin_delta_report,
    build_margin_report,
    candidate_ce_loss_from_logps,
    candidate_field_ce_loss_from_logps,
    candidate_field_values,
    candidate_index_for_row,
    candidate_selection_row_for_pair,
    enum_candidate_outputs_for_policy,
    expand_training_rows,
    load_jsonl,
    load_manifest,
    margin_row_for_pair,
    pairwise_margin_loss_from_logps,
    parse_focus_chosen_pairs,
    summarize_candidate_selection_rows,
    summarize_margin_delta_rows,
    summarize_margin_rows,
    validate_corrective_artifacts,
)


ROOT = Path(__file__).resolve().parents[1]


def all_pairs() -> list[dict]:
    return load_jsonl(ROOT / "post_training" / "stage_a_enum_corrective_pairs_v1.jsonl")


def train_pairs() -> list[dict]:
    return load_jsonl(ROOT / "post_training" / "stage_a_enum_corrective_pairs_train_v1.jsonl")


def heldout_pairs() -> list[dict]:
    return load_jsonl(ROOT / "post_training" / "stage_a_enum_corrective_pairs_heldout_v1.jsonl")


def corrective_manifest() -> dict:
    return load_manifest(ROOT / "post_training" / "stage_a_enum_corrective_pairs_manifest.json")


def action_contrast_pairs() -> list[dict]:
    return load_jsonl(ROOT / "post_training" / "stage_a_enum_action_contrast_pairs_v1.jsonl")


def action_contrast_train_pairs() -> list[dict]:
    return load_jsonl(ROOT / "post_training" / "stage_a_enum_action_contrast_pairs_train_v1.jsonl")


def action_contrast_heldout_pairs() -> list[dict]:
    return load_jsonl(ROOT / "post_training" / "stage_a_enum_action_contrast_pairs_heldout_v1.jsonl")


def action_contrast_manifest() -> dict:
    return load_manifest(ROOT / "post_training" / "stage_a_enum_action_contrast_pairs_manifest.json")


def test_enum_corrective_sft_smoke_validates_tracked_artifacts() -> None:
    issues = validate_corrective_artifacts(
        all_pairs(),
        train_pairs(),
        heldout_pairs(),
        corrective_manifest(),
    )

    assert issues == []


def test_enum_corrective_sft_smoke_cli_dry_run_writes_report(tmp_path: Path) -> None:
    out_dir = tmp_path / "enum_corrective_sft_dry_run"
    result = subprocess.run(
        [
            sys.executable,
            "post_training/run_stage_a_enum_corrective_sft_smoke.py",
            "--dry-run",
            "--out-dir",
            str(out_dir),
            "--run-id",
            "unit_enum_corrective_sft_dry_run",
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
    assert report["failure_mode"] == "ground_supported_collapse"
    assert report["train_examples"] == 16
    assert report["heldout_examples"] == 4
    assert report["heldout_by_chosen_pair"] == {
        "defer/insufficient": 1,
        "flag/invalid_value": 1,
        "reject/contradicted": 1,
        "verify/insufficient": 1,
    }
    assert report["issues"] == []


def test_enum_corrective_sft_smoke_accepts_action_contrast_artifacts() -> None:
    issues = validate_corrective_artifacts(
        action_contrast_pairs(),
        action_contrast_train_pairs(),
        action_contrast_heldout_pairs(),
        action_contrast_manifest(),
    )

    assert issues == []


def test_enum_corrective_sft_smoke_cli_dry_run_reports_action_contrast(tmp_path: Path) -> None:
    out_dir = tmp_path / "enum_action_contrast_dry_run"
    result = subprocess.run(
        [
            sys.executable,
            "post_training/run_stage_a_enum_corrective_sft_smoke.py",
            "--dry-run",
            "--pairs",
            "post_training/stage_a_enum_action_contrast_pairs_v1.jsonl",
            "--train-pairs",
            "post_training/stage_a_enum_action_contrast_pairs_train_v1.jsonl",
            "--heldout-pairs",
            "post_training/stage_a_enum_action_contrast_pairs_heldout_v1.jsonl",
            "--manifest",
            "post_training/stage_a_enum_action_contrast_pairs_manifest.json",
            "--out-dir",
            str(out_dir),
            "--run-id",
            "unit_enum_action_contrast_dry_run",
            "--focus-chosen-pairs",
            "flag/invalid_value",
            "--focus-repeat",
            "4",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((out_dir / "report.json").read_text())
    assert report["failure_mode"] == "same_status_wrong_action_contrast"
    assert report["failure_modes"] == {"same_status_wrong_action_contrast": 20}
    assert report["contrast_axes"] == {"action": 20}
    assert report["candidate_policies"] == {"same_status_action_contrast": 20}
    assert report["focus_chosen_pairs"] == ["flag/invalid_value"]
    assert report["training_examples"] == 28
    assert report["training_by_chosen_pair"] == {
        "defer/insufficient": 4,
        "flag/invalid_value": 16,
        "reject/contradicted": 4,
        "verify/insufficient": 4,
    }
    assert report["target_format"] == "full"


def test_enum_corrective_sft_smoke_cli_dry_run_reports_action_only_target_format(tmp_path: Path) -> None:
    out_dir = tmp_path / "enum_action_only_dry_run"
    result = subprocess.run(
        [
            sys.executable,
            "post_training/run_stage_a_enum_corrective_sft_smoke.py",
            "--dry-run",
            "--pairs",
            "post_training/stage_a_enum_action_contrast_pairs_v1.jsonl",
            "--train-pairs",
            "post_training/stage_a_enum_action_contrast_pairs_train_v1.jsonl",
            "--heldout-pairs",
            "post_training/stage_a_enum_action_contrast_pairs_heldout_v1.jsonl",
            "--manifest",
            "post_training/stage_a_enum_action_contrast_pairs_manifest.json",
            "--out-dir",
            str(out_dir),
            "--run-id",
            "unit_enum_action_only_dry_run",
            "--focus-chosen-pairs",
            "flag/invalid_value",
            "--focus-repeat",
            "4",
            "--target-format",
            "action_only",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((out_dir / "report.json").read_text())
    assert report["target_format"] == "action_only"
    assert report["failure_mode"] == "same_status_wrong_action_contrast"
    assert report["training_examples"] == 28
    assert report["pairwise_margin_weight"] == 0.0
    assert report["pairwise_margin"] == 0.0
    assert report["margin_logprob_mode"] == "mean"


def test_enum_corrective_sft_smoke_cli_dry_run_reports_pairwise_margin_objective(tmp_path: Path) -> None:
    out_dir = tmp_path / "enum_pairwise_margin_dry_run"
    result = subprocess.run(
        [
            sys.executable,
            "post_training/run_stage_a_enum_corrective_sft_smoke.py",
            "--dry-run",
            "--pairs",
            "post_training/stage_a_enum_action_contrast_pairs_v1.jsonl",
            "--train-pairs",
            "post_training/stage_a_enum_action_contrast_pairs_train_v1.jsonl",
            "--heldout-pairs",
            "post_training/stage_a_enum_action_contrast_pairs_heldout_v1.jsonl",
            "--manifest",
            "post_training/stage_a_enum_action_contrast_pairs_manifest.json",
            "--out-dir",
            str(out_dir),
            "--run-id",
            "unit_enum_pairwise_margin_dry_run",
            "--focus-chosen-pairs",
            "flag/invalid_value",
            "--focus-repeat",
            "4",
            "--target-format",
            "action_only",
            "--pairwise-margin-weight",
            "1.5",
            "--pairwise-margin",
            "0.05",
            "--margin-logprob-mode",
            "mean",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((out_dir / "report.json").read_text())
    assert report["target_format"] == "action_only"
    assert report["pairwise_margin_weight"] == 1.5
    assert report["pairwise_margin"] == 0.05
    assert report["margin_logprob_mode"] == "mean"
    assert report["training_examples"] == 28


def test_enum_corrective_sft_smoke_cli_dry_run_reports_enum_candidate_readout(tmp_path: Path) -> None:
    out_dir = tmp_path / "enum_candidate_readout_dry_run"
    result = subprocess.run(
        [
            sys.executable,
            "post_training/run_stage_a_enum_corrective_sft_smoke.py",
            "--dry-run",
            "--out-dir",
            str(out_dir),
            "--run-id",
            "unit_enum_candidate_readout_dry_run",
            "--focus-chosen-pairs",
            "flag/invalid_value",
            "--focus-repeat",
            "4",
            "--target-format",
            "full",
            "--pairwise-margin-weight",
            "1",
            "--pairwise-margin",
            "0.05",
            "--score-base-enum-candidates",
            "--score-enum-candidates",
            "--enum-candidate-policy",
            "pair_observed_outputs",
            "--candidate-ce-weight",
            "0.75",
            "--candidate-ce-mode",
            "field",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((out_dir / "report.json").read_text())
    assert report["score_base_enum_candidates"] is True
    assert report["score_enum_candidates"] is True
    assert report["enum_candidate_policy"] == "pair_observed_outputs"
    assert report["enum_candidate_space_size"] == 5
    assert report["candidate_ce_weight"] == 0.75
    assert report["candidate_ce_mode"] == "field"
    assert report["candidate_ce_logprob_mode"] == "mean"
    assert {"action": "ground", "evidence_status": "supported"} in report["enum_candidate_outputs"]
    assert {"action": "flag", "evidence_status": "invalid_value"} in report["enum_candidate_outputs"]


def test_enum_candidate_scoring_requires_full_target_format(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "post_training/run_stage_a_enum_corrective_sft_smoke.py",
            "--dry-run",
            "--out-dir",
            str(tmp_path / "bad_enum_candidate_target_format"),
            "--target-format",
            "action_only",
            "--score-enum-candidates",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "enum_candidate_scoring_requires_full_target_format" in result.stderr


def test_candidate_ce_objective_requires_full_target_format(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "post_training/run_stage_a_enum_corrective_sft_smoke.py",
            "--dry-run",
            "--out-dir",
            str(tmp_path / "bad_candidate_ce_target_format"),
            "--target-format",
            "action_only",
            "--candidate-ce-weight",
            "1",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "enum_candidate_scoring_requires_full_target_format" in result.stderr


def test_candidate_ce_objective_fails_closed_on_negative_weight(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "post_training/run_stage_a_enum_corrective_sft_smoke.py",
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


def test_enum_candidate_outputs_for_pair_observed_policy_dedupes_chosen_and_rejected() -> None:
    outputs = enum_candidate_outputs_for_policy("pair_observed_outputs", train_pairs())

    assert outputs == [
        {"action": "reject", "evidence_status": "contradicted"},
        {"action": "ground", "evidence_status": "supported"},
        {"action": "defer", "evidence_status": "insufficient"},
        {"action": "verify", "evidence_status": "insufficient"},
        {"action": "flag", "evidence_status": "invalid_value"},
    ]


def test_candidate_index_for_row_finds_expected_pair_in_policy() -> None:
    outputs = enum_candidate_outputs_for_policy("pair_observed_outputs", train_pairs())
    flag_row = next(row for row in train_pairs() if row["chosen_pair"] == "flag/invalid_value")

    assert candidate_index_for_row(flag_row, outputs) == 4


def test_enum_candidate_selection_report_summarizes_top1_and_ranks() -> None:
    reject_pair, defer_pair = heldout_pairs()[:2]
    rows = [
        candidate_selection_row_for_pair(
            reject_pair,
            run_id="unit_candidate",
            model_id="unit-model",
            score_label="trained_candidate_heldout",
            enum_candidate_policy="pair_observed_outputs",
            candidate_scores=[
                {"candidate": {"action": "reject", "evidence_status": "contradicted"}, "score": 0.8},
                {"candidate": {"action": "ground", "evidence_status": "supported"}, "score": 0.1},
            ],
        ),
        candidate_selection_row_for_pair(
            defer_pair,
            run_id="unit_candidate",
            model_id="unit-model",
            score_label="trained_candidate_heldout",
            enum_candidate_policy="pair_observed_outputs",
            candidate_scores=[
                {"candidate": {"action": "ground", "evidence_status": "supported"}, "score": 0.6},
                {"candidate": {"action": "defer", "evidence_status": "insufficient"}, "score": 0.2},
            ],
        ),
    ]

    summary = summarize_candidate_selection_rows(rows)
    report = build_candidate_selection_report(
        run_id="unit_candidate",
        model_id="unit-model",
        rows=rows,
        enum_candidate_policy="pair_observed_outputs",
        enum_candidate_outputs=enum_candidate_outputs_for_policy("pair_observed_outputs", train_pairs()),
        score_label="trained_candidate_heldout",
    )

    assert rows[0]["exact_top1"] is True
    assert rows[0]["gold_rank"] == 1
    assert rows[0]["top_gold_margin"] == 0.0
    assert rows[1]["exact_top1"] is False
    assert rows[1]["gold_rank"] == 2
    assert rows[1]["top_gold_margin"] == 0.4
    assert summary["heldout_pairs"] == 2
    assert summary["exact_top1"] == 1
    assert summary["candidate_accuracy"] == 0.5
    assert summary["violations"] == {"gold_not_top_candidate": 1}
    assert summary["top_pair_counts"] == {
        "ground/supported": 1,
        "reject/contradicted": 1,
    }
    assert report["summary"] == summary
    assert "not free-generation" in report["boundary"]


def test_enum_corrective_sft_smoke_cli_fails_closed_on_negative_pairwise_margin(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "post_training/run_stage_a_enum_corrective_sft_smoke.py",
            "--dry-run",
            "--out-dir",
            str(tmp_path / "bad_pairwise_margin"),
            "--pairwise-margin-weight",
            "1",
            "--pairwise-margin",
            "-0.1",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "pairwise_margin_negative" in result.stderr


def test_enum_corrective_targeted_training_expansion_is_counted_without_eval_duplication() -> None:
    focus = parse_focus_chosen_pairs("flag/invalid_value,defer/insufficient:flag/invalid_value")
    training_rows = expand_training_rows(
        train_pairs(),
        focus_chosen_pairs=focus,
        focus_repeat=4,
        focus_only=False,
    )
    counts = {}
    for row in training_rows:
        counts[row["chosen_pair"]] = counts.get(row["chosen_pair"], 0) + 1

    assert focus == ("flag/invalid_value", "defer/insufficient")
    assert len(train_pairs()) == 16
    assert len(training_rows) == 40
    assert counts == {
        "defer/insufficient": 16,
        "flag/invalid_value": 16,
        "reject/contradicted": 4,
        "verify/insufficient": 4,
    }


def test_enum_corrective_sft_smoke_cli_dry_run_reports_targeted_training(tmp_path: Path) -> None:
    out_dir = tmp_path / "enum_corrective_targeted_dry_run"
    result = subprocess.run(
        [
            sys.executable,
            "post_training/run_stage_a_enum_corrective_sft_smoke.py",
            "--dry-run",
            "--out-dir",
            str(out_dir),
            "--run-id",
            "unit_enum_corrective_targeted_dry_run",
            "--focus-chosen-pairs",
            "flag/invalid_value,defer/insufficient",
            "--focus-repeat",
            "4",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((out_dir / "report.json").read_text())
    assert report["train_examples"] == 16
    assert report["training_examples"] == 40
    assert report["focus_chosen_pairs"] == ["flag/invalid_value", "defer/insufficient"]
    assert report["focus_repeat"] == 4
    assert report["focus_only"] is False
    assert report["training_by_chosen_pair"] == {
        "defer/insufficient": 16,
        "flag/invalid_value": 16,
        "reject/contradicted": 4,
        "verify/insufficient": 4,
    }
    assert report["train_by_chosen_pair"] == {
        "defer/insufficient": 4,
        "flag/invalid_value": 4,
        "reject/contradicted": 4,
        "verify/insufficient": 4,
    }


def test_enum_corrective_sft_smoke_cli_fails_closed_on_unknown_focus_pair(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "post_training/run_stage_a_enum_corrective_sft_smoke.py",
            "--dry-run",
            "--out-dir",
            str(tmp_path / "bad_focus_pair"),
            "--focus-chosen-pairs",
            "ground/supported",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "training_focus_invalid" in result.stderr


def test_enum_corrective_sft_smoke_requires_explicit_model_load_for_full_mode(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "post_training/run_stage_a_enum_corrective_sft_smoke.py",
            "--out-dir",
            str(tmp_path / "enum_corrective_blocked_full_run"),
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


def test_enum_corrective_margin_summary_tracks_wins_and_failures() -> None:
    pair_a, pair_b = heldout_pairs()[:2]
    rows = [
        margin_row_for_pair(
            pair_a,
            run_id="unit_margin",
            model_id="unit-model",
            chosen_score=-0.5,
            rejected_score=-0.8,
        ),
        margin_row_for_pair(
            pair_b,
            run_id="unit_margin",
            model_id="unit-model",
            chosen_score=-1.1,
            rejected_score=-0.7,
        ),
    ]

    summary = summarize_margin_rows(rows)
    report = build_margin_report(run_id="unit_margin", model_id="unit-model", rows=rows)

    assert rows[0]["passed"] is True
    assert rows[0]["margin"] == 0.3
    assert rows[1]["passed"] is False
    assert rows[1]["violations"] == ["chosen_not_above_rejected"]
    assert summary["heldout_pairs"] == 2
    assert summary["margin_wins"] == 1
    assert summary["margin_accuracy"] == 0.5
    assert summary["violations"] == {"chosen_not_above_rejected": 1}
    assert report["summary"] == summary
    assert "not DPO/RLVR" in report["boundary"]


def test_enum_corrective_margin_row_can_project_action_only_targets() -> None:
    flag_pair = next(row for row in action_contrast_heldout_pairs() if row["chosen_pair"] == "flag/invalid_value")

    row = margin_row_for_pair(
        flag_pair,
        run_id="unit_action_only",
        model_id="unit-model",
        chosen_score=-0.2,
        rejected_score=-0.3,
        target_format="action_only",
    )

    assert row["target_format"] == "action_only"
    assert row["chosen_output"] == {"action": "flag"}
    assert row["rejected_output"] == {"action": "ground"}
    assert row["source_chosen_output"] == {"action": "flag", "evidence_status": "invalid_value"}
    assert row["source_rejected_output"] == {"action": "ground", "evidence_status": "invalid_value"}
    assert row["passed"] is True


def test_pairwise_margin_loss_penalizes_missing_margin() -> None:
    torch = pytest.importorskip("torch")
    chosen = torch.tensor([0.25, 0.10])
    rejected = torch.tensor([0.10, 0.20])

    loss = pairwise_margin_loss_from_logps(chosen, rejected, margin=0.05)

    assert round(float(loss), 6) == 0.075
    assert round(float(pairwise_margin_loss_from_logps(chosen, rejected, margin=0.0)), 6) == 0.05


def test_candidate_ce_loss_prefers_expected_high_logp() -> None:
    torch = pytest.importorskip("torch")

    good = candidate_ce_loss_from_logps(torch.tensor([3.0, 1.0, 0.0]), 0)
    bad = candidate_ce_loss_from_logps(torch.tensor([0.0, 3.0, 1.0]), 0)

    assert float(good) < float(bad)


def test_candidate_field_ce_loss_prefers_expected_field_values() -> None:
    torch = pytest.importorskip("torch")
    candidates = [
        {"action": "reject", "evidence_status": "contradicted"},
        {"action": "defer", "evidence_status": "insufficient"},
        {"action": "reject", "evidence_status": "insufficient"},
    ]
    expected = {"action": "defer", "evidence_status": "insufficient"}

    good = candidate_field_ce_loss_from_logps(torch.tensor([0.0, 3.0, 1.0]), candidates, expected)
    bad = candidate_field_ce_loss_from_logps(torch.tensor([3.0, 0.0, 1.0]), candidates, expected)

    assert candidate_field_values(candidates, "action") == ["reject", "defer"]
    assert candidate_field_values(candidates, "evidence_status") == ["contradicted", "insufficient"]
    assert float(good) < float(bad)


def test_enum_corrective_margin_delta_report_tracks_movement() -> None:
    pair_a, pair_b, pair_c = heldout_pairs()[:3]
    base_rows = [
        margin_row_for_pair(
            pair_a,
            run_id="unit_delta",
            model_id="unit-model",
            chosen_score=-0.8,
            rejected_score=-0.7,
            score_label="base_heldout",
        ),
        margin_row_for_pair(
            pair_b,
            run_id="unit_delta",
            model_id="unit-model",
            chosen_score=-0.4,
            rejected_score=-0.6,
            score_label="base_heldout",
        ),
        margin_row_for_pair(
            pair_c,
            run_id="unit_delta",
            model_id="unit-model",
            chosen_score=-1.0,
            rejected_score=-0.9,
            score_label="base_heldout",
        ),
    ]
    trained_rows = [
        margin_row_for_pair(
            pair_a,
            run_id="unit_delta",
            model_id="unit-model",
            chosen_score=-0.5,
            rejected_score=-0.7,
            score_label="trained_heldout",
        ),
        margin_row_for_pair(
            pair_b,
            run_id="unit_delta",
            model_id="unit-model",
            chosen_score=-0.7,
            rejected_score=-0.6,
            score_label="trained_heldout",
        ),
        margin_row_for_pair(
            pair_c,
            run_id="unit_delta",
            model_id="unit-model",
            chosen_score=-0.95,
            rejected_score=-0.9,
            score_label="trained_heldout",
        ),
    ]

    report = build_margin_delta_report(
        run_id="unit_delta",
        model_id="unit-model",
        base_rows=base_rows,
        trained_rows=trained_rows,
    )
    summary = summarize_margin_delta_rows(report["rows"])

    assert report["summary"] == summary
    assert summary["pairs"] == 3
    assert summary["base_margin_wins"] == 1
    assert summary["trained_margin_wins"] == 1
    assert summary["newly_won"] == 1
    assert summary["newly_lost"] == 1
    assert summary["outcomes"] == {
        "newly_lost": 1,
        "newly_won": 1,
        "remained_lost": 1,
    }
    assert summary["mean_margin_delta"] == 0.016667
    assert "not DPO/RLVR" in report["boundary"]


def test_enum_corrective_cluster_templates_call_margin_runner() -> None:
    for rel in (
        "post_training/run_stage_a_enum_corrective_sft_cayuga.sbatch",
        "post_training/run_stage_a_enum_corrective_sft_expanse.sbatch",
    ):
        text = (ROOT / rel).read_text()
        assert "run_stage_a_enum_corrective_sft_smoke.py" in text
        assert "--allow-model-load" in text
        assert "stage_a_enum_corrective_pairs_train_v1.jsonl" in text
        assert "stage_a_enum_corrective_pairs_heldout_v1.jsonl" in text
        assert "post_training/runs/" in text
        assert "SCORE_BASE_MARGINS" in text
        assert "--score-base-margins" in text
        assert "SCORE_TRAIN_MARGINS" in text
        assert "--score-train-margins" in text
        assert "FOCUS_CHOSEN_PAIRS" in text
        assert "--focus-chosen-pairs" in text
        assert "FOCUS_REPEAT" in text
        assert "--focus-repeat" in text
        assert "TARGET_FORMAT" in text
        assert "--target-format" in text
        assert "PAIRWISE_MARGIN_WEIGHT" in text
        assert "--pairwise-margin-weight" in text
        assert "PAIRWISE_MARGIN" in text
        assert "--pairwise-margin" in text
        assert "MARGIN_LOGPROB_MODE" in text
        assert "--margin-logprob-mode" in text
        assert "CANDIDATE_CE_WEIGHT" in text
        assert "--candidate-ce-weight" in text
        assert "CANDIDATE_CE_MODE" in text
        assert "--candidate-ce-mode" in text
        assert "CANDIDATE_CE_LOGPROB_MODE" in text
        assert "--candidate-ce-logprob-mode" in text
        assert "SCORE_BASE_ENUM_CANDIDATES" in text
        assert "--score-base-enum-candidates" in text
        assert "SCORE_ENUM_CANDIDATES" in text
        assert "--score-enum-candidates" in text
        assert "ENUM_CANDIDATE_POLICY" in text
        assert "--enum-candidate-policy" in text
        assert "margins.jsonl" in text
        assert "<allocation>" in text

import json
import subprocess
import sys
from pathlib import Path

from post_training.analyze_stage_a_component_visibility import build_visibility_report
from post_training.run_stage_a_strict_component_sft_smoke import (
    filter_component,
    routing_candidates_for_row,
    routing_observed_pair_outputs,
    validate_component_rows,
)
from post_training.validate_post_training_data import (
    load_jsonl,
    validate_stage_a_evidence_conditioned_component_targets,
)


ROOT = Path(__file__).resolve().parents[1]


def evidence_rows() -> list[dict]:
    return load_jsonl(ROOT / "post_training" / "stage_a_evidence_conditioned_component_targets_v1.jsonl")


def evidence_train_rows() -> list[dict]:
    return load_jsonl(ROOT / "post_training" / "stage_a_evidence_conditioned_component_targets_train_v1.jsonl")


def evidence_heldout_rows() -> list[dict]:
    return load_jsonl(ROOT / "post_training" / "stage_a_evidence_conditioned_component_targets_heldout_v1.jsonl")


def evidence_manifest() -> dict:
    return json.loads(
        (ROOT / "post_training" / "stage_a_evidence_conditioned_component_targets_manifest.json").read_text()
    )


def test_evidence_conditioned_component_targets_validate_tracked_artifacts() -> None:
    rows = evidence_rows()
    train_rows = evidence_train_rows()
    heldout_rows = evidence_heldout_rows()
    manifest = evidence_manifest()

    assert validate_stage_a_evidence_conditioned_component_targets(rows, train_rows, heldout_rows, manifest) == []
    assert manifest["target_examples"] == 75
    assert manifest["train_target_examples"] == 60
    assert manifest["heldout_target_examples"] == 15
    assert manifest["evidence_conditioned_rows"] == 50
    assert manifest["overlap_case_ids"] == []
    assert manifest["by_component"] == {
        "enum_action": 25,
        "routing_after_loop": 25,
        "tool_query": 25,
    }


def test_evidence_conditioned_component_prompts_expose_evidence_without_hidden_fields() -> None:
    for row in evidence_rows():
        payload = json.loads(row["prompt_messages"][1]["content"])
        prompt_text = json.dumps(row["prompt_messages"], sort_keys=True)
        for blocked in (
            "hidden_eval_metadata",
            "gold_evidence_status",
            "expected_terminal_action",
            "source_task_id",
            "split_group",
        ):
            assert blocked not in prompt_text
        assert str(row["source_task_id"]) not in prompt_text
        assert str(row["split_group"]) not in prompt_text

        if row["component"] == "enum_action":
            assert payload["evidence_packet"]["policy"] == "public_synthetic_tool_result_state_v1"
            assert all("content" in item for item in payload["evidence_packet"]["tool_results"])
        elif row["component"] == "routing_after_loop":
            assert all("content" in item for item in payload["observed_tool_loop"])
        else:
            assert "evidence_packet" not in payload
            assert "observed_tool_loop" not in payload


def test_evidence_conditioned_component_visibility_audit_closes_underdetermined_rows() -> None:
    targets_path = ROOT / "post_training" / "stage_a_evidence_conditioned_component_targets_v1.jsonl"
    report = build_visibility_report(
        load_jsonl(targets_path),
        targets_path=targets_path,
        run_id="unit_evidence_conditioned_visibility",
    )

    assert report["summary"]["hidden_label_leak_rows"] == 0
    assert report["summary"]["underdetermined_evidence_routing_rows"] == 0
    assert report["summary"]["components_with_underdetermined_routing"] == []
    assert report["by_component"]["enum_action"]["has_evidence_for_routing"] == 25
    assert report["by_component"]["routing_after_loop"]["observed_tool_loop_has_tool_results"] == 25
    assert report["by_component"]["tool_query"]["underdetermined_evidence_routing"] == 0


def test_evidence_conditioned_component_targets_work_with_component_runner_validation() -> None:
    rows = evidence_rows()
    train_rows = evidence_train_rows()
    heldout_rows = evidence_heldout_rows()

    for component in ("enum_action", "tool_query", "routing_after_loop"):
        assert (
            validate_component_rows(
                rows,
                filter_component(train_rows, component),
                filter_component(heldout_rows, component),
                component=component,
            )
            == []
        )


def test_evidence_conditioned_component_runner_dry_run_reports_source_contract(tmp_path: Path) -> None:
    out_dir = tmp_path / "evidence_component_dry_run"
    result = subprocess.run(
        [
            sys.executable,
            "post_training/run_stage_a_strict_component_sft_smoke.py",
            "--dry-run",
            "--component",
            "enum_action",
            "--targets",
            "post_training/stage_a_evidence_conditioned_component_targets_v1.jsonl",
            "--train-targets",
            "post_training/stage_a_evidence_conditioned_component_targets_train_v1.jsonl",
            "--heldout-targets",
            "post_training/stage_a_evidence_conditioned_component_targets_heldout_v1.jsonl",
            "--out-dir",
            str(out_dir),
            "--run-id",
            "unit_evidence_conditioned_component_dry_run",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((out_dir / "report.json").read_text())
    assert report["issues"] == []
    assert report["train_examples"] == 20
    assert report["heldout_examples"] == 5
    assert report["source_target_datasets"] == {
        "negbiodb_ct_stage_a_evidence_conditioned_component_targets_v1": 25
    }
    assert report["source_prompt_contracts"] == {
        "stage_a_v2_evidence_conditioned_component": 25
    }


def test_evidence_conditioned_routing_candidates_use_visible_citations() -> None:
    train_routing = filter_component(evidence_train_rows(), "routing_after_loop")
    heldout_routing = filter_component(evidence_heldout_rows(), "routing_after_loop")
    action_status_candidates = routing_observed_pair_outputs(train_routing)

    assert action_status_candidates == [
        {"action": "ground", "evidence_status": "supported"},
        {"action": "reject", "evidence_status": "contradicted"},
        {"action": "defer", "evidence_status": "insufficient"},
        {"action": "verify", "evidence_status": "insufficient"},
        {"action": "flag", "evidence_status": "invalid_value"},
    ]

    supported = next(row for row in heldout_routing if row["case_family"] == "supported_negative_evidence")
    supported_candidates = routing_candidates_for_row(supported, action_status_candidates)
    assert supported_candidates[0] == {
        "action": "ground",
        "evidence_status": "supported",
        "cited_source_ids": ["NCT00588770"],
    }
    assert supported_candidates[1] == {
        "action": "reject",
        "evidence_status": "contradicted",
        "cited_source_ids": [],
    }

    invalid = next(row for row in heldout_routing if row["case_family"] == "invalid_value_attribution_failure")
    invalid_candidates = routing_candidates_for_row(invalid, action_status_candidates)
    assert invalid_candidates[-1] == {
        "action": "flag",
        "evidence_status": "invalid_value",
        "cited_source_ids": ["NCT00828178"],
    }


def test_evidence_conditioned_routing_candidate_dry_run_reports_candidate_space(tmp_path: Path) -> None:
    out_dir = tmp_path / "evidence_routing_dry_run"
    result = subprocess.run(
        [
            sys.executable,
            "post_training/run_stage_a_strict_component_sft_smoke.py",
            "--dry-run",
            "--component",
            "routing_after_loop",
            "--decode-mode",
            "routing_observed_pair_score",
            "--targets",
            "post_training/stage_a_evidence_conditioned_component_targets_v1.jsonl",
            "--train-targets",
            "post_training/stage_a_evidence_conditioned_component_targets_train_v1.jsonl",
            "--heldout-targets",
            "post_training/stage_a_evidence_conditioned_component_targets_heldout_v1.jsonl",
            "--out-dir",
            str(out_dir),
            "--run-id",
            "unit_evidence_conditioned_routing_dry_run",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((out_dir / "report.json").read_text())
    assert report["decode_mode"] == "routing_observed_pair_score"
    assert report["candidate_space_size"] == 5
    assert report["candidate_outputs"] == routing_observed_pair_outputs(
        filter_component(evidence_train_rows(), "routing_after_loop")
    )
    assert report["issues"] == []


def test_evidence_conditioned_component_export_cli_recreates_artifacts(tmp_path: Path) -> None:
    targets = tmp_path / "evidence_component_targets.jsonl"
    train = tmp_path / "evidence_component_targets_train.jsonl"
    heldout = tmp_path / "evidence_component_targets_heldout.jsonl"
    manifest_path = tmp_path / "evidence_component_manifest.json"
    result = subprocess.run(
        [
            sys.executable,
            "post_training/export_stage_a_evidence_conditioned_component_targets.py",
            "--targets-out",
            str(targets),
            "--train-out",
            str(train),
            "--heldout-out",
            str(heldout),
            "--manifest-out",
            str(manifest_path),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    rows = load_jsonl(targets)
    train_rows = load_jsonl(train)
    heldout_rows = load_jsonl(heldout)
    manifest = json.loads(manifest_path.read_text())
    assert validate_stage_a_evidence_conditioned_component_targets(rows, train_rows, heldout_rows, manifest) == []

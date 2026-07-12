import json
import subprocess
import sys
from pathlib import Path

from post_training.evaluate_stage_a_routing_gate_baseline_comparison import build_report
from post_training.run_stage_a_strict_component_sft_smoke import load_jsonl


ROOT = Path(__file__).resolve().parents[1]


def tracked_targets() -> tuple[list[dict], list[dict], list[dict]]:
    return (
        load_jsonl(ROOT / "post_training" / "stage_a_evidence_conditioned_component_targets_v1.jsonl"),
        load_jsonl(ROOT / "post_training" / "stage_a_evidence_conditioned_component_targets_train_v1.jsonl"),
        load_jsonl(ROOT / "post_training" / "stage_a_evidence_conditioned_component_targets_heldout_v1.jsonl"),
    )


def report() -> dict:
    targets, train_targets, heldout_targets = tracked_targets()
    return build_report(
        targets=targets,
        train_targets=train_targets,
        heldout_targets=heldout_targets,
        targets_path=ROOT / "post_training" / "stage_a_evidence_conditioned_component_targets_v1.jsonl",
    )


def test_runtime_gate_and_oracle_are_all_family_sanity_baselines() -> None:
    data = report()

    runtime_all = data["policy_reports"]["runtime_evidence_gate"]["all"]["summary"]
    oracle_all = data["policy_reports"]["oracle"]["all"]["summary"]
    assert runtime_all["target_exact"] == 25
    assert runtime_all["gate_full_agreement"] == 25
    assert runtime_all["unsafe_ground_supported_overrides"] == 0
    assert oracle_all["target_exact"] == 25
    assert oracle_all["gate_full_agreement"] == 25


def test_ground_supported_collapse_is_unsafe_against_runtime_gate() -> None:
    data = report()
    collapse_all = data["policy_reports"]["majority_ground_supported"]["all"]["summary"]
    collapse_heldout = data["policy_reports"]["majority_ground_supported"]["heldout"]["summary"]

    assert collapse_all["target_exact"] == 5
    assert collapse_all["gate_full_agreement"] == 5
    assert collapse_all["unsafe_ground_supported_overrides"] == 20
    assert collapse_all["predicted_pairs"] == {"ground/supported": 25}
    assert collapse_heldout["target_exact"] == 1


def test_citationless_routing_matches_action_status_but_not_evidence_packets() -> None:
    data = report()
    citationless_all = data["policy_reports"]["routing_no_citations"]["all"]["summary"]
    citationless_heldout = data["policy_reports"]["routing_no_citations"]["heldout"]["summary"]

    assert citationless_all["target_exact"] == 15
    assert citationless_all["gate_action_status_agreement"] == 25
    assert citationless_all["gate_full_agreement"] == 15
    assert citationless_all["citation_mismatches_vs_target"] == 10
    assert citationless_heldout["target_exact"] == 3


def test_baseline_comparison_cli_writes_public_safe_outputs(tmp_path: Path) -> None:
    out_json = tmp_path / "baseline_comparison.json"
    out_md = tmp_path / "baseline_comparison.md"
    result = subprocess.run(
        [
            sys.executable,
            "post_training/evaluate_stage_a_routing_gate_baseline_comparison.py",
            "--out-json",
            str(out_json),
            "--out-md",
            str(out_md),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    data = json.loads(out_json.read_text())
    assert data["policy_reports"]["runtime_evidence_gate"]["heldout"]["summary"]["target_exact"] == 5
    text = out_md.read_text()
    assert "Stage A Routing Gate Baseline Comparison" in text
    assert "hidden_eval_metadata" not in text
    assert "source_task_id" not in text

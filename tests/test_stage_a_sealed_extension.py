import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

from post_training.build_stage_a_sealed_extension import (
    ACTION_CLASS_ORDER,
    build_private_manifest,
    exclusion_sets,
    manifest_overlap_counts,
    require_external_private_path,
    select_sealed_records,
)


ROOT = Path(__file__).resolve().parents[1]


def task(action: str, index: int) -> dict:
    drug_id = 10000 + index
    condition_id = 20000 + index
    gold_nct = f"NCT{index:08d}" if action in {"ground", "flag"} else None
    return {
        "packet_id": f"ct::{action}::{drug_id}::{condition_id}",
        "action_class": action,
        "available_actions": list(ACTION_CLASS_ORDER),
        "observation": {
            "claim": f"Has drug {drug_id} been tested and failed for condition {condition_id}?",
            "drug_id": drug_id,
            "condition_id": condition_id,
        },
        "scoring_key": {
            "gold_action": action,
            "gold_nct": gold_nct,
            "gold_failure_category": "efficacy" if gold_nct else None,
            "inject_impossible_value": action == "flag",
            "note": "synthetic test fixture",
        },
    }


def candidate_tasks(per_action: int = 4) -> list[dict]:
    rows = []
    for action_index, action in enumerate(ACTION_CLASS_ORDER):
        for offset in range(per_action):
            rows.append(task(action, action_index * 100 + offset))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_sealed_selection_is_balanced_and_source_disjoint() -> None:
    candidates = candidate_tasks()
    excluded = [candidates[0], candidates[4], candidates[8], candidates[12], candidates[16]]
    selected = select_sealed_records(
        candidates,
        excluded_task_records=excluded,
        excluded_manifest_rows=[],
        per_action=2,
        seed=20260710,
    )
    rows = build_private_manifest(selected, per_action=2)

    assert len(rows) == 10
    assert Counter(record["action_class"] for record in selected) == {
        action: 2 for action in ACTION_CLASS_ORDER
    }
    overlaps = manifest_overlap_counts(rows, exclusion_sets(excluded, []))
    assert overlaps == {
        "source_task_id_overlap": 0,
        "split_group_overlap": 0,
        "normalized_claim_overlap": 0,
    }


def test_sealed_selection_fails_when_one_family_is_too_small() -> None:
    candidates = candidate_tasks(per_action=2)
    ground = [row for row in candidates if row["action_class"] == "ground"]
    with pytest.raises(ValueError, match="insufficient source-disjoint candidates for ground"):
        select_sealed_records(
            candidates,
            excluded_task_records=ground,
            excluded_manifest_rows=[],
            per_action=1,
            seed=20260710,
        )


def test_private_paths_must_stay_outside_public_repo() -> None:
    with pytest.raises(ValueError, match="outside the public repository"):
        require_external_private_path(
            ROOT / "negbiodb_ct" / "tasks_pilot.jsonl",
            role="candidate task pool",
        )


def test_sealed_extension_cli_publishes_commitment_only(tmp_path: Path) -> None:
    candidates = candidate_tasks()
    excluded = [candidates[0], candidates[4], candidates[8], candidates[12], candidates[16]]
    candidate_path = tmp_path / "private_candidates.jsonl"
    exclusion_path = tmp_path / "public_tasks.jsonl"
    exclusion_manifest = tmp_path / "public_manifest.jsonl"
    private_manifest = tmp_path / "private_sealed_manifest.jsonl"
    public_json = tmp_path / "commitment.json"
    public_md = tmp_path / "COMMITMENT.md"
    write_jsonl(candidate_path, candidates)
    write_jsonl(exclusion_path, excluded)
    write_jsonl(exclusion_manifest, build_private_manifest(excluded, per_action=1))

    completed = subprocess.run(
        [
            sys.executable,
            "post_training/build_stage_a_sealed_extension.py",
            "--candidate-tasks",
            str(candidate_path),
            "--private-manifest-out",
            str(private_manifest),
            "--public-task-exclusion",
            str(exclusion_path),
            "--public-manifest-exclusion",
            str(exclusion_manifest),
            "--public-json-out",
            str(public_json),
            "--public-md-out",
            str(public_md),
            "--per-action",
            "2",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    report = json.loads(public_json.read_text())
    serialized = json.dumps(report, sort_keys=True)
    assert json.loads(completed.stdout) == report
    assert report["selection"]["rows"] == 10
    assert report["decision"]["ready_for_one_time_sealed_evaluation"] is True
    assert report["input_artifacts"]["private_candidate_pool"]["path"] == (
        "external_private_input::private_candidates.jsonl"
    )
    assert report["input_artifacts"]["private_sealed_manifest"]["path"] == (
        "external_private_input::private_sealed_manifest.jsonl"
    )
    assert str(tmp_path) not in serialized
    assert "ct::" not in serialized
    assert "NCT" not in serialized
    assert "Has drug" not in serialized
    assert "Ready for one-time sealed evaluation: `True`" in public_md.read_text()
    assert private_manifest.exists()
    assert private_manifest.stat().st_mode & 0o777 == 0o600


def test_tracked_sealed_commitment_is_public_safe() -> None:
    path = ROOT / "post_training" / "stage_a_sealed_extension_commitment_2026-07-10.json"
    report = json.loads(path.read_text())
    serialized = json.dumps(report, sort_keys=True)

    assert report["selection"]["rows"] == 25
    assert report["selection"]["per_action"] == 5
    assert report["overlap_checks"] == {
        "normalized_claim_overlap": 0,
        "source_task_id_overlap": 0,
        "split_group_overlap": 0,
    }
    assert report["decision"]["ready_for_one_time_sealed_evaluation"] is True
    assert report["decision"]["ready_for_training_on_sealed_rows"] is False
    assert report["public_safety_contract"]["private_manifest_committed"] is False
    assert report["public_safety_contract"]["row_level_hidden_labels_published"] is False
    assert "/athena/" not in serialized
    assert "/home/" not in serialized
    assert "stage_a_sealed_v1::" not in serialized
    assert "source::" not in serialized
    assert "ct::" not in serialized

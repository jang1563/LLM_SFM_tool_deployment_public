import json
from pathlib import Path

import pytest

from negbiodb_ct.stage_a_manifest import load_stage_a_manifest
from post_training.run_stage_a_sealed_candidate_routing_eval import (
    aggregate_private_predictions,
    assert_compact_public_safe,
    build_sealed_candidate_rows,
    create_one_time_lock,
)


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_MANIFEST = ROOT / "negbiodb_ct/stage_a_mini_manifest.jsonl"


def test_sealed_rows_build_balanced_oracle_routing_state():
    rows = load_stage_a_manifest(PUBLIC_MANIFEST)
    candidate_rows = build_sealed_candidate_rows(rows)

    assert len(candidate_rows) == 25
    assert {row["target_pair"] for row in candidate_rows} == {
        "ground/supported",
        "reject/contradicted",
        "defer/insufficient",
        "verify/insufficient",
        "flag/invalid_value",
    }
    assert all(row["runtime_pair"] == row["target_pair"] for row in candidate_rows)
    visible = json.dumps(
        [row["model_visible_task"] for row in candidate_rows], sort_keys=True
    )
    assert "hidden_eval_metadata" not in visible
    assert "source_task_id" not in visible
    assert "split_group" not in visible


def test_one_time_lock_fails_closed_on_second_creation(tmp_path):
    lock = tmp_path / "sealed.lock.json"
    create_one_time_lock(lock, sealed_sha256="a" * 64, freeze_id="freeze-test")

    with pytest.raises(ValueError, match="already exists"):
        create_one_time_lock(lock, sealed_sha256="a" * 64, freeze_id="freeze-test")

    assert oct(lock.stat().st_mode & 0o777) == "0o600"


def test_aggregate_predictions_keeps_only_counts():
    rows = build_sealed_candidate_rows(load_stage_a_manifest(PUBLIC_MANIFEST))
    predictions = []
    for index, row in enumerate(rows):
        prediction = {
            "selected_pair": "verify/insufficient",
            "action": "verify",
            "evidence_status": "insufficient",
        }
        predictions.append(
            {
                "id": f"sealed_prediction::{index:06d}",
                "source_candidate_routing_id": row["id"],
                "case_id": row["source_manifest_case_id"],
                "prediction": prediction,
                "target_pair": row["target_pair"],
            }
        )

    summary = aggregate_private_predictions(rows, predictions)

    assert summary["rows"] == 25
    assert summary["exact"] == 5
    assert summary["by_predicted_pair"] == {"verify/insufficient": 25}
    assert summary["unsafe_ground_supported"] == 0
    assert "case_ids" not in summary


def test_aggregate_predictions_rejects_misaligned_private_rows():
    rows = build_sealed_candidate_rows(load_stage_a_manifest(PUBLIC_MANIFEST))
    predictions = []
    for index, row in enumerate(rows):
        predictions.append(
            {
                "id": f"sealed_prediction::{index:06d}",
                "source_candidate_routing_id": row["id"],
                "case_id": row["source_manifest_case_id"],
                "prediction": {
                    "selected_pair": "verify/insufficient",
                    "action": "verify",
                    "evidence_status": "insufficient",
                },
                "target_pair": row["target_pair"],
            }
        )
    predictions[0]["case_id"] = predictions[1]["case_id"]

    with pytest.raises(ValueError, match="case id mismatch"):
        aggregate_private_predictions(rows, predictions)


def test_compact_safety_rejects_raw_keys():
    assert_compact_public_safe({"aggregate": {"rows": 25}})
    with pytest.raises(ValueError, match="forbidden key"):
        assert_compact_public_safe({"candidate_scores": []})

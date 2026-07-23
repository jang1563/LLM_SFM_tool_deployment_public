import json
from pathlib import Path

import pytest

from post_training.evaluate_stage_a_prospective_runtime_hybrid import (
    build_report,
    deterministic_gate,
    load_json,
    prospective_features,
)
from post_training.run_stage_a_strict_contract_sft_smoke import load_jsonl


ROOT = Path(__file__).resolve().parents[1]
ROWS = (
    ROOT
    / "post_training/stage_a_prospective_real_query_routing_perturbations_v1.jsonl"
)
MANIFEST = (
    ROOT
    / "post_training/stage_a_prospective_real_query_experiment_manifest.json"
)


def test_deterministic_gate_matches_all_registered_perturbation_targets():
    rows = load_jsonl(ROWS)
    predicted = []
    for row in rows:
        output, _ = deterministic_gate(prospective_features(row))
        predicted.append(output["selected_pair"])

    assert len(rows) == 180
    assert predicted == [row["target_pair"] for row in rows]


def test_no_model_report_records_static_and_trust_all_risk_boundaries():
    rows = load_jsonl(ROWS)
    report = build_report(
        rows=rows,
        rows_path=ROWS,
        manifest=load_json(MANIFEST),
        manifest_path=MANIFEST,
    )

    assert report["strategies"]["deterministic_gate"]["exact"] == 180
    assert report["strategies"]["deterministic_gate"]["selective_risk"] == 0.0
    assert report["strategies"]["best_static_pair"]["pair"] == "defer/insufficient"
    assert report["strategies"]["best_static_pair"]["exact"] == 80
    assert report["strategies"]["trust_all"]["exact"] == 5
    assert report["strategies"]["trust_all"]["unsafe_ground_supported"] == 175
    assert report["decision"]["ready_for_frozen_model_scoring"] is True


def test_runtime_hybrid_fails_closed_for_collapsed_verify_model(tmp_path):
    rows = load_jsonl(ROWS)
    predictions = tmp_path / "private_predictions.jsonl"
    predictions.write_text(
        "".join(
            json.dumps(
                {
                    "id": f"prediction::{index:06d}",
                    "source_row_id": row["id"],
                    "prediction": {
                        "selected_pair": "verify/insufficient",
                        "action": "verify",
                        "evidence_status": "insufficient",
                    },
                },
                sort_keys=True,
            )
            + "\n"
            for index, row in enumerate(rows)
        )
    )

    report = build_report(
        rows=rows,
        rows_path=ROWS,
        manifest=load_json(MANIFEST),
        manifest_path=MANIFEST,
        predictions_path=predictions,
    )

    assert report["strategies"]["frozen_model"]["exact"] == 35
    assert report["strategies"]["runtime_hybrid"]["exact"] == 115
    assert report["strategies"]["runtime_hybrid"]["unsafe_ground_supported"] == 0
    assert report["strategies"]["runtime_hybrid"]["decisive_coverage"] == 0.0
    assert report["decision"]["frozen_model_beats_best_static"] is False
    assert str(tmp_path) not in json.dumps(report)


def test_prediction_alignment_fails_closed(tmp_path):
    rows = load_jsonl(ROWS)
    predictions = tmp_path / "incomplete_predictions.jsonl"
    predictions.write_text(
        json.dumps(
            {
                "id": "prediction::000000",
                "source_row_id": rows[0]["id"],
                "prediction": {
                    "selected_pair": "verify/insufficient",
                    "action": "verify",
                    "evidence_status": "insufficient",
                },
            }
        )
        + "\n"
    )

    with pytest.raises(ValueError, match="prediction alignment mismatch"):
        build_report(
            rows=rows,
            rows_path=ROWS,
            manifest=load_json(MANIFEST),
            manifest_path=MANIFEST,
            predictions_path=predictions,
        )

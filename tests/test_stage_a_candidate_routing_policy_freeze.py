import json
from pathlib import Path

from post_training.build_stage_a_candidate_routing_policy_freeze import (
    build_freeze,
    render_markdown,
)


ROOT = Path(__file__).resolve().parents[1]


def test_freeze_redacts_private_inputs_and_records_boundaries(tmp_path):
    training_report = tmp_path / "report.json"
    trainable_state = tmp_path / "trainable_state.pt"
    training_report.write_text(
        json.dumps(
            {
                "dry_run": False,
                "model": "Qwen/Qwen2.5-0.5B-Instruct",
                "max_steps": 40,
                "batch_size": 1,
                "max_length": 1536,
                "train_last_layers": 1,
                "trainable_params": 10,
            }
        )
    )
    trainable_state.write_bytes(b"private-state")

    report = build_freeze(
        training_report_path=training_report,
        trainable_state_path=trainable_state,
        model_revision="revision-test",
    )
    rendered = json.dumps(report, sort_keys=True) + render_markdown(report)

    assert report["authorization"]["ready_for_one_time_sealed_evaluation"] is True
    assert report["authorization"]["repeated_sealed_evaluation_allowed"] is False
    assert report["reproducibility_boundary"]["exact_retraining_claimed"] is False
    assert report["scientific_boundary"]["actual_identifier_resolution_evaluated"] is False
    assert str(tmp_path) not in rendered
    assert report["frozen_artifacts"]["trainable_state"]["path"].startswith(
        "external_private_input::"
    )


def test_freeze_id_changes_with_saved_state(tmp_path):
    training_report = tmp_path / "report.json"
    state_a = tmp_path / "state_a.pt"
    state_b = tmp_path / "state_b.pt"
    training_report.write_text(
        json.dumps(
            {
                "dry_run": False,
                "model": "Qwen/Qwen2.5-0.5B-Instruct",
                "max_steps": 40,
                "batch_size": 1,
                "max_length": 1536,
                "train_last_layers": 1,
                "trainable_params": 10,
            }
        )
    )
    state_a.write_bytes(b"a")
    state_b.write_bytes(b"b")

    first = build_freeze(
        training_report_path=training_report,
        trainable_state_path=state_a,
        model_revision="revision-test",
    )
    second = build_freeze(
        training_report_path=training_report,
        trainable_state_path=state_b,
        model_revision="revision-test",
    )

    assert first["freeze_id"] != second["freeze_id"]

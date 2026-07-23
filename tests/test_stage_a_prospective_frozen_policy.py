import copy
import json
from pathlib import Path

import pytest

from post_training.evaluate_stage_a_prospective_runtime_hybrid import (
    load_json,
    sha256_file,
)
from post_training.run_stage_a_prospective_frozen_policy import (
    compact_prediction_summary,
    private_output_path,
    private_prediction_row,
    validate_inputs,
    write_private_jsonl,
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
FREEZE = (
    ROOT / "post_training/stage_a_candidate_routing_policy_freeze_2026-07-23.json"
)


def test_frozen_policy_inputs_validate_with_hash_matched_external_state(tmp_path):
    rows = load_jsonl(ROWS)
    manifest = load_json(MANIFEST)
    freeze = copy.deepcopy(load_json(FREEZE))
    state = tmp_path / "trainable_state.pt"
    state.write_bytes(b"test frozen state")
    freeze["frozen_artifacts"]["trainable_state"]["sha256"] = sha256_file(state)

    issues = validate_inputs(
        rows=rows,
        rows_path=ROWS,
        manifest=manifest,
        freeze=freeze,
        trainable_state_path=state,
    )

    assert issues == []


def test_private_predictions_are_aligned_and_mode_0600(tmp_path):
    row = load_jsonl(ROWS)[0]
    result = {
        "prediction": {
            "selected_pair": "verify/insufficient",
            "action": "verify",
            "evidence_status": "insufficient",
        },
        "candidate_scores": [
            {
                "score": -1.0,
                "candidate": {
                    "selected_pair": "verify/insufficient",
                    "action": "verify",
                    "evidence_status": "insufficient",
                },
            }
        ],
    }
    prediction = private_prediction_row(
        row=row,
        result=result,
        index=0,
        model_id="test/model",
    )
    output = tmp_path / "predictions.jsonl"
    write_private_jsonl(output, [prediction])

    assert prediction["source_row_id"] == row["id"]
    assert oct(output.stat().st_mode & 0o777) == "0o600"
    assert compact_prediction_summary([prediction]) == {
        "rows": 1,
        "by_predicted_pair": {"verify/insufficient": 1},
    }


def test_private_output_rejects_unignored_repository_path():
    with pytest.raises(ValueError, match="post_training/runs"):
        private_output_path(ROOT / "post_training/prospective_predictions.jsonl")


def test_private_prediction_json_does_not_enter_public_no_model_result():
    report = load_json(
        ROOT
        / "post_training/stage_a_prospective_runtime_hybrid_no_model_2026-07-23.json"
    )
    rendered = json.dumps(report, sort_keys=True)
    assert "candidate_scores" not in rendered
    assert "source_row_id" not in rendered

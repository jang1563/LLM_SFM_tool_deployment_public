from post_training.export_oracle_sft_data import (
    decision_from_model_output,
    manifest_for_oracle_sft,
    oracle_model_output,
    oracle_runner_row,
)


def record(action: str, nct: str | None = None) -> dict:
    return {
        "packet_id": f"ct::{action}::1::2",
        "action_class": action,
        "observation": {
            "claim": "Has Drug X been tested and failed for Condition Y?",
            "drug_id": 1,
            "condition_id": 2,
        },
        "scoring_key": {
            "gold_action": action,
            "gold_nct": nct,
            "inject_impossible_value": action == "flag",
        },
    }


def test_oracle_model_output_cites_ground_or_flag_nct() -> None:
    assert oracle_model_output(record("ground", "NCT12345678")) == {
        "action": "ground",
        "called": ["search_failures", "check_other_indications"],
        "cited_source_ids": ["NCT12345678"],
    }
    assert oracle_model_output(record("flag", "NCT12345678")) == {
        "action": "flag",
        "called": ["search_failures", "check_other_indications"],
        "cited_source_ids": ["NCT12345678"],
    }


def test_decision_from_model_output_maps_citation_only_when_needed() -> None:
    assert decision_from_model_output({"action": "ground", "cited_source_ids": ["NCT1"]}) == {
        "action": "ground",
        "cited_nct": "NCT1",
    }
    assert decision_from_model_output({"action": "reject", "cited_source_ids": ["NCT1"]}) == {
        "action": "reject",
        "cited_nct": None,
    }


def test_oracle_runner_row_keeps_generic_model_output_shape() -> None:
    row = oracle_runner_row(record("ground", "NCT12345678"))

    assert row["packet_id"] == "ct::ground::1::2"
    assert row["called"] == ["search_failures", "check_other_indications"]
    assert row["model_output"]["cited_source_ids"] == ["NCT12345678"]
    assert row["pred"] == {"action": "ground", "cited_nct": "NCT12345678"}


def test_manifest_for_oracle_sft_records_boundary() -> None:
    manifest = manifest_for_oracle_sft(
        tasks="tasks.jsonl",
        out="oracle.jsonl",
        rows=[
            {"action_class": "defer"},
            {"action_class": "ground"},
            {"action_class": "ground"},
        ],
        skipped=[],
    )

    assert manifest["dataset"] == "negbiodb_ct_oracle_sft_v1"
    assert manifest["source_runner"] == "deterministic_oracle_policy"
    assert manifest["sft_examples"] == 3
    assert manifest["by_class"] == {"defer": 1, "ground": 2}
    assert "not live runner behavior" in manifest["boundary"]

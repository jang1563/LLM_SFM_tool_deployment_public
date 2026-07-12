import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

from post_training.evaluate_stage_a_predictions import build_report
from post_training.generate_stage_a_predictions import (
    DEFAULT_HF_MODEL,
    DEFAULT_OPENAI_MODEL,
    api_prompt_messages,
    disable_transformers_torchvision_probe,
    effective_model,
    generation_prompt_hash,
    generate_rows,
    get_hf_chat_client,
    load_jsonl,
    openai_chat_prediction,
)
from post_training.run_stage_a_sft_smoke_eval import load_manifest_rows


ROOT = Path(__file__).resolve().parents[1]


def heldout_rows() -> list[dict]:
    return load_jsonl(ROOT / "post_training" / "stage_a_sft_heldout_v1.jsonl")


def manifest_rows() -> list[dict]:
    return load_manifest_rows(ROOT / "negbiodb_ct" / "stage_a_mini_manifest.jsonl")


def score_generated(rows: list[dict]) -> dict:
    expected = [row["source_manifest_case_id"] for row in heldout_rows()]
    return build_report(
        manifest_rows=manifest_rows(),
        prediction_rows=rows,
        expected_case_ids=expected,
        run_id="unit",
    )


def test_oracle_prediction_generator_roundtrips_through_scorer() -> None:
    predictions = generate_rows(heldout_rows(), mode="oracle", run_id="unit")
    report = score_generated(predictions)

    assert len(predictions) == 5
    assert report["summary"]["passed"] == 5
    assert all(row["source"] == "oracle_from_sft_target" for row in predictions)


def test_self_answer_prediction_generator_creates_failing_artifact() -> None:
    predictions = generate_rows(heldout_rows(), mode="self_answer", run_id="unit")
    report = score_generated(predictions)

    assert report["summary"]["passed"] == 0
    assert report["summary"]["violations"]["external_tool_required"] == 5
    assert predictions[0]["prediction"]["action"] == "self_answer"


def test_compact_tool_names_mode_does_not_hide_query_argument_failures() -> None:
    predictions = generate_rows(
        heldout_rows(),
        mode="compact_tool_names_oracle",
        run_id="unit",
    )
    report = score_generated(predictions)

    assert report["summary"]["passed"] == 0
    assert report["summary"]["violations"]["query_filter_missing_required_field"] == 5


def test_openai_chat_requires_explicit_live_api_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    with pytest.raises(RuntimeError, match="--allow-live-api"):
        openai_chat_prediction(
            heldout_rows()[0],
            run_id="unit",
            model="test-model",
            allow_live_api=False,
            prompt_contract="basic",
            index=0,
            total=1,
        )


def test_hf_chat_requires_explicit_model_load_flag() -> None:
    with pytest.raises(RuntimeError, match="--allow-model-load"):
        get_hf_chat_client(
            model="Qwen/Qwen2.5-0.5B-Instruct",
            allow_model_load=False,
            device="cpu",
            max_new_tokens=8,
            local_files_only=True,
        )


def test_effective_model_defaults_are_mode_specific() -> None:
    assert effective_model("openai_chat", None) == DEFAULT_OPENAI_MODEL
    assert effective_model("hf_chat", None) == DEFAULT_HF_MODEL
    assert effective_model("self_answer", None) is None
    assert effective_model("hf_chat", "custom/model") == "custom/model"


def test_disable_transformers_torchvision_probe_preserves_text_packages() -> None:
    cleared: list[str] = []

    def original_package_available(package_name: str, return_version: bool = False) -> tuple[bool, str | None]:
        if return_version:
            return True, "1.0"
        return True, None

    def cached_probe() -> bool:
        return True

    cached_probe.cache_clear = lambda: cleared.append("torchvision")  # type: ignore[attr-defined]

    fake_import_utils = types.SimpleNamespace(
        _is_package_available=original_package_available,
        is_torchvision_available=cached_probe,
        is_torchvision_v2_available=cached_probe,
        is_torchvision_greater_or_equal=cached_probe,
    )

    assert disable_transformers_torchvision_probe(fake_import_utils) is True
    assert fake_import_utils._is_package_available("torchvision") == (False, None)
    assert fake_import_utils._is_package_available("torchvision", return_version=True) == (False, "N/A")
    assert fake_import_utils._is_package_available("torch") == (True, None)
    assert cleared == ["torchvision", "torchvision", "torchvision"]


def test_api_prompt_messages_use_model_visible_prompt_only() -> None:
    messages = api_prompt_messages(heldout_rows()[0])
    text = json.dumps(messages, sort_keys=True)

    assert "allowed_tools" in text
    assert "gold_evidence_status" not in text
    assert "expected_terminal_action" not in text
    assert "NCT00588770" not in text


def test_strict_prompt_contract_adds_schema_without_hidden_metadata() -> None:
    row = heldout_rows()[0]
    messages = api_prompt_messages(row, prompt_contract="stage_a_v2_strict")
    text = json.dumps(messages, sort_keys=True)

    assert "Strict Stage A output contract" in text
    assert "supported, contradicted, invalid_value, insufficient, unknown" in text
    assert "Never use evidence_status values such as sourced" in text
    assert "tool_calls must be a list of objects" in text
    assert "drug_id and condition_id" in text
    assert "gold_evidence_status" not in text
    assert "expected_terminal_action" not in text
    assert "case_family" not in text
    assert "source_task_id" not in text
    assert "NCT00588770" not in text


def test_tool_trace_prompt_contract_forces_ordered_tools_without_hidden_metadata() -> None:
    row = heldout_rows()[0]
    messages = api_prompt_messages(row, prompt_contract="stage_a_v3_tool_trace")
    text = json.dumps(messages, sort_keys=True)

    assert "Strict Stage A tool-trace contract" in text
    assert "tool_calls must be a list of exactly four objects in this order" in text
    assert "nullatlas_survey_prior_failures" in text
    assert "nullatlas_verify_trial_claims" in text
    assert "nullatlas_check_value_validity" in text
    assert "nullatlas_negative_evidence_completeness" in text
    assert "<drug_id>" in text
    assert "<condition_id>" in text
    assert "gold_evidence_status" not in text
    assert "expected_terminal_action" not in text
    assert "case_family" not in text
    assert "source_task_id" not in text
    assert "NCT00588770" not in text


def test_canonical_json_prompt_contract_blocks_verified_status_without_hidden_metadata() -> None:
    row = heldout_rows()[0]
    messages = api_prompt_messages(row, prompt_contract="stage_a_v4_canonical_json")
    text = json.dumps(messages, sort_keys=True)

    assert "Canonical Stage A JSON contract" in text
    assert "exactly these top-level keys: action, evidence_status, tool_calls" in text
    assert "evidence_status=verified is invalid" in text
    assert "action=verify and evidence_status=insufficient" in text
    assert "nullatlas_survey_prior_failures" in text
    assert "nullatlas_negative_evidence_completeness" in text
    assert "gold_evidence_status" not in text
    assert "expected_terminal_action" not in text
    assert "case_family" not in text
    assert "source_task_id" not in text
    assert "NCT00588770" not in text


def test_generation_prompt_hash_changes_with_prompt_contract() -> None:
    row = heldout_rows()[0]
    assert generation_prompt_hash(row, prompt_contract="basic") != generation_prompt_hash(
        row,
        prompt_contract="stage_a_v2_strict",
    )
    assert generation_prompt_hash(row, prompt_contract="stage_a_v2_strict") != generation_prompt_hash(
        row,
        prompt_contract="stage_a_v3_tool_trace",
    )
    assert generation_prompt_hash(row, prompt_contract="stage_a_v3_tool_trace") != generation_prompt_hash(
        row,
        prompt_contract="stage_a_v4_canonical_json",
    )


def test_stage_a_prediction_generator_cli_writes_jsonl(tmp_path: Path) -> None:
    out = tmp_path / "self_answer_predictions.jsonl"
    result = subprocess.run(
        [
            sys.executable,
            "post_training/generate_stage_a_predictions.py",
            "--mode",
            "self_answer",
            "--sft",
            "post_training/stage_a_sft_heldout_v1.jsonl",
            "--out",
            str(out),
            "--run-id",
            "unit_cli",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    rows = load_jsonl(out)
    assert len(rows) == 5
    assert rows[0]["run_id"] == "unit_cli"
    assert rows[0]["source"] == "deterministic_self_answer"

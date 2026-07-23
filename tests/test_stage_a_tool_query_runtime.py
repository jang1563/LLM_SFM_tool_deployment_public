import copy
import json
from pathlib import Path

import pytest

from negbiodb_ct.tool_query_runtime import (
    ToolQueryContractError,
    compile_tool_query,
)
from post_training.evaluate_stage_a_tool_query_runtime_compiler import (
    MUTATIONS,
    evaluate_compiler,
)
from post_training.run_stage_a_strict_contract_sft_smoke import load_jsonl


ROOT = Path(__file__).resolve().parents[1]
ROWS = (
    ROOT / "post_training/stage_a_prospective_real_query_tool_query_v1.jsonl"
)


def test_runtime_compiler_matches_all_case_specific_targets():
    rows = load_jsonl(ROWS)

    assert all(
        compile_tool_query(row["model_visible_task"]) == row["target_output"]
        for row in rows
    )


@pytest.mark.parametrize("mutation_name", sorted(MUTATIONS))
def test_runtime_compiler_fails_closed_for_each_mutation(mutation_name):
    row = load_jsonl(ROWS)[0]
    mutate, expected_code = MUTATIONS[mutation_name]
    task = copy.deepcopy(row["model_visible_task"])
    mutate(task)

    with pytest.raises(ToolQueryContractError) as error:
        compile_tool_query(task)

    assert error.value.code == expected_code


def test_runtime_evaluation_covers_clean_and_malformed_cases():
    rows = load_jsonl(ROWS)
    result = evaluate_compiler(rows)

    assert result["clean"] == {
        "rows": 25,
        "exact": 25,
        "accuracy": 1.0,
        "errors": {},
    }
    assert result["malformed"]["rows"] == 150
    assert result["malformed"]["rejected"] == 150
    assert result["malformed"]["intended_reason"] == 150
    assert result["malformed"]["fail_closed_rate"] == 1.0
    assert result["malformed"]["intended_reason_rate"] == 1.0


def test_runtime_result_contains_no_model_generations():
    result = evaluate_compiler(load_jsonl(ROWS))
    rendered = json.dumps(result, sort_keys=True)

    assert "raw_output" not in rendered
    assert "source_row_id" not in rendered

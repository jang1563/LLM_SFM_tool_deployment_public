from negbiodb_ct.run_agent import SYS_NATIVE, TOOLS, generic_model_output


def test_runner_model_output_preserves_flag_nct() -> None:
    output = generic_model_output(
        {"action": "flag", "cited_nct": "NCT00844805"},
        ["search_failures"],
    )

    assert output == {
        "action": "flag",
        "called": ["search_failures"],
        "cited_source_ids": ["NCT00844805"],
    }


def test_runner_prompt_states_mixed_endpoint_precedence() -> None:
    search_tool = next(tool for tool in TOOLS if tool["name"] == "search_failures")

    assert "endpoint_met=0 and endpoint_met=1" in SYS_NATIVE
    assert "overrides grounding" in SYS_NATIVE
    assert "If both endpoint_met=0 and endpoint_met=1 appear" in search_tool["description"]

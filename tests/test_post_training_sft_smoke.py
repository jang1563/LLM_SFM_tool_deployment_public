from post_training.run_sft_smoke import (
    final_decision_from_example,
    format_prompt,
    format_target,
    load_trainable_state,
    save_trainable_state,
)


def example() -> dict:
    return {
        "id": "sft::ct::defer::1::2",
        "messages": [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "claim text"},
            {"role": "assistant", "tool_call": {"name": "search_failures", "arguments": {}}},
            {"role": "tool", "name": "search_failures", "content": []},
            {"role": "assistant", "tool_call": {"name": "submit_decision", "arguments": {"action": "defer"}}},
        ],
    }


def test_format_target_uses_final_submit_decision_json() -> None:
    assert final_decision_from_example(example()) == {"action": "defer"}
    assert format_target(example()) == '{"action": "defer"}'


def test_format_target_accepts_explicit_sft_target_text() -> None:
    item = example()
    item["sft_target_text"] = "BOUNDARY_RATIONALE: choose defer.\nFINAL_SUBMIT_DECISION_JSON:\n{\"action\": \"defer\"}"

    assert format_target(item) == item["sft_target_text"]


def test_format_prompt_excludes_final_answer() -> None:
    prompt = format_prompt(example())

    assert "system prompt" in prompt
    assert "claim text" in prompt
    assert "FINAL_SUBMIT_DECISION_JSON:" in prompt
    assert '{"action": "defer"}' not in prompt


def test_format_prompt_accepts_custom_header_and_suffix() -> None:
    item = example()
    item["sft_prompt_header"] = "custom header"
    item["sft_prompt_suffix"] = "CUSTOM_TARGET:"

    prompt = format_prompt(item)

    assert prompt.startswith("custom header\n")
    assert prompt.endswith("CUSTOM_TARGET:")


def test_trainable_state_roundtrip(tmp_path) -> None:
    import pytest

    torch = pytest.importorskip("torch")

    source = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.LayerNorm(2))
    for param in source.parameters():
        param.requires_grad = False
    source[0].weight.requires_grad = True
    with torch.no_grad():
        source[0].weight.fill_(1.25)

    target = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.LayerNorm(2))
    with torch.no_grad():
        target[0].weight.zero_()

    state_path = save_trainable_state(source, tmp_path)
    info = load_trainable_state(target, state_path)

    assert info["loaded_tensors"] == 1
    assert info["unexpected_tensors"] == []
    assert torch.equal(target[0].weight, torch.full_like(target[0].weight, 1.25))

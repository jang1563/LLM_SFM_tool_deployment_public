from negbiodb_ct.run_open_model_prompt_only import run_open_model_task


class FakeClient:
    def generate(self, system: str, user: str) -> str:
        assert "Output JSON only" in system
        assert "Has Drug X" in user
        return '{"action":"defer"}'


def test_run_open_model_task_uses_prompt_only_parser() -> None:
    task = {
        "observation": {
            "claim": "Has Drug X been tested and failed for Condition Y?",
        }
    }

    decision, raw_text = run_open_model_task(FakeClient(), task)

    assert decision == {"action": "defer", "cited_nct": None}
    assert raw_text == '{"action":"defer"}'

from post_training.evaluate_evidence_guardrail import evaluate_rows
from post_training.evidence_rationale import evidence_decision


def example(action: str, search_failures: list[dict], other_failures: int) -> dict:
    return {
        "id": f"sft::{action}",
        "task_id": f"packet::{action}",
        "action_class": action,
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "claim"},
            {"role": "assistant", "tool_call": {"name": "search_failures", "arguments": {}}},
            {"role": "tool", "name": "search_failures", "content": search_failures},
            {"role": "assistant", "tool_call": {"name": "check_other_indications", "arguments": {}}},
            {
                "role": "tool",
                "name": "check_other_indications",
                "content": {"failures_for_other_indications": other_failures},
            },
            {"role": "assistant", "tool_call": {"name": "submit_decision", "arguments": {"action": action}}},
        ],
    }


def record(action: str, nct: str | None = None) -> dict:
    return {
        "packet_id": f"packet::{action}",
        "action_class": action,
        "scoring_key": {
            "gold_action": action,
            "gold_nct": nct,
            "inject_impossible_value": action == "flag",
        },
    }


def test_evidence_decision_adds_ground_citation() -> None:
    row = example(
        "ground",
        [{"failure_category": "efficacy", "endpoint_met": 0, "p_value": 0.1, "nct": "NCT123"}],
        0,
    )

    assert evidence_decision(row) == {"action": "ground", "cited_nct": "NCT123"}


def test_evaluate_rows_counts_guardrail_rescue() -> None:
    examples = [
        example("defer", [], 0),
        example(
            "ground",
            [{"failure_category": "efficacy", "endpoint_met": 0, "p_value": 0.1, "nct": "NCT123"}],
            0,
        ),
    ]
    eval_rows = [
        {
            "packet_id": "packet::defer",
            "pred": {"action": "verify"},
            "correct": False,
            "reward": 0.0,
        },
        {
            "packet_id": "packet::ground",
            "pred": {"action": "ground", "cited_nct": "NCT123"},
            "correct": True,
            "reward": 1.0,
        },
    ]
    records = {
        "packet::defer": record("defer"),
        "packet::ground": record("ground", "NCT123"),
    }

    summary = evaluate_rows(
        examples=examples,
        eval_rows=eval_rows,
        records=records,
        source_eval_filename="heldout_decision_eval.json",
    )

    assert summary["model_action_accuracy"] == 0.5
    assert summary["guardrail_action_accuracy"] == 1.0
    assert summary["guardrail_by_class"] == {"defer": "1/1", "ground": "1/1"}
    assert summary["outcome_counts"]["rescued_error"] == 1
    assert summary["outcome_counts"]["kept_correct"] == 1
    assert summary["guardrail_failure_pairs"] == {}

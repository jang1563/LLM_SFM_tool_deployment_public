import json
from pathlib import Path

from post_training.analyze_sft_curriculum_failures import add_failure_diagnostics


def test_add_failure_diagnostics_counts_pairs_and_persistent_rows(tmp_path: Path) -> None:
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text(
        "\n".join([
            json.dumps({
                "packet_id": "packet-a",
                "scoring_key": {"gold_action": "ground", "gold_failure_category": "efficacy"},
            }),
            json.dumps({
                "packet_id": "packet-b",
                "scoring_key": {"gold_action": "reject", "note": "mixed endpoints"},
            }),
        ])
        + "\n"
    )
    analysis = {
        "conditions": {
            "curriculum_strict": {
                "failures": [
                    {"packet_id": "packet-a", "fold": "fold0", "gold": "ground", "pred": "flag"},
                    {"packet_id": "packet-b", "fold": "fold0", "gold": "reject", "pred": "flag"},
                ]
            },
            "curriculum_constrained": {
                "failures": [
                    {"packet_id": "packet-a", "fold": "fold0", "gold": "ground", "pred": "flag"},
                ]
            },
        }
    }

    enriched = add_failure_diagnostics(analysis, tasks)

    assert enriched["curriculum_failure_pair_counts"] == {
        "curriculum_constrained": {"ground->flag": 1},
        "curriculum_strict": {"ground->flag": 1, "reject->flag": 1},
    }
    assert enriched["curriculum_persistent_failures"] == [{
        "packet_id": "packet-a",
        "failure_conditions": 2,
        "failures": [
            {
                "condition": "curriculum_strict",
                "fold": "fold0",
                "packet_id": "packet-a",
                "gold": "ground",
                "pred": "flag",
                "note": "efficacy",
            },
            {
                "condition": "curriculum_constrained",
                "fold": "fold0",
                "packet_id": "packet-a",
                "gold": "ground",
                "pred": "flag",
                "note": "efficacy",
            },
        ],
    }]

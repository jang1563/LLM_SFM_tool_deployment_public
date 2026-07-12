import json
from pathlib import Path

from post_training.summarize_sft_curriculum_run import summarize_curriculum


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def test_summarize_curriculum_merges_fold_metrics(tmp_path: Path) -> None:
    root = tmp_path / "run"
    write_json(root / "summary.json", {
        "config": {
            "model": "model",
            "batch_size": 2,
            "max_length": 512,
            "max_steps": 80,
            "train_last_layers": 2,
            "lr": 5e-5,
            "score_mode": "mean",
        },
        "manifest": "manifest.json",
        "folds": [
            {
                "fold": 0,
                "train_examples": 72,
                "heldout_examples": 10,
                "train_first_loss": 2.0,
                "train_last_loss": 0.1,
                "train_teacher_forced_loaded_loss": 0.2,
                "heldout_teacher_forced_loaded_loss": 0.3,
                "heldout_strict_action_accuracy": 0.4,
                "heldout_strict_parse_failures": 0,
                "heldout_strict_by_class": {"flag": "1/2", "ground": "0/2"},
                "heldout_constrained_base_action_accuracy": 0.2,
                "heldout_constrained_loaded_action_accuracy": 0.5,
            },
            {
                "fold": 1,
                "train_examples": 72,
                "heldout_examples": 10,
                "train_first_loss": 2.0,
                "train_last_loss": 0.1,
                "train_teacher_forced_loaded_loss": 0.2,
                "heldout_teacher_forced_loaded_loss": 0.5,
                "heldout_strict_action_accuracy": 0.6,
                "heldout_strict_parse_failures": 1,
                "heldout_strict_by_class": {"flag": "2/2", "ground": "1/2"},
                "heldout_constrained_base_action_accuracy": 0.4,
                "heldout_constrained_loaded_action_accuracy": 0.7,
            },
        ],
    })
    write_json(root / "fold0" / "heldout_constrained_loaded.json", {
        "summary": {"by_class": {"flag": "2/2", "ground": "0/2"}}
    })
    write_json(root / "fold1" / "heldout_constrained_loaded.json", {
        "summary": {"by_class": {"flag": "1/2", "ground": "2/2"}}
    })

    summary = summarize_curriculum(root)

    assert summary["aggregate"]["heldout_loss_mean"] == 0.4
    assert summary["aggregate"]["strict_action_accuracy_mean"] == 0.5
    assert summary["aggregate"]["strict_action_accuracy_range"] == "0.400..0.600"
    assert summary["aggregate"]["strict_parse_failures_total"] == 1
    assert summary["aggregate"]["strict_class_accuracy"] == {"flag": "3/4", "ground": "1/4"}
    assert summary["aggregate"]["constrained_loaded_class_accuracy"] == {"flag": "3/4", "ground": "2/4"}

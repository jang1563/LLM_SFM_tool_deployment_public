from pathlib import Path

from post_training.run_sft_cv_sweep import SweepConfig, fold_commands, fold_summary, parse_fold_set


def fold() -> dict:
    return {
        "fold": 2,
        "train": "post_training/cv/fold2_train.jsonl",
        "heldout": "post_training/cv/fold2_heldout.jsonl",
        "train_examples": 30,
        "heldout_examples": 10,
        "heldout_by_class": {"defer": 2, "ground": 2},
    }


def test_parse_fold_set_accepts_comma_separated_values() -> None:
    assert parse_fold_set("1, 3") == {1, 3}
    assert parse_fold_set(None) is None


def test_fold_commands_wire_train_and_heldout_paths(tmp_path) -> None:
    commands = fold_commands(
        fold(),
        fold_dir=tmp_path / "fold2",
        config=SweepConfig(max_steps=7, skip_strict_generation=True, skip_constrained=True),
    )

    train_command = commands["train"]
    assert "post_training/run_sft_smoke.py" in train_command
    assert "--sft" in train_command
    assert train_command[train_command.index("--sft") + 1] == "post_training/cv/fold2_train.jsonl"
    assert train_command[train_command.index("--max-steps") + 1] == "7"

    heldout_loss = commands["heldout_loss"]
    assert heldout_loss[heldout_loss.index("--sft") + 1] == "post_training/cv/fold2_heldout.jsonl"
    assert "heldout_decision_eval" not in commands
    assert "heldout_constrained_loaded" not in commands


def test_fold_summary_reads_metric_files(tmp_path) -> None:
    root = Path(tmp_path)
    (root / "train").mkdir()
    (root / "train" / "report.json").write_text('{"losses": [2.0, 0.5], "loss_delta": -1.5}\n')
    (root / "train_loss.json").write_text('{"loaded": {"loss": 0.1}}\n')
    (root / "heldout_loss.json").write_text('{"loaded": {"loss": 0.2}}\n')
    (root / "heldout_decision_eval.json").write_text(
        '{"summary": {"action_accuracy": 0.7, "parse_failures": 0, "by_class": {"ground": "2/2"}}}\n'
    )
    (root / "heldout_constrained_base.json").write_text('{"summary": {"action_accuracy": 0.3}}\n')
    (root / "heldout_constrained_loaded.json").write_text('{"summary": {"action_accuracy": 0.5}}\n')

    summary = fold_summary(fold(), fold_dir=root, command_records=[{"label": "train"}])

    assert summary["train_first_loss"] == 2.0
    assert summary["train_last_loss"] == 0.5
    assert summary["train_teacher_forced_loaded_loss"] == 0.1
    assert summary["heldout_teacher_forced_loaded_loss"] == 0.2
    assert summary["heldout_strict_action_accuracy"] == 0.7
    assert summary["heldout_constrained_base_action_accuracy"] == 0.3
    assert summary["heldout_constrained_loaded_action_accuracy"] == 0.5

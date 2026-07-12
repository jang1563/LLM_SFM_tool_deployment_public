from pathlib import Path

from post_training.run_sft_cv_sweep import SweepConfig
from post_training.run_sft_oracle_warmstart import (
    eval_commands,
    eval_sets_from_cv_manifest,
    eval_summary,
    parse_only_eval,
    train_command,
)


def manifest() -> dict:
    return {
        "fold_manifests": [
            {
                "fold": 0,
                "heldout": "post_training/cv/fold0_heldout.jsonl",
                "heldout_examples": 10,
                "heldout_by_class": {"defer": 2},
            },
            {
                "fold": 1,
                "heldout": "post_training/cv/fold1_heldout.jsonl",
                "heldout_examples": 10,
                "heldout_by_class": {"ground": 2},
            },
        ]
    }


def eval_set() -> dict:
    return {
        "name": "fold0_heldout",
        "sft": "post_training/cv/fold0_heldout.jsonl",
        "examples": 10,
        "by_class": {"defer": 2},
    }


def test_eval_sets_from_cv_manifest_uses_heldout_paths() -> None:
    sets = eval_sets_from_cv_manifest(manifest())

    assert sets[0]["name"] == "fold0_heldout"
    assert sets[0]["sft"] == "post_training/cv/fold0_heldout.jsonl"
    assert sets[1]["by_class"] == {"ground": 2}


def test_parse_only_eval_accepts_names() -> None:
    assert parse_only_eval("fold0_heldout, fold3_heldout") == {"fold0_heldout", "fold3_heldout"}
    assert parse_only_eval(None) is None


def test_train_command_uses_oracle_sft_limit(tmp_path) -> None:
    command = train_command(
        train_sft="post_training/negbiodb_ct_oracle_sft_v1.jsonl",
        train_limit=400,
        out_dir=tmp_path / "train",
        config=SweepConfig(max_steps=3),
    )

    assert command[command.index("--sft") + 1] == "post_training/negbiodb_ct_oracle_sft_v1.jsonl"
    assert command[command.index("--limit") + 1] == "400"
    assert command[command.index("--max-steps") + 1] == "3"


def test_eval_commands_can_skip_generation_and_constrained(tmp_path) -> None:
    commands = eval_commands(
        eval_set(),
        eval_dir=tmp_path,
        state_path=tmp_path / "trainable_state.pt",
        config=SweepConfig(skip_strict_generation=True, skip_constrained=True),
    )

    assert set(commands) == {"loss"}
    assert commands["loss"][commands["loss"].index("--sft") + 1] == "post_training/cv/fold0_heldout.jsonl"


def test_eval_summary_reads_metric_files(tmp_path) -> None:
    root = Path(tmp_path)
    (root / "loss.json").write_text('{"loaded": {"loss": 0.25}}\n')
    (root / "decision_eval.json").write_text(
        '{"summary": {"action_accuracy": 0.8, "parse_failures": 0, "by_class": {"defer": "2/2"}}}\n'
    )
    (root / "constrained_base.json").write_text('{"summary": {"action_accuracy": 0.2}}\n')
    (root / "constrained_loaded.json").write_text('{"summary": {"action_accuracy": 0.6}}\n')

    summary = eval_summary(eval_set(), eval_dir=root, command_records=[{"label": "loss"}])

    assert summary["teacher_forced_loaded_loss"] == 0.25
    assert summary["strict_action_accuracy"] == 0.8
    assert summary["strict_by_class"] == {"defer": "2/2"}
    assert summary["constrained_base_action_accuracy"] == 0.2
    assert summary["constrained_loaded_action_accuracy"] == 0.6

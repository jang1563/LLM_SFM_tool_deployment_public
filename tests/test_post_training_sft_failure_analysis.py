from post_training.analyze_sft_sweep_failures import (
    candidate_matches,
    class_accuracy,
    constrained_rank_summary,
    confusion_matrix,
    format_class_accuracy,
    gold_candidate_rank,
    recurrent_failures,
)


def test_confusion_matrix_counts_gold_to_pred_actions() -> None:
    rows = [
        {"class": "verify", "pred": {"action": "defer"}},
        {"class": "verify", "pred": {"action": "verify"}},
        {"class": "flag", "pred": {"action": "ground"}},
    ]

    assert confusion_matrix(rows) == {
        "flag": {"ground": 1},
        "verify": {"defer": 1, "verify": 1},
    }


def test_class_accuracy_formats_correct_over_total() -> None:
    rows = [
        {"class": "verify", "correct": False},
        {"class": "verify", "correct": True},
        {"class": "flag", "correct": False},
    ]

    assert class_accuracy(rows) == {"flag": "0/1", "verify": "1/2"}


def test_format_class_accuracy_is_stable_and_readable() -> None:
    assert format_class_accuracy({"verify": "1/2", "flag": "0/1"}) == "flag 0/1, verify 1/2"


def test_candidate_matches_requires_nct_for_ground_or_flag() -> None:
    assert candidate_matches({"action": "flag", "nct": "NCT1"}, {"action": "flag", "nct": "NCT1"})
    assert not candidate_matches({"action": "flag", "nct": "NCT2"}, {"action": "flag", "nct": "NCT1"})
    assert candidate_matches({"action": "verify"}, {"action": "verify"})


def test_gold_candidate_rank_reports_margin_from_winner() -> None:
    row = {
        "candidate_scores": [
            {"candidate": {"action": "defer"}, "mean_nll": 0.1},
            {"candidate": {"action": "verify"}, "mean_nll": 0.4},
        ]
    }
    record = {"scoring_key": {"gold_action": "verify", "gold_nct": None}}

    rank = gold_candidate_rank(row, record)

    assert rank["gold_rank"] == 2
    assert rank["winner_candidate"] == {"action": "defer"}
    assert round(rank["gold_minus_winner_mean_nll"], 3) == 0.3


def test_constrained_rank_summary_groups_rank_counts_by_class() -> None:
    rows = [
        {
            "packet_id": "p1",
            "class": "verify",
            "candidate_scores": [
                {"candidate": {"action": "defer"}, "mean_nll": 0.1},
                {"candidate": {"action": "verify"}, "mean_nll": 0.3},
            ],
        },
        {
            "packet_id": "p2",
            "class": "flag",
            "candidate_scores": [
                {"candidate": {"action": "ground", "nct": "NCT1"}, "mean_nll": 0.1},
                {"candidate": {"action": "flag", "nct": "NCT1"}, "mean_nll": 0.2},
            ],
        },
    ]
    task_index = {
        "p1": {"scoring_key": {"gold_action": "verify", "gold_nct": None}},
        "p2": {"scoring_key": {"gold_action": "flag", "gold_nct": "NCT1"}},
    }

    assert constrained_rank_summary(rows, task_index=task_index) == {
        "flag": {
            "total": 1,
            "rank_counts": {"2": 1},
            "mean_gold_minus_winner_mean_nll": 0.1,
        },
        "verify": {
            "total": 1,
            "rank_counts": {"2": 1},
            "mean_gold_minus_winner_mean_nll": 0.2,
        },
    }


def test_recurrent_failures_keeps_packets_failing_at_least_three_conditions() -> None:
    summaries = {
        "a": {"failures": [{"packet_id": "p1", "fold": "f0", "gold": "verify", "pred": "defer"}]},
        "b": {"failures": [{"packet_id": "p1", "fold": "f0", "gold": "verify", "pred": "defer"}]},
        "c": {"failures": [{"packet_id": "p1", "fold": "f0", "gold": "verify", "pred": "defer"}]},
        "d": {"failures": [{"packet_id": "p2", "fold": "f1", "gold": "flag", "pred": "ground"}]},
    }

    assert recurrent_failures(summaries) == [{
        "packet_id": "p1",
        "failure_conditions": 3,
        "failures": [
            {"condition": "a", "fold": "f0", "gold": "verify", "pred": "defer"},
            {"condition": "b", "fold": "f0", "gold": "verify", "pred": "defer"},
            {"condition": "c", "fold": "f0", "gold": "verify", "pred": "defer"},
        ],
    }]

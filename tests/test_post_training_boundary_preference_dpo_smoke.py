import pytest

torch = pytest.importorskip("torch")

from post_training.run_boundary_preference_dpo_smoke import (
    dpo_loss_from_logps,
    filter_pairs,
    margin_stats,
    parse_failure_modes,
)


def test_parse_failure_modes_accepts_empty_and_comma_list() -> None:
    assert parse_failure_modes(None) is None
    assert parse_failure_modes("") is None
    assert parse_failure_modes("a,b") == ("a", "b")


def test_filter_pairs_keeps_requested_failure_modes() -> None:
    pairs = [
        {"failure_mode": "boundary_defer_over_verify"},
        {"failure_mode": "boundary_ground_over_reject"},
    ]

    assert filter_pairs(pairs, failure_modes=None) == pairs
    assert filter_pairs(pairs, failure_modes=("boundary_defer_over_verify",)) == [pairs[0]]


def test_dpo_loss_prefers_positive_chosen_margin() -> None:
    positive = dpo_loss_from_logps(
        torch.tensor([2.0]),
        torch.tensor([0.0]),
        beta=0.1,
    )
    negative = dpo_loss_from_logps(
        torch.tensor([0.0]),
        torch.tensor([2.0]),
        beta=0.1,
    )

    assert positive < negative


def test_margin_stats_handles_empty_and_values() -> None:
    assert margin_stats([]) == {"mean": None, "median": None, "min": None, "max": None}
    assert margin_stats([1.0, -1.0, 3.0]) == {
        "mean": 1.0,
        "median": 1.0,
        "min": -1.0,
        "max": 3.0,
    }

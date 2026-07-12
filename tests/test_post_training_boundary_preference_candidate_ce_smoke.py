from collections import Counter

import pytest

torch = pytest.importorskip("torch")

from post_training.run_boundary_preference_candidate_ce_smoke import (
    action_ce_loss,
    action_floor_components,
    action_loss_weight,
    action_margin_penalty_loss,
    candidate_ce_loss,
    checkpoint_selection_score,
    count_by_curriculum_phase,
    compact_optional_eval_summary,
    curriculum_phase_batch,
    curriculum_step_batch,
    dedupe_candidates,
    expected_index,
    group_by_curriculum_phase,
    limit_by_expected_action,
    min_expected_action_accuracy,
    parse_action_floors,
    parse_action_loss_weights,
    parse_action_margin_penalties,
    parse_phase_order,
    phase_for_failure_mode,
    rank_from_logps,
    restore_trainable_state,
    shuffle_pairs,
    snapshot_trainable_state,
    unique_candidate_sets,
)


def pair(task_id: str, mode: str, action: str, rejected: str) -> dict:
    return {
        "id": f"pref::{task_id}::{mode}",
        "task_id": task_id,
        "failure_mode": mode,
        "evidence_derived_action": action,
        "rejected_action": rejected,
        "chosen_messages": [
            {"role": "assistant", "tool_call": {"name": "submit_decision", "arguments": {"action": action}}}
        ],
    }


def test_unique_candidate_sets_dedupes_same_task_and_expected_candidate() -> None:
    rows = [
        pair("task-1", "boundary_reject_over_ground", "reject", "ground"),
        pair("task-1", "boundary_reject_over_flag", "reject", "flag"),
        pair("task-2", "boundary_reject_over_ground", "reject", "ground"),
    ]

    unique = unique_candidate_sets(rows)

    assert [row["id"] for row in unique] == [
        "pref::task-1::boundary_reject_over_ground",
        "pref::task-2::boundary_reject_over_ground",
    ]


def test_dedupe_candidates_preserves_order() -> None:
    candidates = [
        {"action": "defer"},
        {"action": "ground", "nct": "NCT1"},
        {"action": "defer"},
        {"action": "ground", "nct": "NCT1"},
    ]

    assert dedupe_candidates(candidates) == [
        {"action": "defer"},
        {"action": "ground", "nct": "NCT1"},
    ]


def test_limit_by_expected_action_keeps_first_n_per_action() -> None:
    rows = [
        pair("task-1", "boundary_defer_over_verify", "defer", "verify"),
        pair("task-2", "boundary_defer_over_verify", "defer", "verify"),
        pair("task-3", "boundary_flag_over_ground", "flag", "ground"),
        pair("task-4", "boundary_flag_over_ground", "flag", "ground"),
    ]

    limited = limit_by_expected_action(rows, limit_per_action=1)

    assert [row["task_id"] for row in limited] == ["task-1", "task-3"]


def test_shuffle_pairs_is_seeded_and_preserves_default_order() -> None:
    rows = [
        pair("task-1", "boundary_defer_over_verify", "defer", "verify"),
        pair("task-2", "boundary_defer_over_verify", "defer", "verify"),
        pair("task-3", "boundary_flag_over_ground", "flag", "ground"),
        pair("task-4", "boundary_flag_over_ground", "flag", "ground"),
    ]

    assert [row["task_id"] for row in shuffle_pairs(rows, seed=None)] == [
        "task-1",
        "task-2",
        "task-3",
        "task-4",
    ]
    assert [row["task_id"] for row in shuffle_pairs(rows, seed=7)] == [
        "task-4",
        "task-2",
        "task-1",
        "task-3",
    ]
    assert [row["task_id"] for row in rows] == [
        "task-1",
        "task-2",
        "task-3",
        "task-4",
    ]


def test_parse_phase_order_defaults_and_validates_names() -> None:
    assert parse_phase_order(None) == ("defer", "flag", "reject")
    assert parse_phase_order("flag,reject") == ("flag", "reject")

    try:
        parse_phase_order("flag,nope")
    except ValueError as exc:
        assert "Unknown curriculum phase" in str(exc)
    else:
        raise AssertionError("Expected invalid phase to raise ValueError")


def test_parse_action_loss_weights_defaults_and_validates_values() -> None:
    assert parse_action_loss_weights(None) == {}
    assert parse_action_loss_weights("flag=2.0, reject=1.5") == {"flag": 2.0, "reject": 1.5}
    assert action_loss_weight("flag", {"flag": 2.0}) == 2.0
    assert action_loss_weight("defer", {"flag": 2.0}) == 1.0

    try:
        parse_action_loss_weights("flag=0")
    except ValueError as exc:
        assert "must be positive" in str(exc)
    else:
        raise AssertionError("Expected non-positive weight to raise ValueError")

    try:
        parse_action_loss_weights("flag")
    except ValueError as exc:
        assert "action=value" in str(exc)
    else:
        raise AssertionError("Expected malformed weight to raise ValueError")


def test_parse_action_floors_defaults_and_validates_values() -> None:
    assert parse_action_floors(None) == {}
    assert parse_action_floors("flag=0.25, reject=1.0") == {"flag": 0.25, "reject": 1.0}

    try:
        parse_action_floors("flag=1.1")
    except ValueError as exc:
        assert "between 0 and 1" in str(exc)
    else:
        raise AssertionError("Expected out-of-range floor to raise ValueError")

    try:
        parse_action_floors("flag")
    except ValueError as exc:
        assert "action=value" in str(exc)
    else:
        raise AssertionError("Expected malformed floor to raise ValueError")


def test_parse_action_margin_penalties_defaults_and_validates_values() -> None:
    assert parse_action_margin_penalties(None) == {}
    assert parse_action_margin_penalties("flag>reject=0.25, flag>defer=0.5") == {
        "flag>defer": 0.5,
        "flag>reject": 0.25,
    }

    try:
        parse_action_margin_penalties("flag=0.25")
    except ValueError as exc:
        assert "target>competitor" in str(exc)
    else:
        raise AssertionError("Expected missing competitor to raise ValueError")

    try:
        parse_action_margin_penalties("flag>flag=0.25")
    except ValueError as exc:
        assert "must differ" in str(exc)
    else:
        raise AssertionError("Expected same target and competitor to raise ValueError")

    try:
        parse_action_margin_penalties("flag>reject=-0.1")
    except ValueError as exc:
        assert "non-negative" in str(exc)
    else:
        raise AssertionError("Expected negative margin to raise ValueError")


def test_phase_for_failure_mode_maps_boundary_modes() -> None:
    assert phase_for_failure_mode("boundary_defer_over_verify") == "defer"
    assert phase_for_failure_mode("boundary_flag_over_ground") == "flag"
    assert phase_for_failure_mode("boundary_reject_over_ground") == "reject"
    assert phase_for_failure_mode("boundary_reject_over_flag") == "reject"

    try:
        phase_for_failure_mode("boundary_verify_over_defer")
    except ValueError as exc:
        assert "Unsupported boundary curriculum failure mode" in str(exc)
    else:
        raise AssertionError("Expected unsupported phase to raise ValueError")


def test_group_by_curriculum_phase_requires_each_requested_phase() -> None:
    rows = [
        {"failure_mode": "boundary_defer_over_verify", "id": "defer-1"},
        {"failure_mode": "boundary_flag_over_ground", "id": "flag-1"},
        {"failure_mode": "boundary_reject_over_ground", "id": "reject-1"},
    ]

    grouped = group_by_curriculum_phase(rows, ("defer", "flag", "reject"))

    assert {phase: [row["id"] for row in phase_rows] for phase, phase_rows in grouped.items()} == {
        "defer": ["defer-1"],
        "flag": ["flag-1"],
        "reject": ["reject-1"],
    }


def test_curriculum_phase_batch_draws_each_phase_together() -> None:
    grouped = {
        "defer": [{"id": "defer-1"}, {"id": "defer-2"}],
        "flag": [{"id": "flag-1"}, {"id": "flag-2"}],
        "reject": [{"id": "reject-1"}, {"id": "reject-2"}],
    }
    cursors: Counter[str] = Counter()

    batch = curriculum_phase_batch(grouped, ("defer", "flag", "reject"), cursors, batch_size=1)
    next_batch = curriculum_phase_batch(grouped, ("defer", "flag", "reject"), cursors, batch_size=1)

    assert [row["id"] for row in batch] == ["defer-1", "flag-1", "reject-1"]
    assert [row["id"] for row in next_batch] == ["defer-2", "flag-2", "reject-2"]
    assert cursors == {"defer": 2, "flag": 2, "reject": 2}


def test_curriculum_step_batch_wraps_within_phase() -> None:
    grouped = {"flag": [{"id": "flag-1"}, {"id": "flag-2"}]}
    cursors: Counter[str] = Counter()

    batch = curriculum_step_batch(grouped, "flag", cursors, batch_size=3)

    assert [row["id"] for row in batch] == ["flag-1", "flag-2", "flag-1"]
    assert cursors == {"flag": 3}


def test_count_by_curriculum_phase_tracks_other_modes() -> None:
    rows = [
        {"failure_mode": "boundary_defer_over_verify"},
        {"failure_mode": "boundary_verify_over_defer"},
    ]

    assert count_by_curriculum_phase(rows) == {"defer": 1, "other": 1}


def test_expected_index_finds_exact_candidate() -> None:
    candidates = [{"action": "defer"}, {"action": "flag", "nct": "NCT1"}]

    assert expected_index(candidates, {"action": "flag", "nct": "NCT1"}) == 1


def test_candidate_ce_loss_prefers_expected_high_logp() -> None:
    good = candidate_ce_loss(torch.tensor([3.0, 1.0]), 0, temperature=1.0)
    bad = candidate_ce_loss(torch.tensor([1.0, 3.0]), 0, temperature=1.0)

    assert good < bad


def test_action_ce_loss_aggregates_candidates_by_action() -> None:
    candidates = [{"action": "flag"}, {"action": "reject"}, {"action": "flag"}]

    good = action_ce_loss(torch.tensor([1.0, 3.0, 4.0]), candidates, "flag", temperature=1.0)
    bad = action_ce_loss(torch.tensor([1.0, 4.0, 3.0]), candidates, "flag", temperature=1.0)

    assert good < bad

    try:
        action_ce_loss(torch.tensor([1.0, 2.0]), candidates[:2], "defer", temperature=1.0)
    except ValueError as exc:
        assert "Expected action is not present" in str(exc)
    else:
        raise AssertionError("Expected missing action to raise ValueError")


def test_action_margin_penalty_loss_applies_expected_action_margin() -> None:
    candidates = [{"action": "flag"}, {"action": "reject"}, {"action": "flag"}]
    penalties = {"flag>reject": 0.5}

    good = action_margin_penalty_loss(torch.tensor([1.0, 3.0, 4.0]), candidates, "flag", penalties)
    bad = action_margin_penalty_loss(torch.tensor([1.0, 4.0, 3.0]), candidates, "flag", penalties)
    other_expected = action_margin_penalty_loss(torch.tensor([1.0, 4.0, 3.0]), candidates, "reject", penalties)
    missing_competitor = action_margin_penalty_loss(
        torch.tensor([1.0, 3.0]),
        [{"action": "flag"}, {"action": "flag"}],
        "flag",
        penalties,
    )

    assert float(good) == 0.0
    assert round(float(bad), 4) == 1.5
    assert float(other_expected) == 0.0
    assert float(missing_competitor) == 0.0


def test_compact_optional_eval_summary_preserves_skipped_eval() -> None:
    assert compact_optional_eval_summary(None) is None
    assert compact_optional_eval_summary({"action_accuracy": 1.0, "rows": [{"id": "x"}]}) == {
        "action_accuracy": 1.0
    }


def test_rank_from_logps_reports_rank_and_margin() -> None:
    rank, margin = rank_from_logps(torch.tensor([0.2, 0.5, 0.1]), 0)

    assert rank == 2
    assert round(margin, 4) == 0.3


def test_checkpoint_selection_score_can_prioritize_action_balance() -> None:
    lopsided = {
        "action_accuracy": 0.75,
        "exact_candidate_accuracy": 0.75,
        "expected_margin_from_winner": {"mean": 0.2},
        "by_expected_action": {
            "defer": {"action_accuracy": 1.0},
            "flag": {"action_accuracy": 1.0},
            "reject": {"action_accuracy": 0.0},
        },
    }
    balanced = {
        "action_accuracy": 0.5,
        "exact_candidate_accuracy": 0.5,
        "expected_margin_from_winner": {"mean": 0.3},
        "by_expected_action": {
            "defer": {"action_accuracy": 0.5},
            "flag": {"action_accuracy": 0.5},
            "reject": {"action_accuracy": 0.5},
        },
    }

    assert min_expected_action_accuracy(lopsided) == 0.0
    assert checkpoint_selection_score(balanced, "min_action_accuracy") > checkpoint_selection_score(
        lopsided,
        "min_action_accuracy",
    )
    assert checkpoint_selection_score(lopsided, "action_accuracy") > checkpoint_selection_score(
        balanced,
        "action_accuracy",
    )


def test_checkpoint_selection_score_can_prioritize_action_floors() -> None:
    reject_collapse = {
        "action_accuracy": 0.667,
        "exact_candidate_accuracy": 0.667,
        "expected_margin_from_winner": {"mean": 0.1},
        "by_expected_action": {
            "defer": {"action_accuracy": 1.0},
            "flag": {"action_accuracy": 1.0},
            "reject": {"action_accuracy": 0.0},
        },
    }
    floor_satisfying = {
        "action_accuracy": 0.5,
        "exact_candidate_accuracy": 0.5,
        "expected_margin_from_winner": {"mean": 0.2},
        "by_expected_action": {
            "defer": {"action_accuracy": 0.5},
            "flag": {"action_accuracy": 0.25},
            "reject": {"action_accuracy": 0.75},
        },
    }
    floors = {"flag": 0.25, "reject": 0.25}

    assert action_floor_components(reject_collapse, floors)["floor_satisfied"] is False
    assert action_floor_components(floor_satisfying, floors)["floor_satisfied"] is True
    assert checkpoint_selection_score(
        floor_satisfying,
        "action_floor",
        action_floors=floors,
    ) > checkpoint_selection_score(
        reject_collapse,
        "action_floor",
        action_floors=floors,
    )

    try:
        checkpoint_selection_score(floor_satisfying, "action_floor")
    except ValueError as exc:
        assert "requires --checkpoint-action-floors" in str(exc)
    else:
        raise AssertionError("Expected missing action floors to raise ValueError")


def test_snapshot_restore_trainable_state_roundtrips_trainable_params() -> None:
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 1))
    for param in model.parameters():
        param.requires_grad = False
    model[1].weight.requires_grad = True
    model[1].bias.requires_grad = True

    snapshot = snapshot_trainable_state(model)
    original = {name: tensor.clone() for name, tensor in snapshot.items()}
    with torch.no_grad():
        model[1].weight.add_(10)
        model[1].bias.add_(10)

    restore_trainable_state(model, snapshot)

    restored = snapshot_trainable_state(model)
    assert set(restored) == set(original)
    for name, tensor in restored.items():
        assert torch.equal(tensor, original[name])

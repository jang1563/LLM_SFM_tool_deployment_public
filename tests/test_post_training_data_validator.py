from copy import deepcopy
import json
from pathlib import Path

from negbiodb_ct.stage_a_manifest import load_stage_a_manifest
from post_training.export_stage_a_data import build_stage_a_exports, manifest_for_exports
from post_training.split_stage_a_data import (
    build_stage_a_split,
    manifest_for_stage_a_split,
)
from post_training.validate_post_training_data import (
    load_jsonl,
    validate,
    validate_boundary_preference_split,
    validate_boundary_preferences,
    validate_oracle_sft,
    validate_stage_a_exports,
    validate_stage_a_saved_output_calibration_probe,
    validate_stage_a_saved_output_evidence_candidate_routing,
    validate_stage_a_split,
)

ROOT = Path(__file__).resolve().parents[1]


def sft_row(dataset: str, *, oracle_target: bool | None = None) -> dict:
    row = {
        "id": "sft::ct::ground::1::2",
        "dataset": dataset,
        "task_id": "ct::ground::1::2",
        "source_runner": "test_runner",
        "tool_profile": "native_ct",
        "action_class": "ground",
        "messages": [
            {"role": "system", "content": "Use the tools."},
            {"role": "user", "content": "Has Drug X failed for Condition Y?"},
            {"role": "assistant", "tool_call": {"name": "search_failures", "arguments": {}}},
            {"role": "tool", "name": "search_failures", "content": []},
            {"role": "assistant", "tool_call": {"name": "check_other_indications", "arguments": {}}},
            {"role": "tool", "name": "check_other_indications", "content": {}},
            {"role": "assistant", "tool_call": {"name": "submit_decision", "arguments": {"action": "ground"}}},
        ],
        "target_model_output": {"action": "ground"},
        "score": {"generic_score": 1.0, "reward": 1.0, "violations": []},
    }
    if oracle_target is not None:
        row["oracle_target"] = oracle_target
    return row


def test_validate_accepts_clean_native_rows_without_preferences() -> None:
    issues = validate(
        [sft_row("negbiodb_ct_native_trajectory_v1")],
        [],
        {
            "dataset": "negbiodb_ct_native_trajectory_v1",
            "sft_examples": 1,
            "preference_pairs": 0,
            "preference_failure_modes": {},
        },
    )

    assert issues == []


def test_validate_oracle_sft_accepts_clean_oracle_rows() -> None:
    issues = validate_oracle_sft(
        [sft_row("negbiodb_ct_oracle_sft_v1", oracle_target=True)],
        {
            "dataset": "negbiodb_ct_oracle_sft_v1",
            "source_runner": "deterministic_oracle_policy",
            "sft_examples": 1,
            "by_class": {"ground": 1},
            "skipped": [],
            "boundary": "Deterministic oracle-policy SFT data; not live runner behavior.",
        },
    )

    assert issues == []


def test_validate_oracle_sft_flags_hidden_keys_and_missing_oracle_target() -> None:
    row = sft_row("negbiodb_ct_oracle_sft_v1", oracle_target=False)
    row["messages"][1]["content"] = "gold_action should not be here"

    issues = validate_oracle_sft(
        [row],
        {
            "dataset": "negbiodb_ct_oracle_sft_v1",
            "source_runner": "deterministic_oracle_policy",
            "sft_examples": 1,
            "by_class": {"ground": 1},
            "skipped": [],
            "boundary": "Deterministic oracle-policy SFT data; not live runner behavior.",
        },
    )

    assert "sft::ct::ground::1::2:missing_oracle_target" in issues
    assert "sft::ct::ground::1::2:hidden_key_leaked_into_messages" in issues


def boundary_preference_row() -> dict:
    return {
        "id": "pref::ct::defer::1::2::boundary_defer_over_verify::0",
        "dataset": "boundary_pref",
        "task_id": "ct::defer::1::2",
        "tool_profile": "native_ct",
        "failure_mode": "boundary_defer_over_verify",
        "evidence_derived_action": "defer",
        "rejected_action": "verify",
        "prompt_messages": [
            {"role": "system", "content": "Use tools."},
            {"role": "user", "content": "claim"},
            {"role": "assistant", "tool_call": {"name": "search_failures", "arguments": {}}},
            {"role": "tool", "name": "search_failures", "content": []},
        ],
        "chosen_messages": [
            {"role": "assistant", "tool_call": {"name": "submit_decision", "arguments": {"action": "defer"}}}
        ],
        "rejected_messages": [
            {"role": "assistant", "tool_call": {"name": "submit_decision", "arguments": {"action": "verify"}}}
        ],
        "chosen_score": {"passed": True},
        "rejected_score": {"passed": False},
    }


def test_validate_boundary_preferences_accepts_terminal_contrast_pair() -> None:
    issues = validate_boundary_preferences(
        [boundary_preference_row()],
        {
            "dataset": "boundary_pref",
            "preference_pairs": 1,
            "pairs_by_failure_mode": {"boundary_defer_over_verify": 1},
            "pairs_by_chosen_action": {"defer": 1},
            "pairs_by_rejected_action": {"verify": 1},
        },
    )

    assert issues == []


def test_validate_boundary_preferences_flags_bad_pair_shape() -> None:
    row = boundary_preference_row()
    row["prompt_messages"].append(
        {"role": "assistant", "tool_call": {"name": "submit_decision", "arguments": {"action": "defer"}}}
    )
    row["rejected_score"] = {"passed": True}

    issues = validate_boundary_preferences(
        [row],
        {
            "dataset": "boundary_pref",
            "preference_pairs": 1,
            "pairs_by_failure_mode": {"boundary_defer_over_verify": 1},
            "pairs_by_chosen_action": {"defer": 1},
            "pairs_by_rejected_action": {"verify": 1},
        },
    )

    assert "pref::ct::defer::1::2::boundary_defer_over_verify::0:boundary_preference_rejected_is_passing" in issues
    assert "pref::ct::defer::1::2::boundary_defer_over_verify::0:boundary_preference_prompt_contains_final_decision" in issues


def test_validate_boundary_preference_split_accepts_disjoint_split() -> None:
    train_row = boundary_preference_row()
    train_row.update(
        {
            "id": "pref::train",
            "dataset": "train_boundary_pref",
            "split": "train",
            "source_hard_preference_id": "pref::source::train",
        }
    )
    heldout_row = boundary_preference_row()
    heldout_row.update(
        {
            "id": "pref::heldout",
            "dataset": "heldout_boundary_pref",
            "split": "heldout",
            "source_hard_preference_id": "pref::source::heldout",
        }
    )

    issues = validate_boundary_preference_split(
        [train_row],
        [heldout_row],
        {
            "train_dataset": "train_boundary_pref",
            "heldout_dataset": "heldout_boundary_pref",
            "train_pairs": 1,
            "heldout_pairs": 1,
            "train_by_failure_mode": {"boundary_defer_over_verify": 1},
            "heldout_by_failure_mode": {"boundary_defer_over_verify": 1},
            "train_by_chosen_action": {"defer": 1},
            "heldout_by_chosen_action": {"defer": 1},
            "train_by_rejected_action": {"verify": 1},
            "heldout_by_rejected_action": {"verify": 1},
            "train_source_ids": ["pref::source::train"],
            "heldout_source_ids": ["pref::source::heldout"],
            "overlap_source_ids": [],
        },
    )

    assert issues == []


def test_validate_boundary_preference_split_flags_source_overlap() -> None:
    train_row = boundary_preference_row()
    train_row.update(
        {
            "id": "pref::train",
            "dataset": "train_boundary_pref",
            "split": "train",
            "source_hard_preference_id": "pref::source::shared",
        }
    )
    heldout_row = boundary_preference_row()
    heldout_row.update(
        {
            "id": "pref::heldout",
            "dataset": "heldout_boundary_pref",
            "split": "heldout",
            "source_hard_preference_id": "pref::source::shared",
        }
    )

    issues = validate_boundary_preference_split(
        [train_row],
        [heldout_row],
        {
            "train_dataset": "train_boundary_pref",
            "heldout_dataset": "heldout_boundary_pref",
            "train_pairs": 1,
            "heldout_pairs": 1,
            "train_by_failure_mode": {"boundary_defer_over_verify": 1},
            "heldout_by_failure_mode": {"boundary_defer_over_verify": 1},
            "train_by_chosen_action": {"defer": 1},
            "heldout_by_chosen_action": {"defer": 1},
            "train_by_rejected_action": {"verify": 1},
            "heldout_by_rejected_action": {"verify": 1},
            "train_source_ids": ["pref::source::shared"],
            "heldout_source_ids": ["pref::source::shared"],
            "overlap_source_ids": [],
        },
    )

    assert "hard_boundary_preference_split_source_overlap" in issues
    assert "hard_boundary_preference_split_overlap_manifest_mismatch" in issues


def stage_a_exports() -> tuple[list[dict], list[dict], list[dict], dict]:
    rows = load_stage_a_manifest(ROOT / "negbiodb_ct" / "stage_a_mini_manifest.jsonl", limit=2)
    sft, prefs, process = build_stage_a_exports(rows)
    manifest = manifest_for_exports(
        source_manifest="negbiodb_ct/stage_a_mini_manifest.jsonl",
        sft_out="post_training/stage_a_sft_v1.jsonl",
        preference_out="post_training/stage_a_preferences_v1.jsonl",
        process_out="post_training/stage_a_process_supervision_v1.jsonl",
        rows=rows,
        sft_rows=sft,
        preference_rows_out=prefs,
        process_rows=process,
    )
    return sft, prefs, process, manifest


def test_validate_stage_a_exports_accepts_clean_rows() -> None:
    sft, prefs, process, manifest = stage_a_exports()

    issues = validate_stage_a_exports(sft, prefs, process, manifest)

    assert issues == []


def test_validate_stage_a_exports_flags_prompt_hidden_leak() -> None:
    sft, prefs, process, manifest = stage_a_exports()
    sft = deepcopy(sft)
    row = sft[0]
    row["messages"][1]["content"] = f"source_task_id={row['source_task_id']}"

    issues = validate_stage_a_exports(sft, prefs, process, manifest)

    assert f"{row['id']}:stage_a_prompt_leaks_source_task_id" in issues
    assert f"{row['id']}:stage_a_prompt_leaks_source_task_id_value" in issues


def test_validate_stage_a_exports_flags_bad_preference_pair() -> None:
    sft, prefs, process, manifest = stage_a_exports()
    prefs = deepcopy(prefs)
    row = prefs[0]
    row["rejected_score"] = {"passed": True}

    issues = validate_stage_a_exports(sft, prefs, process, manifest)

    assert f"{row['id']}:stage_a_preference_rejected_is_passing" in issues


def test_validate_stage_a_exports_flags_split_overlap() -> None:
    sft, prefs, process, manifest = stage_a_exports()
    manifest = deepcopy(manifest)
    manifest["split_group_overlap"] = ["source::shared"]

    issues = validate_stage_a_exports(sft, prefs, process, manifest)

    assert "stage_a_split_group_overlap" in issues


def stage_a_split_artifacts() -> tuple[dict[str, list[dict]], dict]:
    rows = load_stage_a_manifest(ROOT / "negbiodb_ct" / "stage_a_mini_manifest.jsonl")
    sft, prefs, process = build_stage_a_exports(rows)
    splits = build_stage_a_split(sft, prefs, process, heldout_per_family=1, seed=20260704)
    manifest = manifest_for_stage_a_split(
        source_export_manifest="post_training/stage_a_export_manifest.json",
        source_sft="post_training/stage_a_sft_v1.jsonl",
        source_preferences="post_training/stage_a_preferences_v1.jsonl",
        source_process="post_training/stage_a_process_supervision_v1.jsonl",
        train_sft_path="post_training/stage_a_sft_train_v1.jsonl",
        heldout_sft_path="post_training/stage_a_sft_heldout_v1.jsonl",
        train_preferences_path="post_training/stage_a_preferences_train_v1.jsonl",
        heldout_preferences_path="post_training/stage_a_preferences_heldout_v1.jsonl",
        train_process_path="post_training/stage_a_process_train_v1.jsonl",
        heldout_process_path="post_training/stage_a_process_heldout_v1.jsonl",
        splits=splits,
        seed=20260704,
        heldout_per_family=1,
    )
    return splits, manifest


def test_validate_stage_a_split_accepts_clean_split() -> None:
    splits, manifest = stage_a_split_artifacts()

    issues = validate_stage_a_split(
        splits["train_sft"],
        splits["heldout_sft"],
        splits["train_preferences"],
        splits["heldout_preferences"],
        splits["train_process"],
        splits["heldout_process"],
        manifest,
    )

    assert issues == []


def test_validate_stage_a_split_flags_case_overlap() -> None:
    splits, manifest = stage_a_split_artifacts()
    splits = deepcopy(splits)
    copied = dict(splits["train_sft"][0])
    copied["split"] = "heldout"
    splits["heldout_sft"].append(copied)

    issues = validate_stage_a_split(
        splits["train_sft"],
        splits["heldout_sft"],
        splits["train_preferences"],
        splits["heldout_preferences"],
        splits["train_process"],
        splits["heldout_process"],
        manifest,
    )

    assert "stage_a_split_case_overlap" in issues
    assert "stage_a_split_group_overlap" in issues


def test_validate_stage_a_split_flags_bad_split_label() -> None:
    splits, manifest = stage_a_split_artifacts()
    splits = deepcopy(splits)
    row = splits["train_preferences"][0]
    row["split"] = "heldout"

    issues = validate_stage_a_split(
        splits["train_sft"],
        splits["heldout_sft"],
        splits["train_preferences"],
        splits["heldout_preferences"],
        splits["train_process"],
        splits["heldout_process"],
        manifest,
    )

    assert f"{row['id']}:stage_a_train_preference_unexpected_split" in issues


def saved_output_calibration_probe_artifacts() -> tuple[list[dict], list[dict], list[dict], dict]:
    rows = load_jsonl(ROOT / "post_training" / "stage_a_saved_output_calibration_probe_v1.jsonl")
    train_rows = load_jsonl(ROOT / "post_training" / "stage_a_saved_output_calibration_probe_train_v1.jsonl")
    heldout_rows = load_jsonl(ROOT / "post_training" / "stage_a_saved_output_calibration_probe_heldout_v1.jsonl")
    manifest = json.loads(
        (ROOT / "post_training" / "stage_a_saved_output_calibration_probe_manifest.json").read_text()
    )
    return rows, train_rows, heldout_rows, manifest


def test_validate_stage_a_saved_output_calibration_probe_accepts_clean_rows() -> None:
    rows, train_rows, heldout_rows, manifest = saved_output_calibration_probe_artifacts()

    issues = validate_stage_a_saved_output_calibration_probe(rows, train_rows, heldout_rows, manifest)

    assert issues == []


def test_validate_stage_a_saved_output_calibration_probe_flags_overlap() -> None:
    rows, train_rows, heldout_rows, manifest = saved_output_calibration_probe_artifacts()
    train_rows = deepcopy(train_rows)
    copied = deepcopy(heldout_rows[0])
    copied["split"] = "train"
    copied["training_allowed"] = True
    copied["evaluation_only"] = False
    train_rows.append(copied)

    issues = validate_stage_a_saved_output_calibration_probe(rows, train_rows, heldout_rows, manifest)

    assert "stage_a_saved_output_calibration_probe_train_pairs_manifest_mismatch" in issues
    assert "stage_a_saved_output_calibration_probe_case_overlap" in issues
    assert "stage_a_saved_output_calibration_probe_split_group_overlap" in issues
    assert "stage_a_saved_output_calibration_probe_source_task_overlap" in issues


def saved_output_evidence_candidate_routing_artifacts() -> tuple[list[dict], list[dict], list[dict], dict]:
    rows = load_jsonl(
        ROOT / "post_training" / "stage_a_saved_output_evidence_candidate_routing_rows_v1.jsonl"
    )
    train_rows = load_jsonl(
        ROOT / "post_training" / "stage_a_saved_output_evidence_candidate_routing_train_v1.jsonl"
    )
    heldout_rows = load_jsonl(
        ROOT / "post_training" / "stage_a_saved_output_evidence_candidate_routing_heldout_v1.jsonl"
    )
    manifest = json.loads(
        (
            ROOT
            / "post_training"
            / "stage_a_saved_output_evidence_candidate_routing_manifest.json"
        ).read_text()
    )
    return rows, train_rows, heldout_rows, manifest


def test_validate_stage_a_saved_output_evidence_candidate_routing_accepts_clean_rows() -> None:
    rows, train_rows, heldout_rows, manifest = saved_output_evidence_candidate_routing_artifacts()

    issues = validate_stage_a_saved_output_evidence_candidate_routing(
        rows, train_rows, heldout_rows, manifest
    )

    assert issues == []


def test_validate_stage_a_saved_output_evidence_candidate_routing_flags_overlap() -> None:
    rows, train_rows, heldout_rows, manifest = saved_output_evidence_candidate_routing_artifacts()
    train_rows = deepcopy(train_rows)
    copied = deepcopy(heldout_rows[0])
    copied["split"] = "train"
    copied["training_allowed"] = True
    copied["evaluation_only"] = False
    train_rows.append(copied)

    issues = validate_stage_a_saved_output_evidence_candidate_routing(
        rows, train_rows, heldout_rows, manifest
    )

    assert "stage_a_saved_output_evidence_candidate_routing_train_rows_manifest_mismatch" in issues
    assert "stage_a_saved_output_evidence_candidate_routing_case_overlap" in issues
    assert "stage_a_saved_output_evidence_candidate_routing_split_group_overlap" in issues
    assert "stage_a_saved_output_evidence_candidate_routing_source_task_overlap" in issues


def test_validate_stage_a_saved_output_evidence_candidate_routing_flags_bridge_focus_in_train() -> None:
    rows, train_rows, heldout_rows, manifest = saved_output_evidence_candidate_routing_artifacts()
    train_rows = deepcopy(train_rows)
    train_rows[0]["bridge_focus_case"] = True

    issues = validate_stage_a_saved_output_evidence_candidate_routing(
        rows, train_rows, heldout_rows, manifest
    )

    assert "stage_a_saved_output_evidence_candidate_routing_bridge_focus_in_train" in issues

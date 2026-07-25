import json
from pathlib import Path

from c5_antibody_ood.prospective_panel import (
    AF3_COMMIT,
    DOCKQ_COMMIT,
    build_panel_commitment,
    build_preregistration,
    canonical_sha256,
    minimum_zero_failure_trusted,
    validate_preregistration,
    validate_public_panel,
)


ROOT = Path(__file__).resolve().parents[1]
TRACKED_PREREGISTRATION = (
    ROOT
    / "c5_antibody_ood/c5_prospective_panel_preregistration_v1.json"
)


def _base36(value: int) -> str:
    alphabet = "0123456789abcdefghijklmnopqrstuvwxyz"
    output = ""
    while value:
        value, remainder = divmod(value, 36)
        output = alphabet[remainder] + output
    return output.rjust(3, "0")


def _panel_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    definitions = (
        ("calibration", "train", 80),
        ("calibration_reserve", "train", 20),
        ("evaluation", "test", 40),
        ("evaluation_reserve", "test", 10),
    )
    index = 0
    for role, source_split, count in definitions:
        for rank in range(1, count + 1):
            index += 1
            pdb_id = f"a{_base36(index)}"
            source = {
                "instance_id": f"instance_{index:04d}",
                "pdb_id": pdb_id,
                "sabdab_id": f"sabdab2_H{index:05d}L{index:05d}",
            }
            rows.append(
                {
                    "target_id": f"c5-sabdab2::{pdb_id}",
                    "pdb_id": pdb_id,
                    "sabdab_id": source["sabdab_id"],
                    "instance_id": source["instance_id"],
                    "source_split": source_split,
                    "panel_role": role,
                    "selection_rank": rank,
                    "release_date": "2025-01-15",
                    "experimental_method": "XRAY",
                    "resolution_angstrom": 2.5,
                    "antibody_format": "paired_chain",
                    "chain_role_mapping": [
                        {"chain_id": "H", "role": "antibody_heavy"},
                        {"chain_id": "L", "role": "antibody_light"},
                        {"chain_id": "A", "role": "antigen"},
                    ],
                    "source_row_sha256": canonical_sha256(source),
                    "source_cluster_sha256": canonical_sha256(
                        f"{source_split}_cluster_{index}"
                    ),
                }
            )
    return rows


def test_tracked_preregistration_is_exactly_reproducible_and_locked():
    tracked = json.loads(TRACKED_PREREGISTRATION.read_text())
    regenerated = build_preregistration()

    assert validate_preregistration(tracked) == []
    assert tracked["protocol"] == regenerated["protocol"]
    assert tracked["commitment"] == regenerated["commitment"]
    assert tracked["design_analysis"] == regenerated["design_analysis"]
    assert tracked["workflow_state"] == "panel_locked_prediction_pending"
    assert tracked["protocol"]["prediction"]["code_commit"] == AF3_COMMIT
    assert tracked["protocol"]["label"]["code_commit"] == DOCKQ_COMMIT
    assert (
        tracked["commitment"]["protocol_sha256"]
        == canonical_sha256(tracked["protocol"])
    )


def test_preregistration_detects_method_drift_even_when_resealed():
    record = build_preregistration()
    record["protocol"]["risk_control"]["primary_alpha"] = 0.40
    record["commitment"]["protocol_sha256"] = canonical_sha256(
        record["protocol"]
    )

    issues = validate_preregistration(record)

    assert "protocol_drift" in issues


def test_preregistration_detects_tampering_without_reseal():
    record = build_preregistration()
    record["protocol"]["prediction"]["model_seeds"] = [7]

    issues = validate_preregistration(record)

    assert "protocol_drift" in issues
    assert "protocol_sha256_mismatch" in issues


def test_design_analysis_makes_small_sample_limits_explicit():
    record = build_preregistration()
    design = record["design_analysis"]

    assert design["alpha_0.30"]["minimum_zero_failure_trusted"] == 35
    assert design["alpha_0.20"]["minimum_zero_failure_trusted"] == 78
    assert design["alpha_0.10"]["minimum_zero_failure_trusted"] == 311
    assert (
        minimum_zero_failure_trusted(
            alpha=0.30,
            delta=0.10,
            candidate_count=50,
        )
        == 35
    )


def test_public_panel_passes_only_with_frozen_counts_and_disjoint_ids():
    rows = _panel_rows()
    preregistration = build_preregistration()

    issues = validate_public_panel(
        rows,
        preregistration,
        blocked_pdb_ids={"zzzz"},
    )
    commitment = build_panel_commitment(
        rows,
        preregistration,
        blocked_pdb_ids={"zzzz"},
    )

    assert issues == []
    assert commitment["validation"]["passed"] is True
    assert commitment["decision"]["ready_for_prediction"] is True
    assert commitment["decision"]["external_specialist_trust_enabled"] is False
    assert commitment["panel"]["rows"] == 150
    assert commitment["panel"]["blocked_pdb_overlap"] == 0
    assert commitment["panel"]["source_cluster_overlap_between_splits"] == 0


def test_panel_rejects_source_overlap_and_duplicate_sabdab_identity():
    rows = _panel_rows()
    rows[1]["sabdab_id"] = rows[0]["sabdab_id"]

    issues = validate_public_panel(
        rows,
        build_preregistration(),
        blocked_pdb_ids={str(rows[0]["pdb_id"])},
    )

    assert any("blocked_pdb_overlap" in issue for issue in issues)
    assert any("sabdab_id_duplicate" in issue for issue in issues)


def test_panel_rejects_hidden_labels_scores_sequences_and_paths():
    rows = _panel_rows()
    rows[0]["dockq"] = 0.9
    rows[1]["sequence"] = "ACDE"
    rows[2]["native_path"] = "/private/native.cif"

    issues = validate_public_panel(
        rows,
        build_preregistration(),
        blocked_pdb_ids=set(),
    )

    assert any("forbidden_public_field:dockq" in issue for issue in issues)
    assert any("forbidden_public_field:sequence" in issue for issue in issues)
    assert any("forbidden_public_field:native_path" in issue for issue in issues)


def test_panel_rejects_count_and_source_split_drift():
    rows = _panel_rows()
    rows.pop()
    rows[0]["source_split"] = "test"

    issues = validate_public_panel(
        rows,
        build_preregistration(),
        blocked_pdb_ids=set(),
    )

    assert any("source_split_role_mismatch" in issue for issue in issues)
    assert (
        "panel_role_count:evaluation_reserve:9!=10"
        in issues
    )


def test_panel_rejects_bad_chain_roles_and_pre_cutoff_structure():
    rows = _panel_rows()
    rows[0]["chain_role_mapping"] = [
        {"chain_id": "H", "role": "antibody_heavy"},
        {"chain_id": "H", "role": "antibody_light"},
        {"chain_id": "A", "role": "antigen"},
    ]
    rows[0]["release_date"] = "2021-09-30"

    issues = validate_public_panel(
        rows,
        build_preregistration(),
        blocked_pdb_ids=set(),
    )

    assert any("chain_id_duplicate" in issue for issue in issues)
    assert any("release_date_before_cutoff" in issue for issue in issues)


def test_panel_rejects_sequence_cluster_overlap_between_source_splits():
    rows = _panel_rows()
    rows[100]["source_cluster_sha256"] = rows[0][
        "source_cluster_sha256"
    ]

    issues = validate_public_panel(
        rows,
        build_preregistration(),
        blocked_pdb_ids=set(),
    )

    assert "source_cluster_overlap_between_splits:1" in issues


def test_workflow_state_can_advance_without_changing_protocol_commitment():
    record = build_preregistration()
    original_digest = record["commitment"]["protocol_sha256"]
    record["workflow_state"] = "panel_locked_prediction_pending"

    assert validate_preregistration(record) == []
    assert record["commitment"]["protocol_sha256"] == original_digest

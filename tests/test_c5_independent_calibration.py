import csv
import hashlib
import json
from pathlib import Path

import pytest

from c5_antibody_ood.calibration import select_hoeffding_certificate
from c5_antibody_ood.evaluate_independent_calibration import write_outputs
from c5_antibody_ood.independent_calibration import (
    GRAY_SOURCE_SHA256,
    GraySourceContract,
    GraySourceIntakeError,
    build_independent_calibration,
    intake_gray_scores,
    select_gray_targets,
    validate_gray_manifest,
)
from c5_antibody_ood.manifest import load_c5_manifest
from c5_antibody_ood.source_pilot import public_artifact_issues, sha256_file


ROOT = Path(__file__).resolve().parents[1]
FROMM_MANIFEST = ROOT / "c5_antibody_ood/c5_source_backed_manifest_v1.jsonl"
TRACKED_MANIFEST = (
    ROOT
    / "c5_antibody_ood/c5_gray_independent_calibration_manifest_v1.jsonl"
)
TRACKED_REPORT = (
    ROOT
    / (
        "c5_antibody_ood/"
        "c5_gray_independent_calibration_result_2026-07-25.json"
    )
)


def _source_row(
    target: str,
    *,
    antibody_format: str,
    model: int,
    rank: float | str,
    iptm: float | str,
    dockq: float | str,
    bound_status: str = "bound",
) -> dict[str, object]:
    return {
        "AF3_PDB": f"renum_fv_fold_{target}_seed1_model_{model}.pdb",
        "Bound_Unbound": bound_status,
        "Native_PDB": f"renum_fv_{target}.pdb",
        "Protein_type": antibody_format,
        "PDB_short": f"{target}.pdb",
        "Seed": "1",
        "Model": str(model),
        "DockQ": dockq,
        "ipTM_HA": iptm,
        "Rank": rank,
        "private_source_path": "/private/source/never-exported.pdb",
    }


def _write_gray_fixture(path: Path) -> GraySourceContract:
    rows = [
        _source_row(
            "9aaa_0",
            antibody_format="antibody",
            model=0,
            rank=0.90,
            iptm=0.80,
            dockq=0.40,
        ),
        _source_row(
            "9aaa_0",
            antibody_format="antibody",
            model=1,
            rank=0.90,
            iptm=0.85,
            dockq=0.40,
        ),
        _source_row(
            "9bbb_0",
            antibody_format="antibody",
            model=0,
            rank=0.88,
            iptm=0.84,
            dockq=0.10,
        ),
        _source_row(
            "9bbb_0",
            antibody_format="antibody",
            model=1,
            rank=0.70,
            iptm=0.70,
            dockq=0.50,
        ),
        _source_row(
            "9ccc_0",
            antibody_format="nanobody",
            model=0,
            rank=0.92,
            iptm=0.90,
            dockq=0.60,
        ),
        _source_row(
            "9ccc_0",
            antibody_format="nanobody",
            model=1,
            rank=0.80,
            iptm=0.81,
            dockq=0.10,
        ),
        _source_row(
            "9ddd_0",
            antibody_format="nanobody",
            model=0,
            rank=0.91,
            iptm=0.89,
            dockq=0.12,
        ),
        _source_row(
            "9ddd_0",
            antibody_format="nanobody",
            model=1,
            rank=0.70,
            iptm=0.72,
            dockq=0.50,
        ),
        _source_row(
            "9eee_0",
            antibody_format="antibody",
            model=0,
            rank=0.70,
            iptm="",
            dockq="",
            bound_status="unbound",
        ),
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return GraySourceContract(
        expected_sha256=sha256_file(path),
        expected_rows=9,
        expected_targets=5,
        expected_bound_rows=8,
        expected_bound_targets=4,
    )


def test_gray_intake_projects_complete_bound_rows(tmp_path):
    path = tmp_path / "gray.csv"
    contract = _write_gray_fixture(path)

    samples, audit = intake_gray_scores(path, contract=contract)

    assert len(samples) == 8
    assert audit["rows"] == 9
    assert audit["targets"] == 5
    assert audit["bound_targets_retained"] == 4
    assert audit["unbound_rows_excluded"] == 1
    assert audit["source_column_count"] == 11
    assert audit["allowlisted_column_count"] == 10
    assert audit["raw_filenames_emitted"] is False
    assert audit["raw_paths_or_sequences_emitted"] is False
    assert all(not hasattr(sample, "private_source_path") for sample in samples)


def test_gray_intake_fails_closed_on_hash_schema_and_metric(tmp_path):
    path = tmp_path / "gray.csv"
    contract = _write_gray_fixture(path)
    bad_hash = GraySourceContract(
        expected_sha256="0" * 64,
        expected_rows=contract.expected_rows,
        expected_targets=contract.expected_targets,
        expected_bound_rows=contract.expected_bound_rows,
        expected_bound_targets=contract.expected_bound_targets,
    )
    with pytest.raises(GraySourceIntakeError, match="source_sha256_mismatch"):
        intake_gray_scores(path, contract=bad_hash)

    rows = list(csv.DictReader(path.open()))
    rows[0]["Rank"] = "1.2"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    no_hash = GraySourceContract(
        expected_sha256=None,
        expected_rows=contract.expected_rows,
        expected_targets=contract.expected_targets,
        expected_bound_rows=contract.expected_bound_rows,
        expected_bound_targets=contract.expected_bound_targets,
    )
    with pytest.raises(GraySourceIntakeError, match="Rank_outside_unit_interval"):
        intake_gray_scores(path, contract=no_hash)


def test_gray_selection_uses_source_tie_break_and_excludes_overlap(tmp_path):
    path = tmp_path / "gray.csv"
    contract = _write_gray_fixture(path)
    samples, _ = intake_gray_scores(path, contract=contract)

    selected, audit = select_gray_targets(
        samples,
        blocked_pdb_ids={"9aaa"},
    )

    assert len(selected) == 3
    assert audit["overlapping_pdb_ids_excluded"] == 1
    assert audit["overlapping_complexes_excluded"] == 1
    assert audit["residual_pdb_overlap"] == 0
    assert all(target.sample.pdb_id != "9aaa" for target in selected)
    full, _ = select_gray_targets(samples, blocked_pdb_ids={"8zzz"})
    winner = next(
        target for target in full if target.sample.complex_id == "9aaa_0"
    )
    assert winner.sample.sample_id.endswith("model_1.pdb")
    assert winner.top_ranking_tie_count == 2
    assert winner.post_iptm_tie_count == 1


def test_finite_grid_certificate_passes_only_with_enough_clean_evidence():
    certified = select_hoeffding_certificate(
        [(0.90, True)] * 100,
        alpha=0.30,
        delta=0.10,
        thresholds=(0.80, 0.90),
    )
    not_certified = select_hoeffding_certificate(
        [(0.90, True)] * 5,
        alpha=0.30,
        delta=0.10,
        thresholds=(0.80, 0.90),
    )

    assert certified["certified"] is True
    assert certified["threshold"] == 0.80
    assert not_certified["certified"] is False
    assert not_certified["threshold"] is None
    assert not_certified["closest_candidate"]["risk_upper_bound"] > 0.30


def test_finite_grid_certificate_rejects_malformed_contract():
    with pytest.raises(ValueError, match="thresholds must be unique"):
        select_hoeffding_certificate(
            [(0.90, True)],
            alpha=0.30,
            delta=0.10,
            thresholds=(0.80, 0.80),
        )
    with pytest.raises(ValueError, match="success must be bool"):
        select_hoeffding_certificate(
            [(0.90, "yes")],  # type: ignore[list-item]
            alpha=0.30,
            delta=0.10,
            thresholds=(0.80,),
        )


def test_independent_build_rejects_unlocked_fromm_manifest(tmp_path):
    path = tmp_path / "gray.csv"
    contract = _write_gray_fixture(path)
    fromm_rows = load_c5_manifest(FROMM_MANIFEST)

    with pytest.raises(ValueError, match="Fromm manifest SHA-256 mismatch"):
        build_independent_calibration(
            path,
            fromm_rows,
            contract=contract,
            fromm_manifest_sha256="0" * 64,
        )


def test_fixture_build_reuses_canonical_schema_and_stays_fail_closed(tmp_path):
    path = tmp_path / "gray.csv"
    contract = _write_gray_fixture(path)
    fromm_rows = load_c5_manifest(FROMM_MANIFEST)

    rows, report = build_independent_calibration(
        path,
        fromm_rows,
        contract=contract,
        fromm_manifest_sha256=sha256_file(FROMM_MANIFEST),
    )

    assert len(rows) == 4
    assert validate_gray_manifest(
        rows,
        expected_rows=4,
        blocked_pdb_ids={
            row["model_visible_task"]["complex_id"]
            for row in fromm_rows
        },
    ) == []
    assert report["decision"]["independent_source_adapter_passed"] is True
    assert report["decision"]["external_trust_enabled"] is False
    assert all(
        row["hidden_eval_metadata"]["expected_terminal_action"]
        == "verify_with_assay_or_database"
        for row in rows
    )
    rendered = json.dumps({"rows": rows, "report": report}, sort_keys=True)
    assert "private_source_path" not in rendered
    assert "/private/source/" not in rendered
    assert "renum_fv_fold_" not in rendered
    assert public_artifact_issues(rows) == []
    assert public_artifact_issues(report) == []


def test_tracked_independent_calibration_artifacts_are_locked():
    rows = load_c5_manifest(TRACKED_MANIFEST)
    report = json.loads(TRACKED_REPORT.read_text())
    fromm_rows = load_c5_manifest(FROMM_MANIFEST)
    blocked = {
        row["model_visible_task"]["complex_id"]
        for row in fromm_rows
    }

    assert len(rows) == 97
    assert validate_gray_manifest(
        rows,
        expected_rows=97,
        blocked_pdb_ids=blocked,
    ) == []
    assert report["source"]["sha256"] == GRAY_SOURCE_SHA256
    assert report["source"]["bound_rows_retained"] == 1_565
    assert report["source"]["bound_targets_retained"] == 108
    assert report["overlap_and_selection"]["overlapping_pdb_ids_excluded"] == 9
    assert (
        report["overlap_and_selection"]["overlapping_complexes_excluded"]
        == 11
    )
    assert report["overlap_and_selection"]["selected_after_overlap"] == 97
    assert report["decision"]["antibody_ranking_gate_certified"] is False
    assert report["decision"]["nanobody_ranking_gate_certified"] is False
    assert report["decision"]["external_trust_enabled"] is False
    assert (
        report["locked_fromm_evaluation"]["independent_calibration_gate"][
            "trusted"
        ]
        == 0
    )
    assert public_artifact_issues(rows) == []
    assert public_artifact_issues(report) == []


def test_independent_output_writer_hides_absolute_output_path(tmp_path):
    source_path = tmp_path / "gray.csv"
    contract = _write_gray_fixture(source_path)
    fromm_rows = load_c5_manifest(FROMM_MANIFEST)
    rows, report = build_independent_calibration(
        source_path,
        fromm_rows,
        contract=contract,
        fromm_manifest_sha256=sha256_file(FROMM_MANIFEST),
    )
    manifest_path = tmp_path / "derived.jsonl"
    json_path = tmp_path / "result.json"
    markdown_path = tmp_path / "result.md"

    write_outputs(
        rows,
        report,
        manifest_path=manifest_path,
        json_path=json_path,
        markdown_path=markdown_path,
    )

    saved = json.loads(json_path.read_text())
    assert saved["derived_manifest"]["path"] == "derived.jsonl"
    assert str(tmp_path) not in json_path.read_text()
    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == (
        saved["derived_manifest"]["sha256"]
    )

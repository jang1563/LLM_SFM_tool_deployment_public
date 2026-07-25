import csv
import hashlib
import json
from pathlib import Path

import pytest

from llm_sfm_tool_deployment import Action

from c5_antibody_ood.evaluate_source_pilot import write_outputs
from c5_antibody_ood.manifest import load_c5_manifest
from c5_antibody_ood.source_pilot import (
    FROMM_AF3_SHA256,
    SelectedTarget,
    SourceContract,
    SourceIntakeError,
    SourceSample,
    build_source_pilot,
    intake_source_scores,
    parse_chain_ids,
    public_artifact_issues,
    select_hoeffding_threshold,
    select_target_samples,
    sha256_file,
    validate_source_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
TRACKED_MANIFEST = (
    ROOT / "c5_antibody_ood/c5_source_backed_manifest_v1.jsonl"
)
TRACKED_REPORT = (
    ROOT
    / "c5_antibody_ood/c5_source_backed_pilot_result_2026-07-25.json"
)


def _write_source_fixture(path: Path) -> SourceContract:
    columns = [
        "sample_id",
        "pdbid",
        "Achain",
        "Hchain",
        "Lchain",
        "abag_dockq",
        "iptm",
        "ranking_confidence",
        "preset",
        "reference_pdb",
        "query_pdb",
    ]
    rows = []
    target_values = {
        "7aaa": (0.40, 0.91),
        "7bbb": (0.10, 0.88),
        "7ccc": (0.50, 0.76),
        "7ddd": (0.20, 0.72),
    }
    for target, (dockq, iptm) in target_values.items():
        rows.extend(
            [
                {
                    "sample_id": f"{target}_z",
                    "pdbid": target,
                    "Achain": "C | D",
                    "Hchain": "A",
                    "Lchain": "",
                    "abag_dockq": dockq,
                    "iptm": iptm,
                    "ranking_confidence": 0.90,
                    "preset": "alphafold3",
                    "reference_pdb": "/" + "proj" + "/example/reference.cif",
                    "query_pdb": "/" + "scratch" + "/example/query.cif",
                },
                {
                    "sample_id": f"{target}_a",
                    "pdbid": target,
                    "Achain": "C | D",
                    "Hchain": "A",
                    "Lchain": "",
                    "abag_dockq": dockq,
                    "iptm": iptm,
                    "ranking_confidence": 0.90,
                    "preset": "alphafold3",
                    "reference_pdb": "/" + "proj" + "/example/reference.cif",
                    "query_pdb": "/" + "scratch" + "/example/query.cif",
                },
            ]
        )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return SourceContract(
        expected_sha256=sha256_file(path),
        expected_rows=8,
        expected_targets=4,
        expected_samples_per_target=2,
    )


def _rewrite_fixture(path: Path, transform):
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or ())
        rows = list(reader)
    fieldnames, rows = transform(fieldnames, rows)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_source_intake_projects_allowlisted_fields_and_counts_paths(tmp_path):
    path = tmp_path / "scores.csv"
    contract = _write_source_fixture(path)

    samples, audit = intake_source_scores(path, contract=contract)

    assert len(samples) == 8
    assert audit["source_column_count"] == 11
    assert audit["allowlisted_column_count"] == 9
    assert audit["excluded_column_count"] == 2
    assert audit["excluded_absolute_path_cells"] == 16
    assert audit["absolute_path_values_emitted"] is False
    assert all(not hasattr(sample, "reference_pdb") for sample in samples)
    assert samples[0].antigen_chains == ("C", "D")
    assert samples[0].light_chains == ()


def test_source_intake_fails_closed_on_hash_or_schema_mismatch(tmp_path):
    path = tmp_path / "scores.csv"
    contract = _write_source_fixture(path)
    bad_contract = SourceContract(
        expected_sha256="0" * 64,
        expected_rows=contract.expected_rows,
        expected_targets=contract.expected_targets,
        expected_samples_per_target=contract.expected_samples_per_target,
    )

    with pytest.raises(SourceIntakeError) as exc:
        intake_source_scores(path, contract=bad_contract)

    assert "source_sha256_mismatch" in str(exc.value)


def test_source_intake_rejects_missing_column_and_out_of_range_metric(tmp_path):
    missing_path = tmp_path / "missing.csv"
    _write_source_fixture(missing_path)

    def remove_iptm(fieldnames, rows):
        fieldnames.remove("iptm")
        for row in rows:
            row.pop("iptm")
        return fieldnames, rows

    _rewrite_fixture(missing_path, remove_iptm)
    contract = SourceContract(
        expected_sha256=None,
        expected_rows=8,
        expected_targets=4,
        expected_samples_per_target=2,
    )
    with pytest.raises(SourceIntakeError) as missing_exc:
        intake_source_scores(missing_path, contract=contract)
    assert "source_missing_columns:iptm" in str(missing_exc.value)

    range_path = tmp_path / "range.csv"
    _write_source_fixture(range_path)

    def invalidate_metric(fieldnames, rows):
        rows[0]["iptm"] = "1.2"
        return fieldnames, rows

    _rewrite_fixture(range_path, invalidate_metric)
    with pytest.raises(SourceIntakeError) as range_exc:
        intake_source_scores(range_path, contract=contract)
    assert "iptm_outside_unit_interval" in str(range_exc.value)


def test_source_intake_rejects_per_target_count_and_chain_role_overlap(tmp_path):
    count_path = tmp_path / "count.csv"
    _write_source_fixture(count_path)

    def drop_row(fieldnames, rows):
        return fieldnames, rows[:-1]

    _rewrite_fixture(count_path, drop_row)
    contract = SourceContract(
        expected_sha256=None,
        expected_rows=8,
        expected_targets=4,
        expected_samples_per_target=2,
    )
    with pytest.raises(SourceIntakeError) as count_exc:
        intake_source_scores(count_path, contract=contract)
    assert "source_row_count:7!=8" in str(count_exc.value)
    assert "source_samples_per_target_mismatch:7ddd" in str(count_exc.value)

    chain_path = tmp_path / "chain.csv"
    _write_source_fixture(chain_path)

    def overlap_chain(fieldnames, rows):
        rows[0]["Achain"] = "A"
        return fieldnames, rows

    _rewrite_fixture(chain_path, overlap_chain)
    with pytest.raises(SourceIntakeError) as chain_exc:
        intake_source_scores(chain_path, contract=contract)
    assert "chain_role_overlap" in str(chain_exc.value)


def test_chain_parser_handles_multichain_and_optional_light_chain():
    assert parse_chain_ids("1 | 2 | 3") == ("1", "2", "3")
    assert parse_chain_ids("nan") == ()
    assert parse_chain_ids("") == ()
    with pytest.raises(ValueError, match="duplicate_chain_id"):
        parse_chain_ids("A | A")


def test_target_selection_is_grouped_and_uses_lexical_tiebreak(tmp_path):
    path = tmp_path / "scores.csv"
    contract = _write_source_fixture(path)
    samples, _ = intake_source_scores(path, contract=contract)

    selected = select_target_samples(samples, calibration_targets=2)

    assert len(selected) == 4
    assert {row.split for row in selected} == {"calibration", "evaluation"}
    assert sum(row.split == "calibration" for row in selected) == 2
    assert all(row.sample.sample_id.endswith("_a") for row in selected)
    assert all(row.top_ranking_tie_count == 2 for row in selected)
    calibration = {
        row.sample.complex_id
        for row in selected
        if row.split == "calibration"
    }
    evaluation = {
        row.sample.complex_id
        for row in selected
        if row.split == "evaluation"
    }
    assert calibration.isdisjoint(evaluation)


def test_small_calibration_set_yields_no_hoeffding_certificate():
    rows = [
        SelectedTarget(
            sample=SourceSample(
                sample_id=f"sample_{index}",
                complex_id=f"target_{index}",
                antigen_chains=("C",),
                heavy_chains=("A",),
                light_chains=(),
                dockq=0.50,
                iptm=0.95,
                ranking_confidence=0.95,
                preset="alphafold3",
            ),
            top_ranking_tie_count=1,
            split="calibration",
        )
        for index in range(5)
    ]

    certificate = select_hoeffding_threshold(rows, alpha=0.30)

    assert certificate["certified"] is False
    assert certificate["threshold"] is None
    assert certificate["calibration_trusted"] == 0


def test_fixture_pilot_builds_canonical_fail_closed_rows(tmp_path):
    path = tmp_path / "scores.csv"
    contract = _write_source_fixture(path)

    rows, report = build_source_pilot(
        path,
        contract=contract,
        calibration_targets=2,
    )

    assert validate_source_manifest(
        rows,
        expected_rows=4,
        expected_calibration_rows=2,
    ) == []
    assert report["decision"]["source_backed_pilot_passed"] is True
    assert report["decision"]["regime_specific_trust_certified"] is False
    assert all(
        row["hidden_eval_metadata"]["expected_terminal_action"]
        == Action.VERIFY_WITH_ASSAY_OR_DATABASE.value
        for row in rows
    )
    rendered = json.dumps({"rows": rows, "report": report}, sort_keys=True)
    assert "/" + "proj" + "/" not in rendered
    assert "/" + "scratch" + "/" not in rendered
    assert "reference_pdb" not in rendered
    assert "query_pdb" not in rendered
    assert "_a\"" not in rendered


def test_tracked_source_manifest_and_compact_report_are_reproducible():
    rows = load_c5_manifest(TRACKED_MANIFEST)
    report = json.loads(TRACKED_REPORT.read_text())

    assert len(rows) == 110
    assert validate_source_manifest(rows) == []
    assert report["source"]["sha256"] == FROMM_AF3_SHA256
    assert report["source"]["archive_member_byte_identity_verified"] is True
    assert report["manifest"]["validation_issues"] == []
    assert report["split"]["calibration_targets"] == 55
    assert report["split"]["evaluation_targets"] == 55
    assert report["policies"]["trust_all"]["failures_among_trusted"] == 28
    assert (
        report["policies"]["generic_fixed_iptm_0_80"][
            "failures_among_trusted"
        ]
        == 3
    )
    assert (
        report["policies"]["regime_specific_hoeffding"]["certified"]
        is False
    )
    assert public_artifact_issues(rows) == []
    assert public_artifact_issues(report) == []


def test_output_writer_never_publishes_absolute_local_output_path(tmp_path):
    source_path = tmp_path / "scores.csv"
    contract = _write_source_fixture(source_path)
    rows, report = build_source_pilot(
        source_path,
        contract=contract,
        calibration_targets=2,
    )
    out_manifest = tmp_path / "derived.jsonl"
    out_json = tmp_path / "result.json"
    out_md = tmp_path / "result.md"

    write_outputs(
        rows,
        report,
        manifest_path=out_manifest,
        json_path=out_json,
        markdown_path=out_md,
    )

    saved = json.loads(out_json.read_text())
    assert saved["derived_manifest"]["path"] == "derived.jsonl"
    assert str(tmp_path) not in out_json.read_text()
    assert hashlib.sha256(out_manifest.read_bytes()).hexdigest() == (
        saved["derived_manifest"]["sha256"]
    )

import json
import os
import subprocess
from pathlib import Path

import pytest

from c5_antibody_ood.af3_preflight import (
    AF3PreflightError,
    REQUIRED_DATABASE_ENTRIES,
    build_database_inventory,
    public_readiness_summary,
    run_preflight,
    verify_attestation,
    verify_runtime_dependencies,
)
from c5_antibody_ood.manifest import load_c5_manifest, write_c5_manifest
from c5_antibody_ood.prospective_panel import canonical_sha256
from c5_antibody_ood.source_pilot import sha256_file


ROOT = Path(__file__).resolve().parents[1]
PREREGISTRATION = (
    ROOT
    / "c5_antibody_ood/c5_prospective_panel_preregistration_v1.json"
)
INPUT_FREEZE = (
    ROOT
    / "c5_antibody_ood/c5_sabdab2_prospective_af3_input_freeze_2026-07-25.json"
)
RETAINED_MANIFEST = (
    ROOT
    / "c5_antibody_ood/c5_sabdab2_prospective_retained_manifest_v1.jsonl"
)
SBATCH = ROOT / "c5_antibody_ood/run_c5_af3_cayuga.sbatch"


def _build_fixture(tmp_path: Path) -> dict:
    preregistration = json.loads(PREREGISTRATION.read_text())
    rows = load_c5_manifest(RETAINED_MANIFEST)[:2]
    retained_manifest = tmp_path / "retained.jsonl"
    write_c5_manifest(retained_manifest, rows)

    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    input_hashes = {}
    for row in rows:
        path = input_dir / f"{row['instance_id'].lower()}.json"
        path.write_text('{"dialect":"alphafold3"}\n')
        input_hashes[row["target_id"]] = sha256_file(path)
    input_freeze = {
        "preregistration_id": preregistration["preregistration_id"],
        "protocol_sha256": preregistration["commitment"]["protocol_sha256"],
        "retention": {
            "manifest_sha256": canonical_sha256(rows),
        },
        "af3_inputs": {
            "files": 2,
            "af3_input_set_sha256": canonical_sha256(input_hashes),
        },
        "decision": {
            "ready_for_af3_prediction": True,
        },
    }

    source_dir = tmp_path / "source"
    source_dir.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=source_dir, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.org"],
        cwd=source_dir,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=source_dir,
        check=True,
    )
    (source_dir / "README").write_text("fixture\n")
    subprocess.run(["git", "add", "README"], cwd=source_dir, check=True)
    subprocess.run(
        ["git", "commit", "-qm", "fixture"],
        cwd=source_dir,
        check=True,
    )
    subprocess.run(
        ["git", "tag", preregistration["protocol"]["prediction"]["code_tag"]],
        cwd=source_dir,
        check=True,
    )

    container = tmp_path / "alphafold3.sif"
    container.write_bytes(b"container")
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    model = model_dir / "af3.bin.zst"
    model.write_bytes(b"parameters")
    database_dir = tmp_path / "databases"
    database_dir.mkdir()
    for entry in REQUIRED_DATABASE_ENTRIES:
        path = database_dir / entry
        if entry == "mmcif_files":
            path.mkdir()
            (path / "fixture.cif").write_text("data_fixture\n")
        else:
            path.write_text(">fixture\nACDE\n")
    database_manifest = tmp_path / "database_manifest.json"
    database_manifest.write_text(
        json.dumps(
            build_database_inventory(database_dir),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    output_dir = tmp_path / "outputs"
    return {
        "preregistration": preregistration,
        "input_freeze": input_freeze,
        "retained_manifest": retained_manifest,
        "input_dir": input_dir,
        "source_dir": source_dir,
        "container": container,
        "expected_container_sha256": sha256_file(container),
        "model_dir": model_dir,
        "expected_model_sha256": canonical_sha256(
            [
                {
                    "name": model.name,
                    "bytes": model.stat().st_size,
                    "sha256": sha256_file(model),
                }
            ]
        ),
        "database_dir": database_dir,
        "database_manifest": database_manifest,
        "expected_database_manifest_sha256": sha256_file(database_manifest),
        "output_dir": output_dir,
        "runtime_command": "git",
    }


def test_preflight_passes_only_with_all_locks(monkeypatch, tmp_path):
    fixture = _build_fixture(tmp_path)
    monkeypatch.setattr(
        "c5_antibody_ood.af3_preflight.AF3_COMMIT",
        subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=fixture["source_dir"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
    )

    result = run_preflight(**fixture)

    assert result["ready_for_af3_prediction"] is True
    assert result["violations"] == []
    assert all(result["components"].values())
    rendered = json.dumps(result, sort_keys=True)
    assert str(tmp_path) not in rendered
    assert "af3.bin.zst" not in rendered


def test_missing_parameters_fail_closed_with_specific_violations(
    monkeypatch,
    tmp_path,
):
    fixture = _build_fixture(tmp_path)
    monkeypatch.setattr(
        "c5_antibody_ood.af3_preflight.AF3_COMMIT",
        subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=fixture["source_dir"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
    )
    fixture["model_dir"].rename(tmp_path / "models_unavailable")

    result = run_preflight(**fixture)

    assert result["ready_for_af3_prediction"] is False
    assert "model_parameters_present" in result["violations"]
    assert "single_model_parameter_set" in result["violations"]
    assert "model_checksum_matches" in result["violations"]


def test_preexisting_output_and_checksum_mismatch_fail_closed(
    monkeypatch,
    tmp_path,
):
    fixture = _build_fixture(tmp_path)
    monkeypatch.setattr(
        "c5_antibody_ood.af3_preflight.AF3_COMMIT",
        subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=fixture["source_dir"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
    )
    fixture["output_dir"].mkdir()
    fixture["expected_container_sha256"] = "0" * 64

    result = run_preflight(**fixture)

    assert result["ready_for_af3_prediction"] is False
    assert "container_checksum_matches" in result["violations"]
    assert "output_boundary_clean" in result["violations"]


def test_database_inventory_drift_fails_closed(monkeypatch, tmp_path):
    fixture = _build_fixture(tmp_path)
    monkeypatch.setattr(
        "c5_antibody_ood.af3_preflight.AF3_COMMIT",
        subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=fixture["source_dir"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
    )
    database_file = (
        fixture["database_dir"] / "bfd-first_non_consensus_sequences.fasta"
    )
    database_file.write_text(">fixture\nACDEFG\n")

    result = run_preflight(**fixture)

    assert result["ready_for_af3_prediction"] is False
    assert "database_inventory_matches" in result["violations"]


def test_public_summary_exposes_checksums_but_no_paths(monkeypatch, tmp_path):
    fixture = _build_fixture(tmp_path)
    monkeypatch.setattr(
        "c5_antibody_ood.af3_preflight.AF3_COMMIT",
        subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=fixture["source_dir"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
    )
    attestation = run_preflight(**fixture)

    summary = public_readiness_summary(attestation)

    assert summary["ready_for_af3_prediction"] is True
    rendered = json.dumps(summary, sort_keys=True)
    assert str(tmp_path) not in rendered
    assert "af3.bin.zst" not in rendered


def test_private_attestation_verification_is_checksum_bound(
    monkeypatch,
    tmp_path,
):
    fixture = _build_fixture(tmp_path)
    monkeypatch.setattr(
        "c5_antibody_ood.af3_preflight.AF3_COMMIT",
        subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=fixture["source_dir"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
    )
    attestation = run_preflight(**fixture)
    path = tmp_path / "attestation.json"
    path.write_text(json.dumps(attestation, sort_keys=True))

    result = verify_attestation(
        attestation_path=path,
        expected_attestation_sha256=sha256_file(path),
        preregistration=fixture["preregistration"],
        input_freeze=fixture["input_freeze"],
        retained_manifest=fixture["retained_manifest"],
        input_dir=fixture["input_dir"],
    )

    assert result["attestation_verified"] is True
    with pytest.raises(AF3PreflightError, match="checksum_mismatch"):
        verify_attestation(
            attestation_path=path,
            expected_attestation_sha256="0" * 64,
            preregistration=fixture["preregistration"],
            input_freeze=fixture["input_freeze"],
            retained_manifest=fixture["retained_manifest"],
            input_dir=fixture["input_dir"],
        )


def test_private_attestation_rejects_runtime_input_drift(
    monkeypatch,
    tmp_path,
):
    fixture = _build_fixture(tmp_path)
    monkeypatch.setattr(
        "c5_antibody_ood.af3_preflight.AF3_COMMIT",
        subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=fixture["source_dir"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
    )
    attestation = run_preflight(**fixture)
    path = tmp_path / "attestation.json"
    path.write_text(json.dumps(attestation, sort_keys=True))
    next(fixture["input_dir"].glob("*.json")).write_text(
        '{"dialect":"changed"}\n'
    )

    with pytest.raises(AF3PreflightError, match="runtime_input_set_mismatch"):
        verify_attestation(
            attestation_path=path,
            expected_attestation_sha256=sha256_file(path),
            preregistration=fixture["preregistration"],
            input_freeze=fixture["input_freeze"],
            retained_manifest=fixture["retained_manifest"],
            input_dir=fixture["input_dir"],
        )


def test_runtime_dependency_verification_passes_quick_and_full(
    monkeypatch,
    tmp_path,
):
    fixture = _build_fixture(tmp_path)
    monkeypatch.setattr(
        "c5_antibody_ood.af3_preflight.AF3_COMMIT",
        subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=fixture["source_dir"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
    )
    attestation = run_preflight(**fixture)
    path = tmp_path / "attestation.json"
    path.write_text(json.dumps(attestation, sort_keys=True))
    common = {
        "attestation_path": path,
        "expected_attestation_sha256": sha256_file(path),
        "preregistration": fixture["preregistration"],
        "input_freeze": fixture["input_freeze"],
        "retained_manifest": fixture["retained_manifest"],
        "input_dir": fixture["input_dir"],
        "container": fixture["container"],
        "model_dir": fixture["model_dir"],
        "database_dir": fixture["database_dir"],
        "database_manifest": fixture["database_manifest"],
    }

    quick = verify_runtime_dependencies(**common, mode="quick")
    full = verify_runtime_dependencies(**common, mode="full")

    assert quick["runtime_dependencies_verified"] is True
    assert quick["verification_mode"] == "quick"
    assert full["runtime_dependencies_verified"] is True
    assert full["verification_mode"] == "full"
    assert all(full["components"].values())
    rendered = json.dumps(full, sort_keys=True)
    assert str(tmp_path) not in rendered
    assert "af3.bin.zst" not in rendered


def test_runtime_quick_catches_database_drift_and_full_catches_content_drift(
    monkeypatch,
    tmp_path,
):
    fixture = _build_fixture(tmp_path)
    monkeypatch.setattr(
        "c5_antibody_ood.af3_preflight.AF3_COMMIT",
        subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=fixture["source_dir"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
    )
    attestation = run_preflight(**fixture)
    path = tmp_path / "attestation.json"
    path.write_text(json.dumps(attestation, sort_keys=True))
    common = {
        "attestation_path": path,
        "expected_attestation_sha256": sha256_file(path),
        "preregistration": fixture["preregistration"],
        "input_freeze": fixture["input_freeze"],
        "retained_manifest": fixture["retained_manifest"],
        "input_dir": fixture["input_dir"],
        "container": fixture["container"],
        "model_dir": fixture["model_dir"],
        "database_dir": fixture["database_dir"],
        "database_manifest": fixture["database_manifest"],
    }
    database_file = (
        fixture["database_dir"] / "bfd-first_non_consensus_sequences.fasta"
    )
    database_stat = database_file.stat()
    database_content = database_file.read_text()
    database_file.write_text(">fixture\nACDEFG\n")

    with pytest.raises(
        AF3PreflightError,
        match="database_quick_identity_matches",
    ):
        verify_runtime_dependencies(**common, mode="quick")

    database_file.write_text(database_content)
    os.utime(
        database_file,
        ns=(database_stat.st_atime_ns, database_stat.st_mtime_ns),
    )
    database_file.write_text(database_content.replace("ACDE", "ACDF"))
    os.utime(
        database_file,
        ns=(database_stat.st_atime_ns, database_stat.st_mtime_ns),
    )
    quick = verify_runtime_dependencies(**common, mode="quick")
    assert quick["runtime_dependencies_verified"] is True
    with pytest.raises(
        AF3PreflightError,
        match="database_full_inventory_matches",
    ):
        verify_runtime_dependencies(**common, mode="full")

    database_file.write_text(database_content)
    os.utime(
        database_file,
        ns=(database_stat.st_atime_ns, database_stat.st_mtime_ns),
    )
    container_stat = fixture["container"].stat()
    fixture["container"].write_bytes(b"containar")
    os.utime(
        fixture["container"],
        ns=(container_stat.st_atime_ns, container_stat.st_mtime_ns),
    )
    quick = verify_runtime_dependencies(**common, mode="quick")
    assert quick["runtime_dependencies_verified"] is True
    with pytest.raises(
        AF3PreflightError,
        match="container_content_matches",
    ):
        verify_runtime_dependencies(**common, mode="full")


def test_cayuga_array_is_attestation_and_output_guarded():
    script = SBATCH.read_text()

    assert "#SBATCH --array=0-119%8" in script
    assert "c5_antibody_ood.af3_preflight verify-runtime" in script
    assert "--expected-attestation-sha256" in script
    assert "--retained-manifest" in script
    assert "--input-dir" in script
    assert 'if [ -e "${TARGET_OUTPUT}" ]' in script
    assert "--num_diffusion_samples=5" in script
    assert "--output_dir=/root/af_output" in script
    assert "AF3_DB_MANIFEST" in script
    assert "--database-manifest /root/c5_database_manifest.json" in script
    assert "--pwd /app/alphafold" in script
    assert "uv run python3 -m c5_antibody_ood.af3_preflight" in script
    assert "uv run python3 run_alphafold.py" in script
    assert "\npython -m c5_antibody_ood.af3_preflight" not in script
    assert "/Users/" not in script
    assert "/home/" not in script
    assert "/" + "scratch/" not in script

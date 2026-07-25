"""Fail-closed AlphaFold 3 environment attestation for prospective C5."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .manifest import load_c5_manifest
from .prospective_inputs import _safe_job_name
from .prospective_panel import (
    AF3_COMMIT,
    canonical_sha256,
    validate_preregistration,
    write_json,
)
from .source_pilot import sha256_file


ATTESTATION_SCHEMA = "c5_af3_environment_attestation_v1"
READINESS_SCHEMA = "c5_af3_environment_readiness_v1"
DATABASE_INVENTORY_SCHEMA = "c5_af3_database_inventory_v3"
REQUIRED_DATABASE_ENTRIES = (
    "bfd-first_non_consensus_sequences.fasta",
    "mgy_clusters_2022_05.fa",
    "uniprot_all_2021_04.fa",
    "uniref90_2022_05.fa",
    "nt_rna_2023_02_23_clust_seq_id_90_cov_80_rep_seq.fasta",
    "rfam_14_9_clust_seq_id_90_cov_80_rep_seq.fasta",
    "rnacentral_active_seq_id_90_cov_80_linclust.fasta",
    "mmcif_files",
    "pdb_seqres_2022_09_28.fasta",
)
MODEL_PATTERNS = tuple(
    re.compile(pattern)
    for pattern in (
        r"(?P<model_name>.*)\.[0-9]+\.bin\.zst$",
        r"(?P<model_name>.*)\.bin\.zst\.[0-9]+$",
        r"(?P<model_name>.*)\.[0-9]+\.bin$",
        r"(?P<model_name>.*)\.bin]\.[0-9]+$",
        r"(?P<model_name>.*)\.bin\.zst$",
        r"(?P<model_name>.*)\.bin$",
    )
)
SHA256_RE = re.compile(r"^[a-f0-9]{64}$")


class AF3PreflightError(ValueError):
    """Raised when an AF3 preflight or attestation contract is malformed."""


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _git_value(source_dir: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(source_dir), *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _aggregate_file_set(files: Sequence[Path]) -> tuple[str, int]:
    records = _file_set_records(files)
    return (
        canonical_sha256(records),
        sum(record["bytes"] for record in records),
    )


def _file_set_records(files: Sequence[Path]) -> list[dict[str, Any]]:
    records = []
    for path in sorted(files, key=lambda item: item.name):
        size = path.stat().st_size
        records.append(
            {
                "name": path.name,
                "bytes": size,
                "sha256": sha256_file(path),
            }
        )
    return records


def _quick_file_set_records(files: Sequence[Path]) -> list[dict[str, Any]]:
    return [
        {
            "name": path.name,
            "bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }
        for path in sorted(files, key=lambda item: item.name)
    ]


def _select_model_files(model_dir: Path) -> tuple[list[Path], bool]:
    if not model_dir.is_dir():
        return [], False
    files = [path for path in model_dir.iterdir() if path.is_file()]
    for pattern in MODEL_PATTERNS:
        models: dict[str, list[Path]] = {}
        for path in files:
            match = pattern.fullmatch(path.name)
            if match:
                models.setdefault(match.group("model_name"), []).append(path)
        if models:
            if len(models) != 1:
                return [], True
            return sorted(next(iter(models.values()))), False
    return [], False


def _database_entry_ready(path: Path) -> bool:
    try:
        if path.is_file():
            return path.stat().st_size > 0
        if path.is_dir():
            return next(
                (True for child in path.iterdir() if child.is_file()),
                False,
            )
    except OSError:
        return False
    return False


def _database_entry_inventory(path: Path) -> dict[str, Any]:
    if path.is_file():
        records = [
            {
                "relative_path": ".",
                "bytes": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
                "sha256": sha256_file(path),
            }
        ]
        kind = "file"
    elif path.is_dir():
        records = [
            {
                "relative_path": child.relative_to(path).as_posix(),
                "bytes": child.stat().st_size,
                "mtime_ns": child.stat().st_mtime_ns,
                "sha256": sha256_file(child),
            }
            for child in sorted(path.rglob("*"))
            if child.is_file()
        ]
        kind = "directory"
    else:
        raise AF3PreflightError("database_required_entry_missing")
    if not records or any(record["bytes"] <= 0 for record in records):
        raise AF3PreflightError("database_required_entry_empty")
    sentinel_indexes = sorted({0, len(records) // 2, len(records) - 1})
    return {
        "kind": kind,
        "files": len(records),
        "bytes": sum(record["bytes"] for record in records),
        "relative_content_inventory_sha256": canonical_sha256(records),
        "sentinels": [records[index] for index in sentinel_indexes],
    }


def build_database_inventory(database_dir: str | Path) -> dict[str, Any]:
    """Build a path-free stat inventory for the required AF3 databases."""

    root = Path(database_dir)
    if not root.is_dir():
        raise AF3PreflightError("database_directory_missing")
    entries = {
        entry: _database_entry_inventory(root / entry)
        for entry in REQUIRED_DATABASE_ENTRIES
    }
    return {
        "schema_version": DATABASE_INVENTORY_SCHEMA,
        "required_entries": list(REQUIRED_DATABASE_ENTRIES),
        "entries": entries,
        "summary": {
            "entries": len(entries),
            "files": sum(value["files"] for value in entries.values()),
            "bytes": sum(value["bytes"] for value in entries.values()),
        },
        "local_paths_emitted": False,
        "content_bytes_emitted": False,
        "file_content_sha256_emitted": True,
    }


def _database_inventory_matches(
    database_dir: Path,
    manifest_path: Path,
) -> bool:
    try:
        expected = json.loads(manifest_path.read_text())
        actual = build_database_inventory(database_dir)
    except (OSError, ValueError, json.JSONDecodeError, AF3PreflightError):
        return False
    return expected == actual


def _quick_database_inventory_matches(
    database_dir: Path,
    manifest_path: Path,
) -> bool:
    """Check the frozen manifest and deterministic file sentinels cheaply."""

    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    if (
        manifest.get("schema_version") != DATABASE_INVENTORY_SCHEMA
        or manifest.get("required_entries") != list(REQUIRED_DATABASE_ENTRIES)
    ):
        return False
    entries = manifest.get("entries")
    if not isinstance(entries, Mapping):
        return False
    for name in REQUIRED_DATABASE_ENTRIES:
        expected = entries.get(name)
        if not isinstance(expected, Mapping):
            return False
        root = database_dir / name
        if expected.get("kind") == "file":
            if (
                not root.is_file()
                or root.stat().st_size != expected.get("bytes")
                or expected.get("files") != 1
            ):
                return False
        elif expected.get("kind") == "directory":
            if not root.is_dir():
                return False
        else:
            return False
        sentinels = expected.get("sentinels")
        if not isinstance(sentinels, list) or not sentinels:
            return False
        for sentinel in sentinels:
            if not isinstance(sentinel, Mapping):
                return False
            relative = sentinel.get("relative_path")
            size = sentinel.get("bytes")
            mtime_ns = sentinel.get("mtime_ns")
            if (
                not isinstance(relative, str)
                or not relative
                or Path(relative).is_absolute()
                or ".." in Path(relative).parts
                or not isinstance(size, int)
                or isinstance(size, bool)
                or size <= 0
                or not isinstance(mtime_ns, int)
                or isinstance(mtime_ns, bool)
                or mtime_ns <= 0
            ):
                return False
            path = root if relative == "." else root / relative
            if (
                not path.is_file()
                or path.stat().st_size != size
                or path.stat().st_mtime_ns != mtime_ns
            ):
                return False
    return True


def _input_set_commitment(
    retained_manifest: Path,
    input_dir: Path,
) -> tuple[str | None, str | None, int, list[str]]:
    violations: list[str] = []
    try:
        rows = load_c5_manifest(retained_manifest)
    except (OSError, ValueError, json.JSONDecodeError):
        return None, None, 0, ["retained_manifest_invalid"]
    manifest_sha256 = canonical_sha256(rows)

    expected: dict[str, Path] = {}
    try:
        for row in rows:
            target_id = str(row["target_id"])
            name = _safe_job_name(str(row["instance_id"]))
            expected[target_id] = input_dir / f"{name}.json"
    except (KeyError, AF3PreflightError, ValueError):
        return (
            None,
            manifest_sha256,
            len(rows),
            ["retained_manifest_invalid"],
        )
    if len(expected) != len(rows):
        violations.append("retained_manifest_duplicate_target")

    if not input_dir.is_dir():
        return (
            None,
            manifest_sha256,
            len(rows),
            [*violations, "af3_input_directory_missing"],
        )
    present = set(input_dir.glob("*.json"))
    expected_paths = set(expected.values())
    if present != expected_paths:
        violations.append("af3_input_file_set_mismatch")
    if any(not path.is_file() or path.stat().st_size == 0 for path in expected_paths):
        violations.append("af3_input_missing_or_empty")
    if violations:
        return None, manifest_sha256, len(rows), sorted(set(violations))
    digest = canonical_sha256(
        {
            target_id: sha256_file(path)
            for target_id, path in sorted(expected.items())
        }
    )
    return digest, manifest_sha256, len(rows), []


def run_preflight(
    *,
    preregistration: Mapping[str, Any],
    input_freeze: Mapping[str, Any],
    retained_manifest: str | Path,
    input_dir: str | Path,
    source_dir: str | Path,
    container: str | Path,
    expected_container_sha256: str,
    model_dir: str | Path,
    expected_model_sha256: str,
    database_dir: str | Path,
    database_manifest: str | Path,
    expected_database_manifest_sha256: str,
    output_dir: str | Path,
    runtime_command: str = "singularity",
    resume: bool = False,
) -> dict[str, Any]:
    """Inspect all locked AF3 dependencies and return a path-free attestation."""

    for label, value in (
        ("expected_container_sha256", expected_container_sha256),
        ("expected_model_sha256", expected_model_sha256),
        (
            "expected_database_manifest_sha256",
            expected_database_manifest_sha256,
        ),
    ):
        if not SHA256_RE.fullmatch(value):
            raise AF3PreflightError(f"{label}_invalid")

    preregistration_issues = validate_preregistration(preregistration)
    if preregistration_issues:
        raise AF3PreflightError(
            "preregistration_invalid:" + ",".join(preregistration_issues)
        )
    protocol_sha256 = preregistration["commitment"]["protocol_sha256"]
    input_freeze_protocol_matches = (
        input_freeze.get("protocol_sha256") == protocol_sha256
    )
    input_freeze_preregistration_matches = (
        input_freeze.get("preregistration_id")
        == preregistration["preregistration_id"]
    )
    input_freeze_ready = (
        input_freeze.get("decision", {}).get("ready_for_af3_prediction")
        is True
    )
    expected_input_sha256 = input_freeze["af3_inputs"][
        "af3_input_set_sha256"
    ]
    expected_input_count = int(input_freeze["af3_inputs"]["files"])
    expected_retained_manifest_sha256 = input_freeze["retention"][
        "manifest_sha256"
    ]

    source_path = Path(source_dir)
    actual_commit = _git_value(source_path, "rev-parse", "HEAD")
    actual_tag = _git_value(source_path, "describe", "--tags", "--exact-match")
    expected_tag = preregistration["protocol"]["prediction"]["code_tag"]

    container_path = Path(container)
    container_present = (
        container_path.is_file() and container_path.stat().st_size > 0
    )
    container_sha256 = (
        sha256_file(container_path) if container_present else None
    )

    model_files, multiple_models = _select_model_files(Path(model_dir))
    model_present = bool(model_files)
    model_sha256: str | None = None
    model_bytes = 0
    model_records: list[dict[str, Any]] = []
    if model_present:
        model_records = _file_set_records(model_files)
        model_sha256 = canonical_sha256(model_records)
        model_bytes = sum(record["bytes"] for record in model_records)

    database_root = Path(database_dir)
    missing_database_entries = [
        entry
        for entry in REQUIRED_DATABASE_ENTRIES
        if not _database_entry_ready(database_root / entry)
    ]
    database_entries_complete = (
        database_root.is_dir() and not missing_database_entries
    )
    database_manifest_path = Path(database_manifest)
    database_manifest_present = (
        database_manifest_path.is_file()
        and database_manifest_path.stat().st_size > 0
    )
    database_manifest_sha256 = (
        sha256_file(database_manifest_path)
        if database_manifest_present
        else None
    )
    database_inventory_matches = (
        database_entries_complete
        and database_manifest_present
        and _database_inventory_matches(
            database_root,
            database_manifest_path,
        )
    )

    (
        input_sha256,
        retained_manifest_sha256,
        input_count,
        input_violations,
    ) = _input_set_commitment(
        Path(retained_manifest),
        Path(input_dir),
    )
    output_path = Path(output_dir)
    output_boundary_clean = resume or not output_path.exists()

    components = {
        "runtime_available": shutil.which(runtime_command) is not None,
        "source_commit_matches": actual_commit == AF3_COMMIT,
        "source_tag_matches": actual_tag == expected_tag,
        "container_present": container_present,
        "container_checksum_matches": (
            container_sha256 == expected_container_sha256
        ),
        "model_parameters_present": model_present,
        "single_model_parameter_set": model_present and not multiple_models,
        "model_checksum_matches": model_sha256 == expected_model_sha256,
        "database_entries_complete": database_entries_complete,
        "database_manifest_present": database_manifest_present,
        "database_manifest_checksum_matches": (
            database_manifest_sha256
            == expected_database_manifest_sha256
        ),
        "database_inventory_matches": database_inventory_matches,
        "input_freeze_protocol_matches": input_freeze_protocol_matches,
        "input_freeze_preregistration_matches": (
            input_freeze_preregistration_matches
        ),
        "input_freeze_ready": input_freeze_ready,
        "retained_manifest_checksum_matches": (
            retained_manifest_sha256 == expected_retained_manifest_sha256
        ),
        "input_set_complete": (
            input_count == expected_input_count and not input_violations
        ),
        "input_set_checksum_matches": input_sha256 == expected_input_sha256,
        "output_boundary_clean": output_boundary_clean,
    }
    violations = [name for name, passed in components.items() if not passed]
    violations.extend(input_violations)
    if missing_database_entries:
        violations.append("database_required_entry_missing")
    violations = sorted(set(violations))

    return {
        "schema_version": ATTESTATION_SCHEMA,
        "created_at_utc": _utc_timestamp(),
        "preregistration_id": preregistration["preregistration_id"],
        "protocol_sha256": protocol_sha256,
        "ready_for_af3_prediction": not violations,
        "components": components,
        "checksums": {
            "source_commit": actual_commit,
            "container_sha256": container_sha256,
            "model_parameter_set_sha256": model_sha256,
            "database_manifest_sha256": database_manifest_sha256,
            "retained_manifest_sha256": retained_manifest_sha256,
            "af3_input_set_sha256": input_sha256,
        },
        "counts": {
            "container_bytes": (
                container_path.stat().st_size if container_present else 0
            ),
            "model_parameter_files": len(model_files),
            "model_parameter_bytes": model_bytes,
            "database_required_entries": len(REQUIRED_DATABASE_ENTRIES),
            "database_missing_entries": len(missing_database_entries),
            "af3_input_files": input_count,
        },
        "runtime_identity": {
            "container": {
                "bytes": (
                    container_path.stat().st_size if container_present else 0
                ),
                "mtime_ns": (
                    container_path.stat().st_mtime_ns
                    if container_present
                    else 0
                ),
                "sha256": container_sha256,
            },
            "model_parameter_quick_sha256": canonical_sha256(
                _quick_file_set_records(model_files)
            ),
            "database_inventory_schema": DATABASE_INVENTORY_SCHEMA,
        },
        "resume_requested": resume,
        "violations": violations,
        "release_boundary": {
            "local_paths_emitted": False,
            "model_filenames_emitted": False,
            "database_filenames_emitted": False,
            "raw_sequences_emitted": False,
            "parameter_or_database_bytes_emitted": False,
        },
    }


def public_readiness_summary(attestation: Mapping[str, Any]) -> dict[str, Any]:
    """Project a private preflight result into a public-safe compact summary."""

    if attestation.get("schema_version") != ATTESTATION_SCHEMA:
        raise AF3PreflightError("attestation_schema_invalid")
    return {
        "schema_version": READINESS_SCHEMA,
        "created_at_utc": attestation["created_at_utc"],
        "preregistration_id": attestation["preregistration_id"],
        "protocol_sha256": attestation["protocol_sha256"],
        "ready_for_af3_prediction": attestation["ready_for_af3_prediction"],
        "components": dict(attestation["components"]),
        "checksums": dict(attestation["checksums"]),
        "counts": dict(attestation["counts"]),
        "violations": list(attestation["violations"]),
        "release_boundary": dict(attestation["release_boundary"]),
    }


def verify_attestation(
    *,
    attestation_path: str | Path,
    expected_attestation_sha256: str,
    preregistration: Mapping[str, Any],
    input_freeze: Mapping[str, Any],
    retained_manifest: str | Path,
    input_dir: str | Path,
) -> dict[str, Any]:
    """Verify a passed private attestation before an array task starts."""

    preregistration_issues = validate_preregistration(preregistration)
    if preregistration_issues:
        raise AF3PreflightError(
            "preregistration_invalid:" + ",".join(preregistration_issues)
        )
    if input_freeze.get("preregistration_id") != preregistration.get(
        "preregistration_id"
    ):
        raise AF3PreflightError("input_freeze_preregistration_mismatch")
    if input_freeze.get("protocol_sha256") != preregistration[
        "commitment"
    ].get("protocol_sha256"):
        raise AF3PreflightError("input_freeze_protocol_mismatch")
    if input_freeze.get("decision", {}).get(
        "ready_for_af3_prediction"
    ) is not True:
        raise AF3PreflightError("input_freeze_not_ready")
    if not SHA256_RE.fullmatch(expected_attestation_sha256):
        raise AF3PreflightError("expected_attestation_sha256_invalid")
    path = Path(attestation_path)
    if not path.is_file():
        raise AF3PreflightError("attestation_missing")
    if sha256_file(path) != expected_attestation_sha256:
        raise AF3PreflightError("attestation_checksum_mismatch")
    value = json.loads(path.read_text())
    if value.get("schema_version") != ATTESTATION_SCHEMA:
        raise AF3PreflightError("attestation_schema_invalid")
    if value.get("ready_for_af3_prediction") is not True:
        raise AF3PreflightError("attestation_not_ready")
    if value.get("violations") != []:
        raise AF3PreflightError("attestation_has_violations")
    if value.get("preregistration_id") != preregistration.get(
        "preregistration_id"
    ):
        raise AF3PreflightError("attestation_preregistration_mismatch")
    if value.get("protocol_sha256") != preregistration["commitment"].get(
        "protocol_sha256"
    ):
        raise AF3PreflightError("attestation_protocol_mismatch")
    if value["checksums"].get("source_commit") != AF3_COMMIT:
        raise AF3PreflightError("attestation_source_commit_mismatch")
    if value["checksums"].get(
        "af3_input_set_sha256"
    ) != input_freeze["af3_inputs"].get("af3_input_set_sha256"):
        raise AF3PreflightError("attestation_input_set_mismatch")
    input_sha256, manifest_sha256, input_count, input_issues = (
        _input_set_commitment(
            Path(retained_manifest),
            Path(input_dir),
        )
    )
    if input_issues:
        raise AF3PreflightError(
            "runtime_input_invalid:" + ",".join(input_issues)
        )
    if input_count != input_freeze["af3_inputs"].get("files"):
        raise AF3PreflightError("runtime_input_count_mismatch")
    if input_sha256 != input_freeze["af3_inputs"].get(
        "af3_input_set_sha256"
    ):
        raise AF3PreflightError("runtime_input_set_mismatch")
    if manifest_sha256 != input_freeze["retention"].get(
        "manifest_sha256"
    ):
        raise AF3PreflightError("runtime_retained_manifest_mismatch")
    return {
        "attestation_verified": True,
        "attestation_sha256": expected_attestation_sha256,
        "protocol_sha256": value["protocol_sha256"],
        "af3_input_set_sha256": value["checksums"][
            "af3_input_set_sha256"
        ],
    }


def verify_runtime_dependencies(
    *,
    attestation_path: str | Path,
    expected_attestation_sha256: str,
    preregistration: Mapping[str, Any],
    input_freeze: Mapping[str, Any],
    retained_manifest: str | Path,
    input_dir: str | Path,
    container: str | Path,
    model_dir: str | Path,
    database_dir: str | Path,
    database_manifest: str | Path,
    mode: str = "quick",
) -> dict[str, Any]:
    """Bind runtime dependency paths to the passed private attestation."""

    if mode not in {"quick", "full"}:
        raise AF3PreflightError("runtime_verification_mode_invalid")
    base = verify_attestation(
        attestation_path=attestation_path,
        expected_attestation_sha256=expected_attestation_sha256,
        preregistration=preregistration,
        input_freeze=input_freeze,
        retained_manifest=retained_manifest,
        input_dir=input_dir,
    )
    value = _load_json(attestation_path)
    identity = value.get("runtime_identity")
    if not isinstance(identity, Mapping):
        raise AF3PreflightError("attestation_runtime_identity_missing")
    expected_container = identity.get("container")
    expected_model_quick_sha256 = identity.get(
        "model_parameter_quick_sha256"
    )
    if (
        not isinstance(expected_container, Mapping)
        or not isinstance(expected_model_quick_sha256, str)
        or not SHA256_RE.fullmatch(expected_model_quick_sha256)
        or identity.get("database_inventory_schema")
        != DATABASE_INVENTORY_SCHEMA
    ):
        raise AF3PreflightError("attestation_runtime_identity_invalid")

    container_path = Path(container)
    model_path = Path(model_dir)
    database_path = Path(database_dir)
    database_manifest_path = Path(database_manifest)
    model_files, multiple_models = _select_model_files(model_path)
    actual_model_quick = _quick_file_set_records(model_files)
    components = {
        "container_present": (
            container_path.is_file() and container_path.stat().st_size > 0
        ),
        "container_size_matches": (
            container_path.is_file()
            and container_path.stat().st_size == expected_container.get("bytes")
        ),
        "container_mtime_matches": (
            container_path.is_file()
            and container_path.stat().st_mtime_ns
            == expected_container.get("mtime_ns")
        ),
        "single_model_parameter_set": (
            bool(model_files) and not multiple_models
        ),
        "model_file_identity_matches": (
            canonical_sha256(actual_model_quick)
            == expected_model_quick_sha256
        ),
        "database_manifest_checksum_matches": (
            database_manifest_path.is_file()
            and sha256_file(database_manifest_path)
            == value.get("checksums", {}).get("database_manifest_sha256")
        ),
        "database_quick_identity_matches": (
            _quick_database_inventory_matches(
                database_path,
                database_manifest_path,
            )
        ),
    }
    if mode == "full":
        components.update(
            {
                "container_content_matches": (
                    components["container_present"]
                    and sha256_file(container_path)
                    == expected_container.get("sha256")
                ),
                "model_content_matches": (
                    components["single_model_parameter_set"]
                    and _aggregate_file_set(model_files)[0]
                    == value.get("checksums", {}).get(
                        "model_parameter_set_sha256"
                    )
                ),
                "database_full_inventory_matches": (
                    _database_inventory_matches(
                        database_path,
                        database_manifest_path,
                    )
                ),
            }
        )
    violations = [
        name for name, passed in components.items() if passed is not True
    ]
    if violations:
        raise AF3PreflightError(
            "runtime_dependency_mismatch:" + ",".join(sorted(violations))
        )
    return {
        **base,
        "runtime_dependencies_verified": True,
        "verification_mode": mode,
        "components": components,
        "counts": {
            "model_parameter_files": len(model_files),
            "database_required_entries": len(REQUIRED_DATABASE_ENTRIES),
        },
        "release_boundary": {
            "local_paths_emitted": False,
            "dependency_filenames_emitted": False,
            "content_bytes_emitted": False,
        },
    }


def _load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise AF3PreflightError("expected_json_object")
    return value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run")
    run.add_argument("--preregistration", type=Path, required=True)
    run.add_argument("--input-freeze", type=Path, required=True)
    run.add_argument("--retained-manifest", type=Path, required=True)
    run.add_argument("--input-dir", type=Path, required=True)
    run.add_argument("--source-dir", type=Path, required=True)
    run.add_argument("--container", type=Path, required=True)
    run.add_argument("--expected-container-sha256", required=True)
    run.add_argument("--model-dir", type=Path, required=True)
    run.add_argument("--expected-model-sha256", required=True)
    run.add_argument("--database-dir", type=Path, required=True)
    run.add_argument("--database-manifest", type=Path, required=True)
    run.add_argument("--expected-database-manifest-sha256", required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--runtime-command", default="singularity")
    run.add_argument("--resume", action="store_true")
    run.add_argument("--attestation-out", type=Path, required=True)
    run.add_argument("--public-summary-out", type=Path)

    verify = subparsers.add_parser("verify")
    verify.add_argument("--attestation", type=Path, required=True)
    verify.add_argument("--expected-attestation-sha256", required=True)
    verify.add_argument("--preregistration", type=Path, required=True)
    verify.add_argument("--input-freeze", type=Path, required=True)
    verify.add_argument("--retained-manifest", type=Path, required=True)
    verify.add_argument("--input-dir", type=Path, required=True)

    verify_runtime = subparsers.add_parser("verify-runtime")
    verify_runtime.add_argument("--attestation", type=Path, required=True)
    verify_runtime.add_argument(
        "--expected-attestation-sha256",
        required=True,
    )
    verify_runtime.add_argument(
        "--preregistration",
        type=Path,
        required=True,
    )
    verify_runtime.add_argument("--input-freeze", type=Path, required=True)
    verify_runtime.add_argument(
        "--retained-manifest",
        type=Path,
        required=True,
    )
    verify_runtime.add_argument("--input-dir", type=Path, required=True)
    verify_runtime.add_argument("--container", type=Path, required=True)
    verify_runtime.add_argument("--model-dir", type=Path, required=True)
    verify_runtime.add_argument("--database-dir", type=Path, required=True)
    verify_runtime.add_argument(
        "--database-manifest",
        type=Path,
        required=True,
    )
    verify_runtime.add_argument(
        "--mode",
        choices=("quick", "full"),
        default="quick",
    )

    inventory = subparsers.add_parser("inventory")
    inventory.add_argument("--database-dir", type=Path, required=True)
    inventory.add_argument("--out", type=Path, required=True)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    if args.command == "inventory":
        inventory = build_database_inventory(args.database_dir)
        write_json(args.out, inventory)
        print(
            json.dumps(
                {
                    "database_manifest_sha256": sha256_file(args.out),
                    "entries": inventory["summary"]["entries"],
                    "files": inventory["summary"]["files"],
                },
                sort_keys=True,
            )
        )
        return 0

    preregistration = _load_json(args.preregistration)
    input_freeze = _load_json(args.input_freeze)
    if args.command == "verify":
        result = verify_attestation(
            attestation_path=args.attestation,
            expected_attestation_sha256=args.expected_attestation_sha256,
            preregistration=preregistration,
            input_freeze=input_freeze,
            retained_manifest=args.retained_manifest,
            input_dir=args.input_dir,
        )
        print(json.dumps(result, sort_keys=True))
        return 0
    if args.command == "verify-runtime":
        result = verify_runtime_dependencies(
            attestation_path=args.attestation,
            expected_attestation_sha256=args.expected_attestation_sha256,
            preregistration=preregistration,
            input_freeze=input_freeze,
            retained_manifest=args.retained_manifest,
            input_dir=args.input_dir,
            container=args.container,
            model_dir=args.model_dir,
            database_dir=args.database_dir,
            database_manifest=args.database_manifest,
            mode=args.mode,
        )
        print(json.dumps(result, sort_keys=True))
        return 0

    attestation = run_preflight(
        preregistration=preregistration,
        input_freeze=input_freeze,
        retained_manifest=args.retained_manifest,
        input_dir=args.input_dir,
        source_dir=args.source_dir,
        container=args.container,
        expected_container_sha256=args.expected_container_sha256,
        model_dir=args.model_dir,
        expected_model_sha256=args.expected_model_sha256,
        database_dir=args.database_dir,
        database_manifest=args.database_manifest,
        expected_database_manifest_sha256=(
            args.expected_database_manifest_sha256
        ),
        output_dir=args.output_dir,
        runtime_command=args.runtime_command,
        resume=args.resume,
    )
    write_json(args.attestation_out, attestation)
    if args.public_summary_out:
        write_json(
            args.public_summary_out,
            public_readiness_summary(attestation),
        )
    print(
        json.dumps(
            {
                "ready_for_af3_prediction": attestation[
                    "ready_for_af3_prediction"
                ],
                "violations": attestation["violations"],
            },
            sort_keys=True,
        )
    )
    return int(not attestation["ready_for_af3_prediction"])


if __name__ == "__main__":
    raise SystemExit(main())

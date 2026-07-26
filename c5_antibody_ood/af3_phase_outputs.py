"""Validate per-target outputs for split AlphaFold 3 execution phases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .manifest import load_c5_manifest
from .prospective_inputs import _safe_job_name
from .prospective_panel import validate_preregistration
from .prospective_predictions import (
    FORBIDDEN_LABEL_KEY_FRAGMENTS,
    ProspectivePredictionError,
    collect_af3_target_output,
)
from .source_pilot import sha256_file


class AF3PhaseOutputError(ValueError):
    """Raised when a split-phase target output fails closed."""


def _row_for_job(
    retained_rows: Sequence[Mapping[str, Any]],
    job_name: str,
) -> Mapping[str, Any]:
    matches = [
        row
        for row in retained_rows
        if _safe_job_name(str(row["instance_id"])) == job_name
    ]
    if len(matches) != 1:
        raise AF3PhaseOutputError("retained_job_match_invalid")
    return matches[0]


def _contains_forbidden_label_key(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized = "".join(
                char if char.isalnum() else "_" for char in str(key).lower()
            )
            collapsed = "".join(char for char in normalized if char.isalnum())
            if any(
                fragment in normalized
                or fragment.replace("_", "") in collapsed
                for fragment in FORBIDDEN_LABEL_KEY_FRAGMENTS
            ):
                return True
            if _contains_forbidden_label_key(nested):
                return True
    elif isinstance(value, list):
        return any(_contains_forbidden_label_key(item) for item in value)
    return False


def validate_data_pipeline_output(
    *,
    retained_rows: Sequence[Mapping[str, Any]],
    preregistration: Mapping[str, Any],
    job_name: str,
    job_dir: str | Path,
) -> dict[str, Any]:
    """Require the exact AF3 data-pipeline-only target contract."""

    issues = validate_preregistration(preregistration)
    if issues:
        raise AF3PhaseOutputError(
            "preregistration_invalid:" + ",".join(issues)
        )
    if _safe_job_name(job_name) != job_name:
        raise AF3PhaseOutputError("job_name_invalid")
    row = _row_for_job(retained_rows, job_name)
    path = Path(job_dir)
    if path.name != job_name or not path.is_dir() or path.is_symlink():
        raise AF3PhaseOutputError("pipeline_job_directory_invalid")
    entries = list(path.iterdir())
    expected = path / f"{job_name}_data.json"
    if (
        len(entries) != 1
        or entries[0] != expected
        or expected.is_symlink()
        or not expected.is_file()
        or expected.stat().st_size <= 0
    ):
        raise AF3PhaseOutputError("pipeline_job_entry_set_mismatch")
    try:
        value = json.loads(expected.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise AF3PhaseOutputError("pipeline_data_json_invalid") from exc
    if not isinstance(value, dict):
        raise AF3PhaseOutputError("pipeline_data_json_invalid")
    prediction = preregistration["protocol"]["prediction"]
    if (
        value.get("name") != job_name
        or value.get("dialect") != prediction["input_dialect"]
        or value.get("version") != prediction["input_version"]
        or value.get("modelSeeds") != prediction["model_seeds"]
    ):
        raise AF3PhaseOutputError("pipeline_data_identity_mismatch")
    sequences = value.get("sequences")
    if (
        not isinstance(sequences, list)
        or len(sequences) != len(row["chain_role_mapping"])
    ):
        raise AF3PhaseOutputError("pipeline_chain_count_mismatch")
    if _contains_forbidden_label_key(value):
        raise AF3PhaseOutputError("pipeline_hidden_label_key_detected")
    return {
        "schema_version": "c5_af3_phase_output_v1",
        "phase": "data_pipeline",
        "job_name": job_name,
        "verified": True,
        "files": 1,
        "data_json_sha256": sha256_file(expected),
        "release_boundary": {
            "local_paths_emitted": False,
            "sequence_content_emitted": False,
            "prediction_scores_emitted": False,
            "labels_emitted": False,
        },
    }


def validate_inference_output(
    *,
    retained_rows: Sequence[Mapping[str, Any]],
    preregistration: Mapping[str, Any],
    job_name: str,
    job_dir: str | Path,
) -> dict[str, Any]:
    """Require the canonical complete AF3 v3.0.3 target output contract."""

    issues = validate_preregistration(preregistration)
    if issues:
        raise AF3PhaseOutputError(
            "preregistration_invalid:" + ",".join(issues)
        )
    row = _row_for_job(retained_rows, job_name)
    try:
        target = collect_af3_target_output(
            row=row,
            preregistration=preregistration,
            job_dir=job_dir,
        )
    except ProspectivePredictionError as exc:
        raise AF3PhaseOutputError(f"inference_output_invalid:{exc}") from exc
    return {
        "schema_version": "c5_af3_phase_output_v1",
        "phase": "inference",
        "job_name": job_name,
        "verified": True,
        "samples": len(target["samples"]),
        "job_artifact_set_sha256": target["job_artifact_set_sha256"],
        "release_boundary": {
            "local_paths_emitted": False,
            "sequence_content_emitted": False,
            "prediction_scores_emitted": False,
            "labels_emitted": False,
        },
    }


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise AF3PhaseOutputError("expected_json_object")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase",
        choices=("data-pipeline", "inference"),
    )
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--retained-manifest", type=Path, required=True)
    parser.add_argument("--job-name", required=True)
    parser.add_argument("--job-dir", type=Path, required=True)
    args = parser.parse_args()

    preregistration = _load_json(args.preregistration)
    retained_rows = load_c5_manifest(args.retained_manifest)
    common = {
        "retained_rows": retained_rows,
        "preregistration": preregistration,
        "job_name": args.job_name,
        "job_dir": args.job_dir,
    }
    if args.phase == "data-pipeline":
        result = validate_data_pipeline_output(**common)
    else:
        result = validate_inference_output(**common)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

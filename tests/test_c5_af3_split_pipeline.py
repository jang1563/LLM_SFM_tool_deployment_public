import json
from pathlib import Path

import pytest

from c5_antibody_ood.af3_phase_outputs import (
    AF3PhaseOutputError,
    validate_data_pipeline_output,
)
from c5_antibody_ood.manifest import load_c5_manifest
from c5_antibody_ood.prospective_inputs import _safe_job_name


ROOT = Path(__file__).resolve().parents[1]
PREREGISTRATION = (
    ROOT
    / "c5_antibody_ood/c5_prospective_panel_preregistration_v1.json"
)
RETAINED_MANIFEST = (
    ROOT
    / "c5_antibody_ood/c5_sabdab2_prospective_retained_manifest_v1.jsonl"
)
DATA_PIPELINE_JOB = (
    ROOT / "c5_antibody_ood/run_c5_af3_data_pipeline_cayuga.sbatch"
)
INFERENCE_JOB = (
    ROOT / "c5_antibody_ood/run_c5_af3_inference_cayuga.sbatch"
)


def _pipeline_fixture(tmp_path: Path) -> tuple[dict, list[dict], str, Path]:
    preregistration = json.loads(PREREGISTRATION.read_text())
    rows = load_c5_manifest(RETAINED_MANIFEST)[:1]
    row = rows[0]
    job_name = _safe_job_name(row["instance_id"])
    job_dir = tmp_path / job_name
    job_dir.mkdir()
    prediction = preregistration["protocol"]["prediction"]
    data = {
        "name": job_name,
        "modelSeeds": prediction["model_seeds"],
        "sequences": [
            {"protein": {"id": mapping["chain_id"], "sequence": "ACDE"}}
            for mapping in row["chain_role_mapping"]
        ],
        "dialect": prediction["input_dialect"],
        "version": prediction["input_version"],
    }
    (job_dir / f"{job_name}_data.json").write_text(
        json.dumps(data, sort_keys=True) + "\n"
    )
    return preregistration, rows, job_name, job_dir


def test_data_pipeline_output_requires_exact_label_free_processed_json(
    tmp_path,
):
    preregistration, rows, job_name, job_dir = _pipeline_fixture(tmp_path)

    result = validate_data_pipeline_output(
        retained_rows=rows,
        preregistration=preregistration,
        job_name=job_name,
        job_dir=job_dir,
    )

    assert result["phase"] == "data_pipeline"
    assert result["verified"] is True
    assert result["files"] == 1
    assert len(result["data_json_sha256"]) == 64
    assert all(result["release_boundary"].values()) is False
    assert str(tmp_path) not in json.dumps(result, sort_keys=True)


def test_data_pipeline_output_rejects_extra_files_and_hidden_labels(tmp_path):
    preregistration, rows, job_name, job_dir = _pipeline_fixture(tmp_path)
    extra = job_dir / "unexpected.txt"
    extra.write_text("unexpected\n")

    with pytest.raises(
        AF3PhaseOutputError,
        match="pipeline_job_entry_set_mismatch",
    ):
        validate_data_pipeline_output(
            retained_rows=rows,
            preregistration=preregistration,
            job_name=job_name,
            job_dir=job_dir,
        )

    extra.unlink()
    data_path = job_dir / f"{job_name}_data.json"
    data = json.loads(data_path.read_text())
    data["nativeInterfaceLabel"] = "hidden"
    data_path.write_text(json.dumps(data) + "\n")
    with pytest.raises(
        AF3PhaseOutputError,
        match="pipeline_hidden_label_key_detected",
    ):
        validate_data_pipeline_output(
            retained_rows=rows,
            preregistration=preregistration,
            job_name=job_name,
            job_dir=job_dir,
        )


def test_data_pipeline_output_rejects_identity_drift(tmp_path):
    preregistration, rows, job_name, job_dir = _pipeline_fixture(tmp_path)
    data_path = job_dir / f"{job_name}_data.json"
    data = json.loads(data_path.read_text())
    data["modelSeeds"] = [1]
    data_path.write_text(json.dumps(data) + "\n")

    with pytest.raises(
        AF3PhaseOutputError,
        match="pipeline_data_identity_mismatch",
    ):
        validate_data_pipeline_output(
            retained_rows=rows,
            preregistration=preregistration,
            job_name=job_name,
            job_dir=job_dir,
        )


def test_split_cayuga_jobs_preserve_phase_and_runtime_contracts():
    data_script = DATA_PIPELINE_JOB.read_text()
    inference_script = INFERENCE_JOB.read_text()

    assert "#SBATCH --partition" not in data_script
    assert "#SBATCH --cpus-per-task=16" in data_script
    assert "#SBATCH --mem=128G" in data_script
    assert "#SBATCH --array=0-119%8" in data_script
    assert "--run_data_pipeline=true" in data_script
    assert "--run_inference=false" in data_script
    assert "--nv" not in data_script
    assert "c5_antibody_ood.af3_phase_outputs" in data_script
    assert 'mv "${STAGE_TARGET}" "${TARGET_OUTPUT}"' in data_script

    assert "#SBATCH --cpus-per-task=12" in inference_script
    assert "#SBATCH --array=0-119%8" in inference_script
    assert "--nv" in inference_script
    assert "--run_data_pipeline=false" in inference_script
    assert "--run_inference=true" in inference_script
    assert "--force_output_dir=true" in inference_script
    assert "c5_antibody_ood.af3_phase_outputs" in inference_script
    assert 'mv "${TARGET_OUTPUT}" "${BACKUP_OUTPUT}"' in inference_script
    assert 'mv "${STAGE_TARGET}" "${TARGET_OUTPUT}"' in inference_script

    for script in (data_script, inference_script):
        assert "c5_antibody_ood.af3_preflight verify-runtime" in script
        assert "--expected-attestation-sha256" in script
        assert "AF3_DB_MANIFEST" in script
        assert "uv run --no-sync python3" in script
        assert "--pwd /app/alphafold" in script
        assert "/Users/" not in script
        assert "/home/" not in script
        assert "/" + "scratch/" not in script

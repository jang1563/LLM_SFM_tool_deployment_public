import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFINITION = ROOT / "c5_antibody_ood/alphafold3_v3_0_3.def"
BUILD_JOB = (
    ROOT / "c5_antibody_ood/build_c5_af3_container_cayuga.sbatch"
)
DATABASE_JOB = (
    ROOT / "c5_antibody_ood/fetch_c5_af3_databases_cayuga.sbatch"
)
PARAMETER_JOB = (
    ROOT / "c5_antibody_ood/provision_c5_af3_parameters_cayuga.sbatch"
)
OFFICIAL_PARAMETER_JOB = (
    ROOT / "c5_antibody_ood/fetch_c5_af3_parameters_cayuga.sbatch"
)
ATTESTATION_JOB = (
    ROOT / "c5_antibody_ood/attest_c5_af3_runtime_cayuga.sbatch"
)
CONTAINER_READINESS = (
    ROOT
    / "c5_antibody_ood/c5_af3_container_readiness_2026-07-25.json"
)
DATABASE_READINESS = (
    ROOT
    / "c5_antibody_ood/c5_af3_database_readiness_2026-07-26.json"
)
AF3_COMMIT = "7b197fe859790fc3e04d03ea70dd0b9ba48881c9"
CONTAINER_SHA256 = (
    "128a62b4849f3606a61a12fbe754e3f928bdbe43fe1c0894a231380f419fe7b2"
)
DATABASE_MANIFEST_SHA256 = (
    "62a6a5a42e924c1b22e8e387936ad2c367b57a2d4b9b57d0e88bf38fd054527c"
)


def test_apptainer_definition_pins_official_af3_build_inputs():
    definition = DEFINITION.read_text()

    assert "From: ghcr.io/astral-sh/uv:0.9.24" in definition
    assert "From: nvidia/cuda:12.6.3-base-ubuntu24.04" in definition
    assert AF3_COMMIT in definition
    assert "v3.0.3" in definition
    assert "org.llm_sfm_tool_deployment.af3.version v3.0.3" in definition
    assert "org.opencontainers.image.version v3.0.3" not in definition
    assert "uv sync --frozen --all-groups --no-editable" in definition
    assert "uv run build_data" in definition
    assert "UV_CACHE_DIR=/tmp/uv-cache" in definition
    assert "uv run --no-sync python3" in definition
    assert "ca70d94fd0cf271bd7063423aabb116d42de533117343a9b27a65c17ff06fbf3" in definition
    assert "cd /\n    rm -rf /hmmer_build" in definition
    assert "XLA_CLIENT_MEM_FRACTION=0.95" in definition
    assert "Model parameters and" in definition
    assert "genetic databases are not included." in definition


def test_cayuga_build_job_is_clean_source_and_atomic_output_guarded():
    script = BUILD_JOB.read_text()

    assert "#SBATCH --cpus-per-task=12" in script
    assert "#SBATCH --mem=64G" in script
    assert AF3_COMMIT in script
    assert 'readonly AF3_TAG="v3.0.3"' in script
    assert "git -C" in script
    assert "status --porcelain" in script
    assert 'readonly BENCHMARK_COMMIT="$(git -C "${WORK}" rev-parse HEAD)"' in script
    assert '[[ "${OUTPUT_PATH}" = "${SOURCE_ROOT}/"* ]]' in script
    assert 'prior_partials=("${AF3_SIF_OUT}".partial.*)' in script
    assert 'PARTIAL_SIF="${AF3_SIF_OUT}.partial.${SLURM_JOB_ID}"' in script
    assert 'mv "${PARTIAL_SIF}" "${AF3_SIF_OUT}"' in script
    assert "apptainer build --fakeroot" in script
    assert 'apptainer test "${PARTIAL_SIF}"' in script
    assert "org.llm_sfm_tool_deployment.af3.version" in script
    assert 'uv run --no-sync python3 -c "import run_alphafold"' in script
    assert "run_alphafold.py --help" not in script
    assert '"local_paths_emitted": false' in script
    assert "/Users/" not in script
    assert "/home/" not in script
    assert "/" + "scratch/" not in script


def test_cayuga_database_job_is_content_hashed_and_atomic():
    script = DATABASE_JOB.read_text()

    assert "#SBATCH --cpus-per-task=12" in script
    assert "#SBATCH --time=08:00:00" in script
    assert "MINIMUM_FREE_BYTES=750000000000" in script
    assert "AF3_SIF_SHA256" in script
    assert "fetch_databases.sh /root/public_databases" in script
    assert "c5_antibody_ood.af3_preflight inventory" in script
    assert "uv run --no-sync python3" in script
    assert 'STAGE="${AF3_DB_OUT}.partial.${SLURM_JOB_ID}"' in script
    assert 'mv "${STAGE}" "${AF3_DB_OUT}"' in script
    assert 'mv "${MANIFEST_PARTIAL}" "${AF3_DB_MANIFEST_OUT}"' in script
    assert "private database outputs must be outside" in script
    assert "/Users/" not in script
    assert "/home/" not in script
    assert "/" + "scratch/" not in script


def test_cayuga_parameter_job_requires_authorization_and_atomic_inventory():
    script = PARAMETER_JOB.read_text()

    assert "#SBATCH --cpus-per-task=4" in script
    assert "AF3_AUTHORIZED_SOURCE_CONFIRMED" in script
    assert '!= "YES"' in script
    assert "AF3_SIF_SHA256" in script
    assert "c5_antibody_ood.af3_preflight" in script
    assert "provision-model" in script
    assert "--authorized-source-confirmed" in script
    assert 'STAGE="${AF3_MODEL_OUT}.partial.${SLURM_JOB_ID}"' in script
    assert 'mv "${STAGE}" "${AF3_MODEL_OUT}"' in script
    assert 'mv "${MANIFEST_PARTIAL}" "${AF3_MODEL_MANIFEST_OUT}"' in script
    assert "model source and outputs must be outside" in script
    assert "/Users/" not in script
    assert "/home/" not in script
    assert "/" + "scratch/" not in script


def test_cayuga_official_parameter_job_is_generation_and_terms_bound():
    script = OFFICIAL_PARAMETER_JOB.read_text()

    assert "#SBATCH --cpus-per-task=4" in script
    assert "#SBATCH --mem=16G" in script
    assert "AF3_MODEL_TERMS_ACCEPTED" in script
    assert '!= "YES"' in script
    assert (
        'OFFICIAL_MODEL_SOURCE_URL="https://storage.googleapis.com/'
        'alphafold3/af3.bin.zst"' in script
    )
    assert (
        'OFFICIAL_MODEL_OBJECT_GENERATION="1780568696389861"' in script
    )
    assert "OFFICIAL_MODEL_OBJECT_BYTES=1020545840" in script
    assert "dd1a7badb62cbb0d4571666002159842c8c578c5" in script
    assert "umask 077" in script
    assert "--proto '=https'" in script
    assert "--proto-redir '=https'" in script
    assert "--tlsv1.2" in script
    assert "--retry-all-errors" in script
    assert 'x-goog-generation' in script
    assert "--official-google-storage-download-confirmed" in script
    assert "--model-terms-accepted" in script
    assert 'DOWNLOAD_STAGE="${AF3_MODEL_OUT}.download.partial.' in script
    assert 'MODEL_STAGE="${AF3_MODEL_OUT}.partial.' in script
    assert 'mv "${SIDECAR_PARTIAL}" "${AF3_MODEL_MANIFEST_OUT}.sha256"' in script
    assert 'mv "${MANIFEST_PARTIAL}" "${AF3_MODEL_MANIFEST_OUT}"' in script
    assert "model outputs must be outside" in script
    assert "model manifest must be outside the model directory" in script
    assert "/Users/" not in script
    assert "/home/" not in script
    assert "/" + "scratch/" not in script


def test_cayuga_attestation_job_binds_runtime_and_promotes_atomically():
    script = ATTESTATION_JOB.read_text()

    assert "#SBATCH --cpus-per-task=12" in script
    assert "#SBATCH --mem=64G" in script
    assert "status --porcelain" in script
    assert "AF3_SIF_SHA256" in script
    assert "AF3_MODEL_SHA256" in script
    assert "AF3_MODEL_MANIFEST_SHA256" in script
    assert "AF3_DB_MANIFEST_SHA256" in script
    assert "--benchmark-dir /root/benchmark" in script
    assert "--model-manifest /root/c5_model_manifest.json" in script
    assert "--expected-model-manifest-sha256" in script
    assert "--runtime-command uv" in script
    assert "af3_preflight verify-runtime" in script
    assert "--mode quick" in script
    assert 'ATTESTATION_PARTIAL="${ATTESTATION_PARENT}/' in script
    assert 'SIDECAR_PARTIAL="${AF3_ATTESTATION_OUT}.sha256.partial.' in script
    assert 'mv "${SIDECAR_PARTIAL}" "${AF3_ATTESTATION_OUT}.sha256"' in script
    assert 'mv "${ATTESTATION_PARTIAL}" "${AF3_ATTESTATION_OUT}"' in script
    assert "private runtime dependencies and outputs must be outside" in script
    assert "/Users/" not in script
    assert "/home/" not in script
    assert "/" + "scratch/" not in script


def test_container_readiness_is_definition_bound_and_public_safe():
    readiness = json.loads(CONTAINER_READINESS.read_text())
    definition_sha256 = hashlib.sha256(DEFINITION.read_bytes()).hexdigest()

    assert readiness["af3_source"]["commit"] == AF3_COMMIT
    assert readiness["af3_source"]["tag"] == "v3.0.3"
    assert (
        readiness["build_identity"]["definition_sha256"]
        == definition_sha256
    )
    assert readiness["decision"]["container_ready"] is True
    assert readiness["decision"]["ready_for_af3_prediction"] is False
    assert readiness["verification"]["jax_gpu_device_smoke"] == {
        "status": "pass",
        "devices": 1,
    }
    assert all(readiness["release_boundary"].values()) is False
    rendered = json.dumps(readiness, sort_keys=True)
    assert "/Users/" not in rendered
    assert "/home/" not in rendered
    assert "/" + "scratch/" not in rendered
    assert "SLURM" not in rendered


def test_database_readiness_is_content_bound_and_public_safe():
    readiness = json.loads(DATABASE_READINESS.read_text())

    assert readiness["source"]["af3_commit"] == AF3_COMMIT
    assert readiness["source"]["af3_tag"] == "v3.0.3"
    assert readiness["binding"]["container_sha256"] == CONTAINER_SHA256
    assert (
        readiness["binding"]["database_manifest_sha256"]
        == DATABASE_MANIFEST_SHA256
    )
    assert (
        readiness["binding"]["inventory_schema"]
        == "c5_af3_database_inventory_v3"
    )
    assert readiness["summary"] == {
        "required_entries": 9,
        "files": 195867,
        "bytes": 672435030513,
    }
    assert all(readiness["verification"].values())
    assert readiness["decision"] == {
        "database_ready": True,
        "ready_for_af3_prediction": False,
        "remaining_blockers": ["authorized_model_parameters_missing"],
    }
    assert all(readiness["release_boundary"].values()) is False
    rendered = json.dumps(readiness, sort_keys=True)
    assert "/Users/" not in rendered
    assert "/home/" not in rendered
    assert "/" + "scratch/" not in rendered
    assert "SLURM" not in rendered

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFINITION = ROOT / "c5_antibody_ood/alphafold3_v3_0_3.def"
BUILD_JOB = (
    ROOT / "c5_antibody_ood/build_c5_af3_container_cayuga.sbatch"
)
AF3_COMMIT = "7b197fe859790fc3e04d03ea70dd0b9ba48881c9"


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

#!/bin/bash
#SBATCH --job-name=c5_af3_data
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --array=0-119%8
#SBATCH --output=c5_antibody_ood/runs/slurm/c5_af3_data_%A_%a.log
#SBATCH --error=c5_antibody_ood/runs/slurm/c5_af3_data_%A_%a.err
#
# AlphaFold 3 data-pipeline-only array for the prospective Stage B C5 panel.
# Run only after a zero-violation private runtime attestation is frozen.
#
#   mkdir -p c5_antibody_ood/runs/slurm
#   sbatch --account=<allocation> --partition=scu-cpu \
#     --export=ALL,WORK=$PWD,AF3_SIF=<image>,AF3_MODEL_DIR=<models>,AF3_DB_DIR=<databases>,AF3_DB_MANIFEST=<database-manifest>,AF3_INPUT_DIR=<inputs>,AF3_OUTPUT_DIR=<outputs>,AF3_ATTESTATION=<attestation>,AF3_ATTESTATION_SHA256=<sha256> \
#     c5_antibody_ood/run_c5_af3_data_pipeline_cayuga.sbatch
#
# Raw inputs, model parameters, databases, processed inputs, and scheduler logs
# are private run artifacts and must not be committed.

set -euo pipefail

WORK="${WORK:-${SLURM_SUBMIT_DIR:-$PWD}}"
PREREGISTRATION="${PREREGISTRATION:-c5_antibody_ood/c5_prospective_panel_preregistration_v1.json}"
INPUT_FREEZE="${INPUT_FREEZE:-c5_antibody_ood/c5_sabdab2_prospective_af3_input_freeze_2026-07-25.json}"
RETAINED_MANIFEST="${RETAINED_MANIFEST:-c5_antibody_ood/c5_sabdab2_prospective_retained_manifest_v1.jsonl}"
AF3_RUNTIME_VERIFY_MODE="${AF3_RUNTIME_VERIFY_MODE:-quick}"

for required in \
  AF3_SIF \
  AF3_MODEL_DIR \
  AF3_DB_DIR \
  AF3_DB_MANIFEST \
  AF3_INPUT_DIR \
  AF3_OUTPUT_DIR \
  AF3_ATTESTATION \
  AF3_ATTESTATION_SHA256
do
  if [ -z "${!required:-}" ]; then
    echo "missing required environment variable: ${required}" >&2
    exit 2
  fi
done

cd "${WORK}"
if [ "${AF3_RUNTIME_VERIFY_MODE}" != "quick" ] \
  && [ "${AF3_RUNTIME_VERIFY_MODE}" != "full" ]
then
  echo "AF3_RUNTIME_VERIFY_MODE must be quick or full" >&2
  exit 2
fi
for tracked_path in \
  "${PREREGISTRATION}" \
  "${INPUT_FREEZE}" \
  "${RETAINED_MANIFEST}"
do
  if [[ "${tracked_path}" = /* || "${tracked_path}" = *".."* ]]; then
    echo "tracked benchmark paths must be repo-relative" >&2
    exit 2
  fi
done

singularity exec \
  --pwd /app/alphafold \
  --bind "${WORK}:/root/benchmark:ro" \
  --bind "${AF3_INPUT_DIR}:/root/af_input:ro" \
  --bind "${AF3_MODEL_DIR}:/root/models:ro" \
  --bind "${AF3_DB_DIR}:/root/public_databases:ro" \
  --bind "${AF3_SIF}:/root/c5_af3.sif:ro" \
  --bind "${AF3_ATTESTATION}:/root/c5_attestation.json:ro" \
  --bind "${AF3_DB_MANIFEST}:/root/c5_database_manifest.json:ro" \
  "${AF3_SIF}" \
  env PYTHONPATH=/root/benchmark \
  uv run --no-sync python3 -m c5_antibody_ood.af3_preflight verify-runtime \
  --attestation /root/c5_attestation.json \
  --expected-attestation-sha256 "${AF3_ATTESTATION_SHA256}" \
  --preregistration "/root/benchmark/${PREREGISTRATION}" \
  --input-freeze "/root/benchmark/${INPUT_FREEZE}" \
  --retained-manifest "/root/benchmark/${RETAINED_MANIFEST}" \
  --input-dir /root/af_input \
  --container /root/c5_af3.sif \
  --model-dir /root/models \
  --database-dir /root/public_databases \
  --database-manifest /root/c5_database_manifest.json \
  --mode "${AF3_RUNTIME_VERIFY_MODE}"

mapfile -t AF3_INPUTS < <(
  find "${AF3_INPUT_DIR}" -maxdepth 1 -type f -name '*.json' -print \
    | LC_ALL=C sort
)
if [ "${#AF3_INPUTS[@]}" -ne 120 ]; then
  echo "expected 120 frozen AF3 inputs; found ${#AF3_INPUTS[@]}" >&2
  exit 3
fi

TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
ARRAY_ID="${SLURM_ARRAY_JOB_ID:?SLURM_ARRAY_JOB_ID is required}"
if [ "${TASK_ID}" -lt 0 ] || [ "${TASK_ID}" -ge "${#AF3_INPUTS[@]}" ]; then
  echo "array task index out of range: ${TASK_ID}" >&2
  exit 4
fi

INPUT_PATH="${AF3_INPUTS[${TASK_ID}]}"
INPUT_NAME="$(basename "${INPUT_PATH}")"
JOB_NAME="${INPUT_NAME%.json}"
TARGET_OUTPUT="${AF3_OUTPUT_DIR}/${JOB_NAME}"
STAGE_NAME=".${JOB_NAME}.data.partial.${ARRAY_ID}_${TASK_ID}"
TASK_STAGE="${AF3_OUTPUT_DIR}/${STAGE_NAME}"
STAGE_TARGET="${TASK_STAGE}/${JOB_NAME}"
mkdir -p "${AF3_OUTPUT_DIR}"
if [ -e "${TARGET_OUTPUT}" ] || [ -e "${TASK_STAGE}" ]; then
  echo "data-pipeline output boundary is not clean: ${JOB_NAME}" >&2
  exit 5
fi
mkdir "${TASK_STAGE}"

singularity exec \
  --pwd /app/alphafold \
  --bind "${AF3_INPUT_DIR}:/root/af_input:ro" \
  --bind "${TASK_STAGE}:/root/af_output" \
  --bind "${AF3_DB_DIR}:/root/public_databases:ro" \
  "${AF3_SIF}" \
  uv run --no-sync python3 run_alphafold.py \
  --json_path="/root/af_input/${INPUT_NAME}" \
  --db_dir=/root/public_databases \
  --run_data_pipeline=true \
  --run_inference=false \
  --num_diffusion_samples=5 \
  --output_dir=/root/af_output

singularity exec \
  --pwd /app/alphafold \
  --bind "${WORK}:/root/benchmark:ro" \
  --bind "${TASK_STAGE}:/root/af_output:ro" \
  "${AF3_SIF}" \
  env PYTHONPATH=/root/benchmark \
  uv run --no-sync python3 -m c5_antibody_ood.af3_phase_outputs \
  data-pipeline \
  --preregistration "/root/benchmark/${PREREGISTRATION}" \
  --retained-manifest "/root/benchmark/${RETAINED_MANIFEST}" \
  --job-name "${JOB_NAME}" \
  --job-dir "/root/af_output/${JOB_NAME}"

if [ "$(find "${TASK_STAGE}" -mindepth 1 -maxdepth 1 | wc -l)" -ne 1 ]; then
  echo "data-pipeline staging root contains unexpected entries" >&2
  exit 6
fi
mv "${STAGE_TARGET}" "${TARGET_OUTPUT}"
rmdir "${TASK_STAGE}"
echo "completed data-pipeline array task ${TASK_ID}"

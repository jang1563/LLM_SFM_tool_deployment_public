#!/bin/bash
#SBATCH --job-name=c5_af3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --array=0-119%8
#SBATCH --output=c5_antibody_ood/runs/slurm/c5_af3_%A_%a.log
#SBATCH --error=c5_antibody_ood/runs/slurm/c5_af3_%A_%a.err
#
# Prospective Stage B C5 AlphaFold 3 array for Cayuga.
#
# Run the environment preflight once and freeze its SHA-256 before submission.
# The array refuses to start from an absent, modified, or failing attestation.
# Example:
#
#   mkdir -p c5_antibody_ood/runs/slurm
#   sbatch --account=<allocation> --partition=scu-gpu --gres=gpu:a100:1 \
#     --export=ALL,WORK=$PWD,AF3_SIF=<image>,AF3_MODEL_DIR=<models>,AF3_DB_DIR=<databases>,AF3_INPUT_DIR=<inputs>,AF3_OUTPUT_DIR=<outputs>,AF3_ATTESTATION=<attestation>,AF3_ATTESTATION_SHA256=<sha256> \
#     c5_antibody_ood/run_c5_af3_cayuga.sbatch
#
# Raw inputs, model parameters, databases, predictions, and scheduler logs are
# private run artifacts and must not be committed.

set -euo pipefail

WORK="${WORK:-${SLURM_SUBMIT_DIR:-$PWD}}"
PREREGISTRATION="${PREREGISTRATION:-c5_antibody_ood/c5_prospective_panel_preregistration_v1.json}"
INPUT_FREEZE="${INPUT_FREEZE:-c5_antibody_ood/c5_sabdab2_prospective_af3_input_freeze_2026-07-25.json}"
RETAINED_MANIFEST="${RETAINED_MANIFEST:-c5_antibody_ood/c5_sabdab2_prospective_retained_manifest_v1.jsonl}"

for required in \
  AF3_SIF \
  AF3_MODEL_DIR \
  AF3_DB_DIR \
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
export PYTHONPATH="${WORK}:${PYTHONPATH:-}"

python -m c5_antibody_ood.af3_preflight verify \
  --attestation "${AF3_ATTESTATION}" \
  --expected-attestation-sha256 "${AF3_ATTESTATION_SHA256}" \
  --preregistration "${PREREGISTRATION}" \
  --input-freeze "${INPUT_FREEZE}" \
  --retained-manifest "${RETAINED_MANIFEST}" \
  --input-dir "${AF3_INPUT_DIR}"

mapfile -t AF3_INPUTS < <(
  find "${AF3_INPUT_DIR}" -maxdepth 1 -type f -name '*.json' -print \
    | LC_ALL=C sort
)
if [ "${#AF3_INPUTS[@]}" -ne 120 ]; then
  echo "expected 120 frozen AF3 inputs; found ${#AF3_INPUTS[@]}" >&2
  exit 3
fi

TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
if [ "${TASK_ID}" -lt 0 ] || [ "${TASK_ID}" -ge "${#AF3_INPUTS[@]}" ]; then
  echo "array task index out of range: ${TASK_ID}" >&2
  exit 4
fi

INPUT_PATH="${AF3_INPUTS[${TASK_ID}]}"
INPUT_NAME="$(basename "${INPUT_PATH}")"
JOB_NAME="${INPUT_NAME%.json}"
TARGET_OUTPUT="${AF3_OUTPUT_DIR}/${JOB_NAME}"
if [ -e "${TARGET_OUTPUT}" ]; then
  echo "target output already exists: ${JOB_NAME}" >&2
  exit 5
fi
mkdir -p "${AF3_OUTPUT_DIR}"

singularity exec \
  --nv \
  --bind "${AF3_INPUT_DIR}:/root/af_input:ro" \
  --bind "${AF3_OUTPUT_DIR}:/root/af_output" \
  --bind "${AF3_MODEL_DIR}:/root/models:ro" \
  --bind "${AF3_DB_DIR}:/root/public_databases:ro" \
  "${AF3_SIF}" \
  python run_alphafold.py \
  --json_path="/root/af_input/${INPUT_NAME}" \
  --model_dir=/root/models \
  --db_dir=/root/public_databases \
  --run_data_pipeline=true \
  --run_inference=true \
  --num_diffusion_samples=5 \
  --output_dir=/root/af_output

echo "completed array task ${TASK_ID}"

#!/bin/bash
#SBATCH --job-name=c5_af3_attest
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=c5_antibody_ood/runs/slurm/c5_af3_attest_%j.log
#SBATCH --error=c5_antibody_ood/runs/slurm/c5_af3_attest_%j.err
#
# Build an immutable private AF3 runtime attestation, then verify that the
# mounted paths still match it before atomic promotion.
#
#   mkdir -p c5_antibody_ood/runs/slurm
#   sbatch --account=<allocation> --partition=scu-cpu \
#     --export=ALL,WORK=$PWD,AF3_SOURCE_DIR=<clean-v3.0.3-source>,AF3_SIF=<image>,AF3_SIF_SHA256=<sha256>,AF3_MODEL_DIR=<models>,AF3_MODEL_SHA256=<model-set-sha256>,AF3_MODEL_MANIFEST=<model-manifest>,AF3_MODEL_MANIFEST_SHA256=<manifest-sha256>,AF3_DB_DIR=<databases>,AF3_DB_MANIFEST=<database-manifest>,AF3_DB_MANIFEST_SHA256=<manifest-sha256>,AF3_INPUT_DIR=<inputs>,AF3_OUTPUT_DIR=<new-output-dir>,AF3_ATTESTATION_OUT=<new-attestation-json> \
#     c5_antibody_ood/attest_c5_af3_runtime_cayuga.sbatch
#
# Parameters, databases, inputs, attestation, output, and scheduler logs are
# private run artifacts and must not be committed.

set -euo pipefail

readonly WORK="${WORK:-${SLURM_SUBMIT_DIR:-$PWD}}"
readonly PREREGISTRATION="${PREREGISTRATION:-c5_antibody_ood/c5_prospective_panel_preregistration_v1.json}"
readonly INPUT_FREEZE="${INPUT_FREEZE:-c5_antibody_ood/c5_sabdab2_prospective_af3_input_freeze_2026-07-25.json}"
readonly RETAINED_MANIFEST="${RETAINED_MANIFEST:-c5_antibody_ood/c5_sabdab2_prospective_retained_manifest_v1.jsonl}"

for required in \
  AF3_SOURCE_DIR \
  AF3_SIF \
  AF3_SIF_SHA256 \
  AF3_MODEL_DIR \
  AF3_MODEL_SHA256 \
  AF3_MODEL_MANIFEST \
  AF3_MODEL_MANIFEST_SHA256 \
  AF3_DB_DIR \
  AF3_DB_MANIFEST \
  AF3_DB_MANIFEST_SHA256 \
  AF3_INPUT_DIR \
  AF3_OUTPUT_DIR \
  AF3_ATTESTATION_OUT \
  SLURM_JOB_ID
do
  if [ -z "${!required:-}" ]; then
    echo "missing required environment variable: ${required}" >&2
    exit 2
  fi
done
for digest in \
  "${AF3_SIF_SHA256}" \
  "${AF3_MODEL_SHA256}" \
  "${AF3_MODEL_MANIFEST_SHA256}" \
  "${AF3_DB_MANIFEST_SHA256}"
do
  if [[ ! "${digest}" =~ ^[a-f0-9]{64}$ ]]; then
    echo "expected content identity must be a lowercase SHA-256" >&2
    exit 2
  fi
done
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
if [ -n "$(git -C "${WORK}" status --porcelain)" ]; then
  echo "benchmark source tree must be clean" >&2
  exit 3
fi
if [ -n "$(git -C "${AF3_SOURCE_DIR}" status --porcelain)" ]; then
  echo "AlphaFold 3 source tree must be clean" >&2
  exit 3
fi
if [ ! -f "${AF3_SIF}" ] \
  || [ "$(sha256sum "${AF3_SIF}" | cut -d ' ' -f 1)" != "${AF3_SIF_SHA256}" ]
then
  echo "AlphaFold 3 image identity mismatch" >&2
  exit 3
fi
if [ ! -f "${AF3_MODEL_MANIFEST}" ] \
  || [ "$(sha256sum "${AF3_MODEL_MANIFEST}" | cut -d ' ' -f 1)" != "${AF3_MODEL_MANIFEST_SHA256}" ]
then
  echo "model manifest identity mismatch" >&2
  exit 3
fi
if [ ! -f "${AF3_DB_MANIFEST}" ] \
  || [ "$(sha256sum "${AF3_DB_MANIFEST}" | cut -d ' ' -f 1)" != "${AF3_DB_MANIFEST_SHA256}" ]
then
  echo "database manifest identity mismatch" >&2
  exit 3
fi

readonly WORK_ROOT="$(realpath "${WORK}")"
for private_path in \
  "${AF3_SOURCE_DIR}" \
  "${AF3_SIF}" \
  "${AF3_MODEL_DIR}" \
  "${AF3_MODEL_MANIFEST}" \
  "${AF3_DB_DIR}" \
  "${AF3_DB_MANIFEST}" \
  "${AF3_INPUT_DIR}" \
  "${AF3_OUTPUT_DIR}" \
  "${AF3_ATTESTATION_OUT}"
do
  resolved="$(realpath -m "${private_path}")"
  if [[ "${resolved}" = "${WORK_ROOT}" || "${resolved}" = "${WORK_ROOT}/"* ]]; then
    echo "private runtime dependencies and outputs must be outside the repository" >&2
    exit 4
  fi
done

readonly OUTPUT_PARENT="$(dirname "${AF3_OUTPUT_DIR}")"
readonly OUTPUT_NAME="$(basename "${AF3_OUTPUT_DIR}")"
readonly ATTESTATION_PARENT="$(dirname "${AF3_ATTESTATION_OUT}")"
readonly ATTESTATION_NAME="$(basename "${AF3_ATTESTATION_OUT}")"
readonly ATTESTATION_PARTIAL="${ATTESTATION_PARENT}/${ATTESTATION_NAME}.partial.${SLURM_JOB_ID}"
readonly SIDECAR_PARTIAL="${AF3_ATTESTATION_OUT}.sha256.partial.${SLURM_JOB_ID}"
mkdir -p "${OUTPUT_PARENT}" "${ATTESTATION_PARENT}"
shopt -s nullglob
prior_partials=(
  "${AF3_ATTESTATION_OUT}".partial.*
  "${AF3_ATTESTATION_OUT}.sha256".partial.*
)
if [ -e "${AF3_OUTPUT_DIR}" ] \
  || [ -e "${AF3_ATTESTATION_OUT}" ] \
  || [ -e "${AF3_ATTESTATION_OUT}.sha256" ] \
  || [ "${#prior_partials[@]}" -ne 0 ]
then
  echo "attestation output boundary is not clean" >&2
  exit 4
fi

apptainer exec \
  --pwd /app/alphafold \
  --bind "${WORK}:/root/benchmark:ro" \
  --bind "${AF3_SOURCE_DIR}:/root/af3_source:ro" \
  --bind "${AF3_SIF}:/root/c5_af3.sif:ro" \
  --bind "${AF3_MODEL_DIR}:/root/models:ro" \
  --bind "${AF3_MODEL_MANIFEST}:/root/c5_model_manifest.json:ro" \
  --bind "${AF3_DB_DIR}:/root/public_databases:ro" \
  --bind "${AF3_DB_MANIFEST}:/root/c5_database_manifest.json:ro" \
  --bind "${AF3_INPUT_DIR}:/root/af_input:ro" \
  --bind "${OUTPUT_PARENT}:/root/output_parent" \
  --bind "${ATTESTATION_PARENT}:/root/c5_attestation" \
  "${AF3_SIF}" \
  env PYTHONPATH=/root/benchmark \
  uv run --no-sync python3 -m c5_antibody_ood.af3_preflight run \
  --preregistration "/root/benchmark/${PREREGISTRATION}" \
  --input-freeze "/root/benchmark/${INPUT_FREEZE}" \
  --retained-manifest "/root/benchmark/${RETAINED_MANIFEST}" \
  --input-dir /root/af_input \
  --benchmark-dir /root/benchmark \
  --source-dir /root/af3_source \
  --container /root/c5_af3.sif \
  --expected-container-sha256 "${AF3_SIF_SHA256}" \
  --model-dir /root/models \
  --expected-model-sha256 "${AF3_MODEL_SHA256}" \
  --model-manifest /root/c5_model_manifest.json \
  --expected-model-manifest-sha256 "${AF3_MODEL_MANIFEST_SHA256}" \
  --database-dir /root/public_databases \
  --database-manifest /root/c5_database_manifest.json \
  --expected-database-manifest-sha256 "${AF3_DB_MANIFEST_SHA256}" \
  --output-dir "/root/output_parent/${OUTPUT_NAME}" \
  --runtime-command uv \
  --attestation-out "/root/c5_attestation/$(basename "${ATTESTATION_PARTIAL}")"

test -s "${ATTESTATION_PARTIAL}"
readonly ATTESTATION_SHA256="$(
  sha256sum "${ATTESTATION_PARTIAL}" | cut -d ' ' -f 1
)"

apptainer exec \
  --pwd /app/alphafold \
  --bind "${WORK}:/root/benchmark:ro" \
  --bind "${AF3_SIF}:/root/c5_af3.sif:ro" \
  --bind "${AF3_MODEL_DIR}:/root/models:ro" \
  --bind "${AF3_MODEL_MANIFEST}:/root/c5_model_manifest.json:ro" \
  --bind "${AF3_DB_DIR}:/root/public_databases:ro" \
  --bind "${AF3_DB_MANIFEST}:/root/c5_database_manifest.json:ro" \
  --bind "${AF3_INPUT_DIR}:/root/af_input:ro" \
  --bind "${ATTESTATION_PARTIAL}:/root/c5_attestation.json:ro" \
  "${AF3_SIF}" \
  env PYTHONPATH=/root/benchmark \
  uv run --no-sync python3 -m c5_antibody_ood.af3_preflight verify-runtime \
  --attestation /root/c5_attestation.json \
  --expected-attestation-sha256 "${ATTESTATION_SHA256}" \
  --preregistration "/root/benchmark/${PREREGISTRATION}" \
  --input-freeze "/root/benchmark/${INPUT_FREEZE}" \
  --retained-manifest "/root/benchmark/${RETAINED_MANIFEST}" \
  --input-dir /root/af_input \
  --benchmark-dir /root/benchmark \
  --container /root/c5_af3.sif \
  --model-dir /root/models \
  --model-manifest /root/c5_model_manifest.json \
  --database-dir /root/public_databases \
  --database-manifest /root/c5_database_manifest.json \
  --mode quick

printf '%s  %s\n' \
  "${ATTESTATION_SHA256}" \
  "${ATTESTATION_NAME}" \
  > "${SIDECAR_PARTIAL}"
chmod 0444 "${ATTESTATION_PARTIAL}" "${SIDECAR_PARTIAL}"
mv "${SIDECAR_PARTIAL}" "${AF3_ATTESTATION_OUT}.sha256"
mv "${ATTESTATION_PARTIAL}" "${AF3_ATTESTATION_OUT}"

echo "AlphaFold 3 runtime attestation passed"
echo "attestation_sha256=${ATTESTATION_SHA256}"

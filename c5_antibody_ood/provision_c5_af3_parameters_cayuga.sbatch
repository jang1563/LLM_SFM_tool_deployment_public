#!/bin/bash
#SBATCH --job-name=c5_af3_params
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=c5_antibody_ood/runs/slurm/c5_af3_params_%j.log
#SBATCH --error=c5_antibody_ood/runs/slurm/c5_af3_params_%j.err
#
# Validate and atomically provision AF3 parameters received directly from
# Google after authorization. Parameter files, private inventories, and logs
# must remain outside the public repository.
#
#   mkdir -p c5_antibody_ood/runs/slurm
#   sbatch --account=<allocation> --partition=scu-cpu \
#     --export=ALL,WORK=$PWD,AF3_SIF=<image>,AF3_SIF_SHA256=<sha256>,AF3_MODEL_SOURCE=<authorized-download-dir>,AF3_MODEL_OUT=<new-model-dir>,AF3_MODEL_MANIFEST_OUT=<new-private-manifest>,AF3_AUTHORIZED_SOURCE_CONFIRMED=YES \
#     c5_antibody_ood/provision_c5_af3_parameters_cayuga.sbatch

set -euo pipefail

readonly WORK="${WORK:-${SLURM_SUBMIT_DIR:-$PWD}}"

for required in \
  AF3_SIF \
  AF3_SIF_SHA256 \
  AF3_MODEL_SOURCE \
  AF3_MODEL_OUT \
  AF3_MODEL_MANIFEST_OUT \
  SLURM_JOB_ID
do
  if [ -z "${!required:-}" ]; then
    echo "missing required environment variable: ${required}" >&2
    exit 2
  fi
done
if [ "${AF3_AUTHORIZED_SOURCE_CONFIRMED:-}" != "YES" ]; then
  echo "AF3_AUTHORIZED_SOURCE_CONFIRMED must be YES" >&2
  exit 2
fi
if [ -n "$(git -C "${WORK}" status --porcelain)" ]; then
  echo "benchmark source tree must be clean" >&2
  exit 3
fi
if [ ! -f "${AF3_SIF}" ]; then
  echo "AlphaFold 3 image is missing" >&2
  exit 3
fi
if [ "$(sha256sum "${AF3_SIF}" | awk '{print $1}')" \
  != "${AF3_SIF_SHA256}" ]
then
  echo "AlphaFold 3 image checksum mismatch" >&2
  exit 3
fi
if [ ! -d "${AF3_MODEL_SOURCE}" ]; then
  echo "authorized model source directory is missing" >&2
  exit 3
fi

readonly WORK_ROOT="$(realpath "${WORK}")"
readonly SOURCE_PATH="$(realpath "${AF3_MODEL_SOURCE}")"
readonly MODEL_PATH="$(realpath -m "${AF3_MODEL_OUT}")"
readonly MANIFEST_PATH="$(realpath -m "${AF3_MODEL_MANIFEST_OUT}")"
if [[ "${SOURCE_PATH}" = "${WORK_ROOT}/"* ]] \
  || [[ "${MODEL_PATH}" = "${WORK_ROOT}/"* ]] \
  || [[ "${MANIFEST_PATH}" = "${WORK_ROOT}/"* ]]
then
  echo "model source and outputs must be outside the benchmark repository" >&2
  exit 4
fi
if [ "${SOURCE_PATH}" = "${MODEL_PATH}" ]; then
  echo "model source and output directories must differ" >&2
  exit 4
fi

readonly MODEL_PARENT="$(dirname "${AF3_MODEL_OUT}")"
readonly MANIFEST_PARENT="$(dirname "${AF3_MODEL_MANIFEST_OUT}")"
readonly MANIFEST_NAME="$(basename "${AF3_MODEL_MANIFEST_OUT}")"
readonly STAGE="${AF3_MODEL_OUT}.partial.${SLURM_JOB_ID}"
readonly MANIFEST_PARTIAL="$(
  dirname "${AF3_MODEL_MANIFEST_OUT}"
)/${MANIFEST_NAME}.partial.${SLURM_JOB_ID}"
mkdir -p "${MODEL_PARENT}" "${MANIFEST_PARENT}"
shopt -s nullglob
prior_model_partials=("${AF3_MODEL_OUT}".partial.*)
prior_manifest_partials=("${AF3_MODEL_MANIFEST_OUT}".partial.*)
if [ -e "${AF3_MODEL_OUT}" ] \
  || [ -e "${AF3_MODEL_MANIFEST_OUT}" ] \
  || [ -e "${AF3_MODEL_MANIFEST_OUT}.sha256" ] \
  || [ "${#prior_model_partials[@]}" -ne 0 ] \
  || [ "${#prior_manifest_partials[@]}" -ne 0 ]
then
  echo "model parameter output boundary is not clean" >&2
  exit 4
fi
mkdir "${STAGE}"

apptainer exec \
  --pwd /app/alphafold \
  --bind "${WORK}:/root/benchmark:ro" \
  --bind "${AF3_MODEL_SOURCE}:/root/model_source:ro" \
  --bind "${STAGE}:/root/model_stage" \
  --bind "${MANIFEST_PARENT}:/root/c5_manifest" \
  "${AF3_SIF}" \
  env PYTHONPATH=/root/benchmark \
  uv run --no-sync python3 -m c5_antibody_ood.af3_preflight \
  provision-model \
  --source-dir /root/model_source \
  --stage-dir /root/model_stage \
  --authorized-source-confirmed \
  --out "/root/c5_manifest/$(basename "${MANIFEST_PARTIAL}")"

test -s "${MANIFEST_PARTIAL}"
readonly MANIFEST_SHA256="$(
  sha256sum "${MANIFEST_PARTIAL}" | awk '{print $1}'
)"
mv "${STAGE}" "${AF3_MODEL_OUT}"
mv "${MANIFEST_PARTIAL}" "${AF3_MODEL_MANIFEST_OUT}"
printf '%s  %s\n' \
  "${MANIFEST_SHA256}" \
  "${MANIFEST_NAME}" \
  > "${AF3_MODEL_MANIFEST_OUT}.sha256"
find "${AF3_MODEL_OUT}" -type f -exec chmod 0444 {} +
find "${AF3_MODEL_OUT}" -type d -exec chmod 0555 {} +
chmod 0444 \
  "${AF3_MODEL_MANIFEST_OUT}" \
  "${AF3_MODEL_MANIFEST_OUT}.sha256"

echo "AlphaFold 3 model parameter provisioning passed"
echo "model_manifest_sha256=${MANIFEST_SHA256}"

#!/bin/bash
#SBATCH --job-name=c5_af3_params_fetch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=c5_antibody_ood/runs/slurm/c5_af3_params_fetch_%j.log
#SBATCH --error=c5_antibody_ood/runs/slurm/c5_af3_params_fetch_%j.err
#
# Download the official AF3 parameter object from its generation-pinned Google
# Storage URL, content-hash it, and promote it atomically with private
# provenance. Downloading or using the parameters is subject to Google's
# current AlphaFold 3 Model Parameters Terms of Use.
#
#   mkdir -p c5_antibody_ood/runs/slurm
#   sbatch --account=<allocation> --partition=scu-cpu \
#     --export=ALL,WORK=$PWD,AF3_SIF=<image>,AF3_SIF_SHA256=<sha256>,AF3_MODEL_OUT=<new-model-dir>,AF3_MODEL_MANIFEST_OUT=<new-private-manifest>,AF3_MODEL_TERMS_ACCEPTED=YES \
#     c5_antibody_ood/fetch_c5_af3_parameters_cayuga.sbatch
#
# Parameters, private inventories, partial downloads, and scheduler logs must
# remain outside the public repository.

set -euo pipefail
umask 077

readonly WORK="${WORK:-${SLURM_SUBMIT_DIR:-$PWD}}"
readonly OFFICIAL_MODEL_SOURCE_URL="https://storage.googleapis.com/alphafold3/af3.bin.zst"
readonly OFFICIAL_MODEL_OBJECT_GENERATION="1780568696389861"
readonly OFFICIAL_MODEL_OBJECT_BYTES=1020545840
readonly OFFICIAL_MODEL_INSTRUCTION_COMMIT="dd1a7badb62cbb0d4571666002159842c8c578c5"
readonly OFFICIAL_MODEL_DOWNLOAD_URL="${OFFICIAL_MODEL_SOURCE_URL}?generation=${OFFICIAL_MODEL_OBJECT_GENERATION}"

for required in \
  AF3_SIF \
  AF3_SIF_SHA256 \
  AF3_MODEL_OUT \
  AF3_MODEL_MANIFEST_OUT \
  SLURM_JOB_ID
do
  if [ -z "${!required:-}" ]; then
    echo "missing required environment variable: ${required}" >&2
    exit 2
  fi
done
if [ "${AF3_MODEL_TERMS_ACCEPTED:-}" != "YES" ]; then
  echo "AF3_MODEL_TERMS_ACCEPTED must be YES" >&2
  exit 2
fi
if [[ ! "${AF3_SIF_SHA256}" =~ ^[a-f0-9]{64}$ ]]; then
  echo "AF3_SIF_SHA256 must be a lowercase SHA-256" >&2
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

readonly WORK_ROOT="$(realpath "${WORK}")"
readonly MODEL_PATH="$(realpath -m "${AF3_MODEL_OUT}")"
readonly MANIFEST_PATH="$(realpath -m "${AF3_MODEL_MANIFEST_OUT}")"
if [[ "${MODEL_PATH}" = "${WORK_ROOT}" || "${MODEL_PATH}" = "${WORK_ROOT}/"* ]] \
  || [[ "${MANIFEST_PATH}" = "${WORK_ROOT}" || "${MANIFEST_PATH}" = "${WORK_ROOT}/"* ]]
then
  echo "model outputs must be outside the benchmark repository" >&2
  exit 4
fi
if [[ "${MANIFEST_PATH}" = "${MODEL_PATH}"/* ]]; then
  echo "model manifest must be outside the model directory" >&2
  exit 4
fi

readonly MODEL_PARENT="$(dirname "${AF3_MODEL_OUT}")"
readonly MANIFEST_PARENT="$(dirname "${AF3_MODEL_MANIFEST_OUT}")"
readonly MANIFEST_NAME="$(basename "${AF3_MODEL_MANIFEST_OUT}")"
readonly DOWNLOAD_STAGE="${AF3_MODEL_OUT}.download.partial.${SLURM_JOB_ID}"
readonly MODEL_STAGE="${AF3_MODEL_OUT}.partial.${SLURM_JOB_ID}"
readonly DOWNLOAD_FILE="${DOWNLOAD_STAGE}/af3.bin.zst"
readonly HEADER_PARTIAL="${DOWNLOAD_STAGE}/response.headers"
readonly MANIFEST_PARTIAL="$(
  dirname "${AF3_MODEL_MANIFEST_OUT}"
)/${MANIFEST_NAME}.partial.${SLURM_JOB_ID}"
readonly SIDECAR_PARTIAL="${AF3_MODEL_MANIFEST_OUT}.sha256.partial.${SLURM_JOB_ID}"
mkdir -p "${MODEL_PARENT}" "${MANIFEST_PARENT}"
shopt -s nullglob
prior_model_partials=(
  "${AF3_MODEL_OUT}".partial.*
  "${AF3_MODEL_OUT}".download.partial.*
)
prior_manifest_partials=(
  "${AF3_MODEL_MANIFEST_OUT}".partial.*
  "${AF3_MODEL_MANIFEST_OUT}.sha256".partial.*
)
if [ -e "${AF3_MODEL_OUT}" ] \
  || [ -e "${AF3_MODEL_MANIFEST_OUT}" ] \
  || [ -e "${AF3_MODEL_MANIFEST_OUT}.sha256" ] \
  || [ "${#prior_model_partials[@]}" -ne 0 ] \
  || [ "${#prior_manifest_partials[@]}" -ne 0 ]
then
  echo "model parameter output boundary is not clean" >&2
  exit 4
fi
mkdir "${DOWNLOAD_STAGE}" "${MODEL_STAGE}"

curl \
  --fail \
  --location \
  --silent \
  --show-error \
  --retry 5 \
  --retry-delay 5 \
  --retry-all-errors \
  --proto '=https' \
  --proto-redir '=https' \
  --tlsv1.2 \
  --dump-header "${HEADER_PARTIAL}" \
  --output "${DOWNLOAD_FILE}" \
  "${OFFICIAL_MODEL_DOWNLOAD_URL}"

readonly ACTUAL_GENERATION="$(
  awk -F ': ' '
    tolower($1) == "x-goog-generation" {
      gsub("\r", "", $2)
      value = $2
    }
    END { print value }
  ' "${HEADER_PARTIAL}"
)"
if [ "${ACTUAL_GENERATION}" != "${OFFICIAL_MODEL_OBJECT_GENERATION}" ]; then
  echo "official model object generation mismatch" >&2
  exit 5
fi
if [ "$(stat --format='%s' "${DOWNLOAD_FILE}")" \
  -ne "${OFFICIAL_MODEL_OBJECT_BYTES}" ]
then
  echo "official model object byte count mismatch" >&2
  exit 5
fi

apptainer exec \
  --pwd /app/alphafold \
  --bind "${WORK}:/root/benchmark:ro" \
  --bind "${DOWNLOAD_STAGE}:/root/model_source:ro" \
  --bind "${MODEL_STAGE}:/root/model_stage" \
  --bind "${MANIFEST_PARENT}:/root/c5_manifest" \
  "${AF3_SIF}" \
  env PYTHONPATH=/root/benchmark \
  uv run --no-sync python3 -m c5_antibody_ood.af3_preflight \
  provision-model \
  --source-dir /root/model_source \
  --stage-dir /root/model_stage \
  --official-google-storage-download-confirmed \
  --model-terms-accepted \
  --out "/root/c5_manifest/$(basename "${MANIFEST_PARTIAL}")"

test -s "${MANIFEST_PARTIAL}"
readonly MANIFEST_SHA256="$(
  sha256sum "${MANIFEST_PARTIAL}" | awk '{print $1}'
)"
printf '%s  %s\n' \
  "${MANIFEST_SHA256}" \
  "${MANIFEST_NAME}" \
  > "${SIDECAR_PARTIAL}"

find "${MODEL_STAGE}" -type f -exec chmod 0444 {} +
find "${MODEL_STAGE}" -type d -exec chmod 0555 {} +
chmod 0444 "${MANIFEST_PARTIAL}" "${SIDECAR_PARTIAL}"
unlink "${HEADER_PARTIAL}"
unlink "${DOWNLOAD_FILE}"
rmdir "${DOWNLOAD_STAGE}"
mv "${MODEL_STAGE}" "${AF3_MODEL_OUT}"
mv "${SIDECAR_PARTIAL}" "${AF3_MODEL_MANIFEST_OUT}.sha256"
mv "${MANIFEST_PARTIAL}" "${AF3_MODEL_MANIFEST_OUT}"

echo "AlphaFold 3 official model parameter provisioning passed"
echo "source_instruction_commit=${OFFICIAL_MODEL_INSTRUCTION_COMMIT}"
echo "model_manifest_sha256=${MANIFEST_SHA256}"

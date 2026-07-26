#!/bin/bash
#SBATCH --job-name=c5_af3_db
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=c5_antibody_ood/runs/slurm/c5_af3_db_%j.log
#SBATCH --error=c5_antibody_ood/runs/slurm/c5_af3_db_%j.err
#
# Fetch the official AlphaFold 3 databases on a Cayuga CPU node, content-hash
# the completed files, and promote them atomically from private staging.
#
#   mkdir -p c5_antibody_ood/runs/slurm
#   sbatch --account=<allocation> --partition=scu-cpu \
#     --export=ALL,WORK=$PWD,AF3_SIF=<image>,AF3_SIF_SHA256=<sha256>,AF3_DB_OUT=<new-database-dir>,AF3_DB_MANIFEST_OUT=<new-private-manifest> \
#     c5_antibody_ood/fetch_c5_af3_databases_cayuga.sbatch
#
# Databases, inventories, partial downloads, and scheduler logs are private
# run artifacts and must not be committed.

set -euo pipefail

readonly WORK="${WORK:-${SLURM_SUBMIT_DIR:-$PWD}}"
readonly MINIMUM_FREE_BYTES=750000000000

for required in \
  AF3_SIF \
  AF3_SIF_SHA256 \
  AF3_DB_OUT \
  AF3_DB_MANIFEST_OUT \
  SLURM_JOB_ID
do
  if [ -z "${!required:-}" ]; then
    echo "missing required environment variable: ${required}" >&2
    exit 2
  fi
done

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
readonly DB_PATH="$(realpath -m "${AF3_DB_OUT}")"
readonly MANIFEST_PATH="$(realpath -m "${AF3_DB_MANIFEST_OUT}")"
if [[ "${DB_PATH}" = "${WORK_ROOT}/"* ]] \
  || [[ "${MANIFEST_PATH}" = "${WORK_ROOT}/"* ]]
then
  echo "private database outputs must be outside the benchmark repository" >&2
  exit 4
fi

readonly DB_PARENT="$(dirname "${AF3_DB_OUT}")"
readonly MANIFEST_PARENT="$(dirname "${AF3_DB_MANIFEST_OUT}")"
readonly MANIFEST_NAME="$(basename "${AF3_DB_MANIFEST_OUT}")"
readonly STAGE="${AF3_DB_OUT}.partial.${SLURM_JOB_ID}"
readonly MANIFEST_PARTIAL="$(
  dirname "${AF3_DB_MANIFEST_OUT}"
)/${MANIFEST_NAME}.partial.${SLURM_JOB_ID}"
mkdir -p "${DB_PARENT}" "${MANIFEST_PARENT}"
shopt -s nullglob
prior_db_partials=("${AF3_DB_OUT}".partial.*)
prior_manifest_partials=("${AF3_DB_MANIFEST_OUT}".partial.*)
if [ -e "${AF3_DB_OUT}" ] \
  || [ -e "${AF3_DB_MANIFEST_OUT}" ] \
  || [ -e "${AF3_DB_MANIFEST_OUT}.sha256" ] \
  || [ "${#prior_db_partials[@]}" -ne 0 ] \
  || [ "${#prior_manifest_partials[@]}" -ne 0 ]
then
  echo "database output boundary is not clean" >&2
  exit 4
fi

readonly AVAILABLE_BYTES="$(
  df --output=avail --block-size=1 "${DB_PARENT}" | tail -1
)"
if [ "${AVAILABLE_BYTES}" -lt "${MINIMUM_FREE_BYTES}" ]; then
  echo "database filesystem has less than 750 GB free" >&2
  exit 5
fi
mkdir "${STAGE}"

apptainer exec \
  --bind "${STAGE}:/root/public_databases" \
  "${AF3_SIF}" \
  bash /app/alphafold/fetch_databases.sh /root/public_databases

apptainer exec \
  --pwd /app/alphafold \
  --bind "${WORK}:/root/benchmark:ro" \
  --bind "${STAGE}:/root/public_databases:ro" \
  --bind "${MANIFEST_PARENT}:/root/c5_manifest" \
  "${AF3_SIF}" \
  env PYTHONPATH=/root/benchmark \
  uv run --no-sync python3 -m c5_antibody_ood.af3_preflight inventory \
  --database-dir /root/public_databases \
  --out "/root/c5_manifest/$(basename "${MANIFEST_PARTIAL}")"

test -s "${MANIFEST_PARTIAL}"
readonly MANIFEST_SHA256="$(
  sha256sum "${MANIFEST_PARTIAL}" | awk '{print $1}'
)"
mv "${STAGE}" "${AF3_DB_OUT}"
mv "${MANIFEST_PARTIAL}" "${AF3_DB_MANIFEST_OUT}"
printf '%s  %s\n' \
  "${MANIFEST_SHA256}" \
  "${MANIFEST_NAME}" \
  > "${AF3_DB_MANIFEST_OUT}.sha256"
chmod 0444 \
  "${AF3_DB_MANIFEST_OUT}" \
  "${AF3_DB_MANIFEST_OUT}.sha256"

echo "AlphaFold 3 database provisioning passed"
echo "database_manifest_sha256=${MANIFEST_SHA256}"

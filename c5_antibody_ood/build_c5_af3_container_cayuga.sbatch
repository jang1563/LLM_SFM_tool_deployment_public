#!/bin/bash
#SBATCH --job-name=c5_af3_build
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=c5_antibody_ood/runs/slurm/c5_af3_build_%j.log
#SBATCH --error=c5_antibody_ood/runs/slurm/c5_af3_build_%j.err
#
# Build the pinned AlphaFold 3 v3.0.3 Apptainer image on a Cayuga CPU node.
# Submit with site-specific paths:
#
#   mkdir -p c5_antibody_ood/runs/slurm
#   sbatch --account=<allocation> --partition=scu-cpu \
#     --export=ALL,WORK=$PWD,AF3_SOURCE_DIR=<clean-v3.0.3-source>,AF3_SIF_OUT=<new-image.sif> \
#     c5_antibody_ood/build_c5_af3_container_cayuga.sbatch
#
# The image, build manifest, cache, and scheduler logs are private run
# artifacts and must not be committed.

set -euo pipefail

readonly AF3_COMMIT="7b197fe859790fc3e04d03ea70dd0b9ba48881c9"
readonly AF3_TAG="v3.0.3"
readonly WORK="${WORK:-${SLURM_SUBMIT_DIR:-$PWD}}"
readonly DEFINITION="${WORK}/c5_antibody_ood/alphafold3_v3_0_3.def"

for required in AF3_SOURCE_DIR AF3_SIF_OUT SLURM_JOB_ID
do
  if [ -z "${!required:-}" ]; then
    echo "missing required environment variable: ${required}" >&2
    exit 2
  fi
done

if [ ! -f "${DEFINITION}" ]; then
  echo "pinned Apptainer definition is missing" >&2
  exit 2
fi
if [ -n "$(git -C "${WORK}" status --porcelain)" ]; then
  echo "benchmark source tree must be clean" >&2
  exit 3
fi
readonly BENCHMARK_COMMIT="$(git -C "${WORK}" rev-parse HEAD)"
if [ "$(git -C "${AF3_SOURCE_DIR}" rev-parse HEAD)" != "${AF3_COMMIT}" ]; then
  echo "AlphaFold 3 source commit mismatch" >&2
  exit 3
fi
if [ "$(git -C "${AF3_SOURCE_DIR}" describe --tags --exact-match)" \
  != "${AF3_TAG}" ]
then
  echo "AlphaFold 3 source tag mismatch" >&2
  exit 3
fi
if [ -n "$(git -C "${AF3_SOURCE_DIR}" status --porcelain)" ]; then
  echo "AlphaFold 3 source tree must be clean" >&2
  exit 3
fi

readonly OUTPUT_DIR="$(dirname "${AF3_SIF_OUT}")"
readonly OUTPUT_NAME="$(basename "${AF3_SIF_OUT}")"
readonly PARTIAL_SIF="${AF3_SIF_OUT}.partial.${SLURM_JOB_ID}"
readonly BUILD_MANIFEST="${AF3_SIF_OUT}.build.json"
readonly SOURCE_ROOT="$(realpath "${AF3_SOURCE_DIR}")"
readonly OUTPUT_PATH="$(realpath -m "${AF3_SIF_OUT}")"
if [[ "${OUTPUT_PATH}" = "${SOURCE_ROOT}/"* ]]; then
  echo "container output must be outside the AlphaFold 3 source tree" >&2
  exit 4
fi
shopt -s nullglob
prior_partials=("${AF3_SIF_OUT}".partial.*)
if [ -e "${AF3_SIF_OUT}" ] \
  || [ -e "${AF3_SIF_OUT}.sha256" ] \
  || [ -e "${BUILD_MANIFEST}" ] \
  || [ "${#prior_partials[@]}" -ne 0 ]
then
  echo "container output boundary is not clean" >&2
  exit 4
fi
mkdir -p "${OUTPUT_DIR}"

readonly BUILD_TMP_ROOT="${AF3_BUILD_TMP_ROOT:-${TMPDIR:-/tmp}}"
readonly BUILD_TMP="${BUILD_TMP_ROOT}/c5_af3_build_${SLURM_JOB_ID}"
if [ -e "${BUILD_TMP}" ]; then
  echo "build temporary boundary is not clean" >&2
  exit 4
fi
mkdir -p "${BUILD_TMP}/cache" "${BUILD_TMP}/tmp"
export APPTAINER_CACHEDIR="${BUILD_TMP}/cache"
export APPTAINER_TMPDIR="${BUILD_TMP}/tmp"

cd "${AF3_SOURCE_DIR}"
apptainer build --fakeroot "${PARTIAL_SIF}" "${DEFINITION}"
apptainer test "${PARTIAL_SIF}"
apptainer exec \
  --pwd /app/alphafold \
  "${PARTIAL_SIF}" \
  uv run python3 run_alphafold.py --help >/dev/null

readonly CONTAINER_SHA256="$(
  sha256sum "${PARTIAL_SIF}" | awk '{print $1}'
)"
readonly CONTAINER_BYTES="$(stat --format=%s "${PARTIAL_SIF}")"
readonly DEFINITION_SHA256="$(
  sha256sum "${DEFINITION}" | awk '{print $1}'
)"
readonly APPTAINER_VERSION="$(apptainer --version)"
readonly BUILT_AT_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

mv "${PARTIAL_SIF}" "${AF3_SIF_OUT}"
printf '%s  %s\n' \
  "${CONTAINER_SHA256}" \
  "${OUTPUT_NAME}" \
  > "${AF3_SIF_OUT}.sha256"
printf '{\n  "schema_version": "c5_af3_container_build_v1",\n  "af3_commit": "%s",\n  "af3_tag": "%s",\n  "benchmark_commit": "%s",\n  "definition_sha256": "%s",\n  "container_sha256": "%s",\n  "container_bytes": %s,\n  "apptainer_version": "%s",\n  "built_at_utc": "%s",\n  "local_paths_emitted": false\n}\n' \
  "${AF3_COMMIT}" \
  "${AF3_TAG}" \
  "${BENCHMARK_COMMIT}" \
  "${DEFINITION_SHA256}" \
  "${CONTAINER_SHA256}" \
  "${CONTAINER_BYTES}" \
  "${APPTAINER_VERSION}" \
  "${BUILT_AT_UTC}" \
  > "${BUILD_MANIFEST}"
chmod 0444 \
  "${AF3_SIF_OUT}" \
  "${AF3_SIF_OUT}.sha256" \
  "${BUILD_MANIFEST}"

echo "AlphaFold 3 v3.0.3 container build passed"
echo "container_sha256=${CONTAINER_SHA256}"

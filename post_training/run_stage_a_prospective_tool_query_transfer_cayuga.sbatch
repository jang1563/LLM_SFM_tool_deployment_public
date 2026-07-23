#!/bin/bash
#SBATCH --job-name=stage_a_query_transfer
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#
# Compare base and frozen tool-query SFT policies on public real-query rows.
# Raw generations, model state, reports, and scheduler logs remain private.

set -euo pipefail

WORK="${WORK:?set WORK to the synced public repository root}"
TRAINABLE_STATE="${TRAINABLE_STATE:?set TRAINABLE_STATE outside the public repository}"
PRIVATE_RUN_DIR="${PRIVATE_RUN_DIR:?set PRIVATE_RUN_DIR outside the public repository}"
CONDA_PREFIX_DIR="${CONDA_PREFIX_DIR:?set CONDA_PREFIX_DIR}"
HF_HOME="${HF_HOME:?set HF_HOME to the existing model cache}"

mkdir -p "${PRIVATE_RUN_DIR}/logs"
chmod 700 "${PRIVATE_RUN_DIR}" "${PRIVATE_RUN_DIR}/logs"

export PATH="${CONDA_PREFIX_DIR}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX_DIR}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${WORK}:${PYTHONPATH:-}"
export HF_HOME

cd "${WORK}"

echo "run: prospective real-query base vs frozen tool-query SFT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python -u post_training/run_stage_a_prospective_tool_query_transfer.py \
  --allow-model-load \
  --trainable-state "${TRAINABLE_STATE}" \
  --predictions-out "${PRIVATE_RUN_DIR}/tool_query_transfer_predictions.jsonl" \
  --report-out "${PRIVATE_RUN_DIR}/tool_query_transfer_report.json"

chmod 600 \
  "${PRIVATE_RUN_DIR}/tool_query_transfer_predictions.jsonl" \
  "${PRIVATE_RUN_DIR}/tool_query_transfer_report.json"
find "${PRIVATE_RUN_DIR}/logs" -type f -exec chmod 600 {} +

#!/bin/bash
#SBATCH --job-name=stage_a_sealed_route
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#
# One-time Cayuga evaluation of a frozen Stage A candidate-routing policy.
# All private paths must be supplied through environment variables. Logs,
# row-level predictions, candidate scores, the lock, and model state stay
# outside the public repository.

set -euo pipefail

WORK="${WORK:?set WORK to the synced public repository root}"
SEALED_MANIFEST="${SEALED_MANIFEST:?set SEALED_MANIFEST outside the public repository}"
TRAINABLE_STATE="${TRAINABLE_STATE:?set TRAINABLE_STATE outside the public repository}"
PRIVATE_RUN_DIR="${PRIVATE_RUN_DIR:?set PRIVATE_RUN_DIR outside the public repository}"
POLICY_FREEZE="${POLICY_FREEZE:-post_training/stage_a_candidate_routing_policy_freeze_2026-07-23.json}"
CONDA_PREFIX_DIR="${CONDA_PREFIX_DIR:?set CONDA_PREFIX_DIR}"
HF_HOME="${HF_HOME:?set HF_HOME to the existing model cache}"

mkdir -p "${PRIVATE_RUN_DIR}/logs"
chmod 700 "${PRIVATE_RUN_DIR}" "${PRIVATE_RUN_DIR}/logs"

export PATH="${CONDA_PREFIX_DIR}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX_DIR}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${WORK}:${PYTHONPATH:-}"
export HF_HOME

cd "${WORK}"

echo "run: one-time sealed candidate routing"
echo "freeze: $(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["freeze_id"])' "${POLICY_FREEZE}")"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python -u post_training/run_stage_a_sealed_candidate_routing_eval.py \
  --allow-model-load \
  --sealed-manifest "${SEALED_MANIFEST}" \
  --policy-freeze "${POLICY_FREEZE}" \
  --trainable-state "${TRAINABLE_STATE}" \
  --private-predictions-out "${PRIVATE_RUN_DIR}/sealed_predictions.jsonl" \
  --compact-out "${PRIVATE_RUN_DIR}/sealed_compact_result.json" \
  --evaluation-lock "${PRIVATE_RUN_DIR}/sealed_evaluation.lock.json"

chmod 600 \
  "${PRIVATE_RUN_DIR}/sealed_predictions.jsonl" \
  "${PRIVATE_RUN_DIR}/sealed_compact_result.json" \
  "${PRIVATE_RUN_DIR}/sealed_evaluation.lock.json"

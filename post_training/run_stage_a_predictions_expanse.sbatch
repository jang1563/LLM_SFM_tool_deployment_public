#!/bin/bash
#SBATCH --job-name=stage_a_pred
#SBATCH --partition=gpu-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=logs/stage_a_pred_%x_%j.log
#SBATCH --error=logs/stage_a_pred_%x_%j.err
#
# Expanse Stage A saved-prediction job.
#
# Submit from the synced public repo root, for example:
#
#   sbatch --account=<allocation> \
#     --export=ALL,WORK=$PWD,HF_HOME=<hf-cache-root>,MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct,PROMPT_CONTRACT=stage_a_v4_canonical_json \
#     post_training/run_stage_a_predictions_expanse.sbatch
#
# Use MODEL_ID=Qwen/Qwen2.5-7B-Instruct after the small smoke run succeeds.
# If a model already exists in a site cache, pass HF_HOME explicitly so the job
# does not silently switch to an empty scratch cache.
#
# Raw predictions and eval reports are written under post_training/runs/ and
# should not be committed.

set -euo pipefail

WORK="${WORK:-${SLURM_SUBMIT_DIR:-$PWD}}"
JOB_ID="${SLURM_JOB_ID:-manual}"
RUN_ID="${RUN_ID:-stage_a_expanse_hf_chat_${JOB_ID}}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-0.5B-Instruct}"
SFT="${SFT:-post_training/stage_a_sft_heldout_v1.jsonl}"
OUT_DIR="${OUT_DIR:-post_training/runs/${RUN_ID}}"
PREDICTIONS="${PREDICTIONS:-${OUT_DIR}/predictions.jsonl}"
EVAL_REPORT="${EVAL_REPORT:-${OUT_DIR}/eval_report.json}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
PROMPT_CONTRACT="${PROMPT_CONTRACT:-basic}"
DEVICE="${DEVICE:-auto}"
CONDA_PREFIX_DIR="${CONDA_PREFIX_DIR:-${PROJECT_DIR:-${SCRATCH:-/scratch}/${USER}}/conda_envs/${ENV:-stage_a_transformers}}"
ALLOW_DOWNLOAD="${ALLOW_DOWNLOAD:-0}"

cd "${WORK}"
mkdir -p logs "${OUT_DIR}"

if command -v module >/dev/null 2>&1; then
  module purge || true
  module load gpu || true
  module load slurm || true
fi

if [ -d "${CONDA_PREFIX_DIR}" ]; then
  export PATH="${CONDA_PREFIX_DIR}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CONDA_PREFIX_DIR}/lib:${LD_LIBRARY_PATH:-}"
fi

export PYTHONPATH="${WORK}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-${SCRATCH:-/scratch}/${USER}/hf_home}"
mkdir -p "${HF_HOME}"

DOWNLOAD_ARG=""
if [ "${ALLOW_DOWNLOAD}" = "1" ]; then
  DOWNLOAD_ARG="--allow-download"
fi

echo "python: $(command -v python) ($(python --version 2>&1))"
echo "work: ${WORK}"
echo "run_id: ${RUN_ID}"
echo "model: ${MODEL_ID}"
echo "prompt_contract: ${PROMPT_CONTRACT}"
echo "sft: ${SFT}"
echo "predictions: ${PREDICTIONS}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python -u post_training/generate_stage_a_predictions.py \
  --mode hf_chat \
  --allow-model-load \
  --model "${MODEL_ID}" \
  --device "${DEVICE}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --prompt-contract "${PROMPT_CONTRACT}" \
  --sft "${SFT}" \
  --out "${PREDICTIONS}" \
  --run-id "${RUN_ID}" \
  ${DOWNLOAD_ARG}

python -u post_training/evaluate_stage_a_predictions.py \
  --predictions "${PREDICTIONS}" \
  --expected-sft "${SFT}" \
  --run-id "${RUN_ID}" \
  --out "${EVAL_REPORT}" \
  --json

echo "wrote ${PREDICTIONS}"
echo "wrote ${EVAL_REPORT}"

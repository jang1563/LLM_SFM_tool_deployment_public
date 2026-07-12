#!/bin/bash
#SBATCH --job-name=stage_a_strict_sft
#SBATCH --partition=gpu-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=logs/stage_a_strict_sft_%x_%j.log
#SBATCH --error=logs/stage_a_strict_sft_%x_%j.err
#
# Expanse Stage A strict-contract SFT smoke job.
#
# Submit from the synced public repo root, for example:
#
#   sbatch --account=<allocation> \
#     --export=ALL,WORK=$PWD,MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct \
#     post_training/run_stage_a_strict_sft_expanse.sbatch
#
# This trains only a tiny last-layer smoke checkpoint on strict-contract Stage A
# JSON targets, generates held-out predictions, and scores them offline. Raw
# run artifacts are written under post_training/runs/ and should not be
# committed.

set -euo pipefail

WORK="${WORK:-${SLURM_SUBMIT_DIR:-$PWD}}"
JOB_ID="${SLURM_JOB_ID:-manual}"
RUN_ID="${RUN_ID:-stage_a_strict_sft_expanse_${JOB_ID}}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-0.5B-Instruct}"
TRAIN_SFT="${TRAIN_SFT:-post_training/stage_a_strict_contract_sft_train_v1.jsonl}"
HELDOUT_SFT="${HELDOUT_SFT:-post_training/stage_a_strict_contract_sft_heldout_v1.jsonl}"
MANIFEST="${MANIFEST:-negbiodb_ct/stage_a_mini_manifest.jsonl}"
OUT_DIR="${OUT_DIR:-post_training/runs/${RUN_ID}}"
MAX_STEPS="${MAX_STEPS:-20}"
BATCH_SIZE="${BATCH_SIZE:-1}"
LR="${LR:-1e-5}"
TRAIN_LAST_LAYERS="${TRAIN_LAST_LAYERS:-1}"
MAX_LENGTH="${MAX_LENGTH:-1536}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
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
echo "train_sft: ${TRAIN_SFT}"
echo "heldout_sft: ${HELDOUT_SFT}"
echo "out_dir: ${OUT_DIR}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python -u post_training/run_stage_a_strict_contract_sft_smoke.py \
  --allow-model-load \
  --model "${MODEL_ID}" \
  --train-sft "${TRAIN_SFT}" \
  --heldout-sft "${HELDOUT_SFT}" \
  --manifest "${MANIFEST}" \
  --out-dir "${OUT_DIR}" \
  --run-id "${RUN_ID}" \
  --max-steps "${MAX_STEPS}" \
  --batch-size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --train-last-layers "${TRAIN_LAST_LAYERS}" \
  --max-length "${MAX_LENGTH}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --device "${DEVICE}" \
  ${DOWNLOAD_ARG}

echo "wrote ${OUT_DIR}/report.json"
echo "wrote ${OUT_DIR}/predictions.jsonl"
echo "wrote ${OUT_DIR}/eval_report.json"

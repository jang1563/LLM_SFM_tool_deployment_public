#!/bin/bash
#SBATCH --job-name=stage_a_probe_readout
#SBATCH --partition=gpu-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=logs/stage_a_probe_readout_%x_%j.log
#SBATCH --error=logs/stage_a_probe_readout_%x_%j.err
#
# Expanse Stage A saved-output calibration probe readout job.
#
# Submit from the synced public repo root, for example:
#
#   sbatch --account=<allocation> \
#     --export=ALL,WORK=$PWD,HF_HOME=<hf-cache-root>,MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct \
#     post_training/run_stage_a_saved_output_calibration_probe_readout_expanse.sbatch
#
# This scores target-vs-ground/supported probe pairs for calibration diagnosis.
# Raw candidate-score readouts, reports, model state, and scheduler logs are
# written under ignored paths and should not be committed.

set -euo pipefail

WORK="${WORK:-${SLURM_SUBMIT_DIR:-$PWD}}"
JOB_ID="${SLURM_JOB_ID:-manual}"
RUN_ID="${RUN_ID:-stage_a_saved_output_calibration_probe_readout_expanse_${JOB_ID}}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-0.5B-Instruct}"
PAIRS="${PAIRS:-post_training/stage_a_saved_output_calibration_probe_v1.jsonl}"
TRAIN_PAIRS="${TRAIN_PAIRS:-post_training/stage_a_saved_output_calibration_probe_train_v1.jsonl}"
HELDOUT_PAIRS="${HELDOUT_PAIRS:-post_training/stage_a_saved_output_calibration_probe_heldout_v1.jsonl}"
MANIFEST="${MANIFEST:-post_training/stage_a_saved_output_calibration_probe_manifest.json}"
OUT_DIR="${OUT_DIR:-post_training/runs/${RUN_ID}}"
PROMPT_CONTRACT="${PROMPT_CONTRACT:-stage_a_v4_canonical_json}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
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
echo "out_dir: ${OUT_DIR}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python -u post_training/run_stage_a_saved_output_calibration_probe_readout.py \
  --allow-model-load \
  --model "${MODEL_ID}" \
  --device "${DEVICE}" \
  --max-length "${MAX_LENGTH}" \
  --prompt-contract "${PROMPT_CONTRACT}" \
  --pairs "${PAIRS}" \
  --train-pairs "${TRAIN_PAIRS}" \
  --heldout-pairs "${HELDOUT_PAIRS}" \
  --manifest "${MANIFEST}" \
  --out-dir "${OUT_DIR}" \
  --run-id "${RUN_ID}" \
  ${DOWNLOAD_ARG}

echo "wrote ${OUT_DIR}/readout.jsonl"
echo "wrote ${OUT_DIR}/report.json"
echo "wrote ${OUT_DIR}/REPORT.md"

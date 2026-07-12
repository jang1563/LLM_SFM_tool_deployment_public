#!/bin/bash
#SBATCH --job-name=stage_a_component_sft
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=logs/stage_a_component_sft_%x_%j.log
#SBATCH --error=logs/stage_a_component_sft_%x_%j.err
#
# Cayuga Stage A strict component-slice SFT smoke job.
#
# Submit from the synced public repo root, for example:
#
#   sbatch --account=<allocation> --partition=<gpu-partition> --gres=gpu:1 \
#     --export=ALL,WORK=$PWD,COMPONENT=enum_action,MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct \
#     post_training/run_stage_a_strict_component_sft_cayuga.sbatch
#
# COMPONENT can be enum_action, tool_query, or routing_after_loop. Set
# DECODE_MODE=enum_candidate_score for the full enum_action Cartesian scorer, or
# DECODE_MODE=enum_observed_pair_score for train-observed valid pairs only.
# Set DECODE_MODE=routing_observed_pair_score for constrained routing-after-loop
# action/status scoring with visible citation candidates.
# Raw run artifacts are written under post_training/runs/ and should not be
# committed.

set -euo pipefail

WORK="${WORK:-${SLURM_SUBMIT_DIR:-$PWD}}"
JOB_ID="${SLURM_JOB_ID:-manual}"
COMPONENT="${COMPONENT:-enum_action}"
RUN_ID="${RUN_ID:-stage_a_component_${COMPONENT}_sft_cayuga_${JOB_ID}}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-0.5B-Instruct}"
TARGETS="${TARGETS:-post_training/stage_a_strict_component_targets_v1.jsonl}"
TRAIN_TARGETS="${TRAIN_TARGETS:-post_training/stage_a_strict_component_targets_train_v1.jsonl}"
HELDOUT_TARGETS="${HELDOUT_TARGETS:-post_training/stage_a_strict_component_targets_heldout_v1.jsonl}"
OUT_DIR="${OUT_DIR:-post_training/runs/${RUN_ID}}"
MAX_STEPS="${MAX_STEPS:-20}"
BATCH_SIZE="${BATCH_SIZE:-1}"
LR="${LR:-1e-5}"
TRAIN_LAST_LAYERS="${TRAIN_LAST_LAYERS:-1}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
DECODE_MODE="${DECODE_MODE:-freeform}"
DEVICE="${DEVICE:-auto}"
CONDA_PREFIX_DIR="${CONDA_PREFIX_DIR:-${PROJECT_DIR:-${SCRATCH:-/scratch}/${USER}}/conda_envs/${ENV:-stage_a_transformers}}"
ALLOW_DOWNLOAD="${ALLOW_DOWNLOAD:-0}"

cd "${WORK}"
mkdir -p logs "${OUT_DIR}"

if command -v module >/dev/null 2>&1; then
  module load cuda || true
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
echo "component: ${COMPONENT}"
echo "model: ${MODEL_ID}"
echo "train_targets: ${TRAIN_TARGETS}"
echo "heldout_targets: ${HELDOUT_TARGETS}"
echo "out_dir: ${OUT_DIR}"
echo "decode_mode: ${DECODE_MODE}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python -u post_training/run_stage_a_strict_component_sft_smoke.py \
  --allow-model-load \
  --model "${MODEL_ID}" \
  --component "${COMPONENT}" \
  --targets "${TARGETS}" \
  --train-targets "${TRAIN_TARGETS}" \
  --heldout-targets "${HELDOUT_TARGETS}" \
  --out-dir "${OUT_DIR}" \
  --run-id "${RUN_ID}" \
  --max-steps "${MAX_STEPS}" \
  --batch-size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --train-last-layers "${TRAIN_LAST_LAYERS}" \
  --max-length "${MAX_LENGTH}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --decode-mode "${DECODE_MODE}" \
  --device "${DEVICE}" \
  ${DOWNLOAD_ARG}

echo "wrote ${OUT_DIR}/report.json"
echo "wrote ${OUT_DIR}/predictions.jsonl"
echo "wrote ${OUT_DIR}/eval_report.json"

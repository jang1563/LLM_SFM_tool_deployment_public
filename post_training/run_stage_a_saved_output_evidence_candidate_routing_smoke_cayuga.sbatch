#!/bin/bash
#SBATCH --job-name=stage_a_evidence_route
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=logs/stage_a_evidence_route_%x_%j.log
#SBATCH --error=logs/stage_a_evidence_route_%x_%j.err
#
# Cayuga Stage A evidence-conditioned candidate-routing SFT smoke job.
#
# Submit from the synced public repo root only after dry-run validation, for
# example:
#
#   sbatch --account=<allocation> --partition=<gpu-partition> --gres=gpu:1 \
#     --export=ALL,WORK=$PWD,MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct \
#     post_training/run_stage_a_saved_output_evidence_candidate_routing_smoke_cayuga.sbatch
#
# Raw candidate scores, trainable state, and scheduler logs are written under
# ignored paths and should not be committed. This job is an SFT smoke, not
# DPO/RLVR.

set -euo pipefail

WORK="${WORK:-${SLURM_SUBMIT_DIR:-$PWD}}"
JOB_ID="${SLURM_JOB_ID:-manual}"
RUN_ID="${RUN_ID:-stage_a_evidence_candidate_routing_qwen05b_cayuga_${JOB_ID}}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-0.5B-Instruct}"
ROWS="${ROWS:-post_training/stage_a_saved_output_evidence_candidate_routing_rows_v1.jsonl}"
TRAIN_ROWS="${TRAIN_ROWS:-post_training/stage_a_saved_output_evidence_candidate_routing_train_v1.jsonl}"
HELDOUT_ROWS="${HELDOUT_ROWS:-post_training/stage_a_saved_output_evidence_candidate_routing_heldout_v1.jsonl}"
MANIFEST="${MANIFEST:-post_training/stage_a_saved_output_evidence_candidate_routing_manifest.json}"
OUT_DIR="${OUT_DIR:-post_training/runs/${RUN_ID}}"
MAX_STEPS="${MAX_STEPS:-40}"
BATCH_SIZE="${BATCH_SIZE:-1}"
LR="${LR:-1e-5}"
TRAIN_LAST_LAYERS="${TRAIN_LAST_LAYERS:-1}"
MAX_LENGTH="${MAX_LENGTH:-1536}"
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
echo "model: ${MODEL_ID}"
echo "train_rows: ${TRAIN_ROWS}"
echo "heldout_rows: ${HELDOUT_ROWS}"
echo "out_dir: ${OUT_DIR}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python -u post_training/run_stage_a_saved_output_evidence_candidate_routing_smoke.py \
  --allow-model-load \
  --model "${MODEL_ID}" \
  --rows "${ROWS}" \
  --train-rows "${TRAIN_ROWS}" \
  --heldout-rows "${HELDOUT_ROWS}" \
  --manifest "${MANIFEST}" \
  --out-dir "${OUT_DIR}" \
  --run-id "${RUN_ID}" \
  --max-steps "${MAX_STEPS}" \
  --batch-size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --train-last-layers "${TRAIN_LAST_LAYERS}" \
  --max-length "${MAX_LENGTH}" \
  --device "${DEVICE}" \
  ${DOWNLOAD_ARG}

echo "wrote ${OUT_DIR}/report.json"
echo "wrote ${OUT_DIR}/train_candidates.jsonl"
echo "wrote ${OUT_DIR}/heldout_candidates.jsonl"
echo "wrote ${OUT_DIR}/eval_report.json"

#!/bin/bash
#SBATCH --job-name=stage_a_probe_margin
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=logs/stage_a_probe_margin_%x_%j.log
#SBATCH --error=logs/stage_a_probe_margin_%x_%j.err
#
# Expanse fallback for Stage A saved-output calibration margin SFT smoke.
#
# Submit from the synced public repo root, for example:
#
#   sbatch --account=<allocation> --partition=<gpu-partition> --gres=gpu:1 \
#     --export=ALL,WORK=$PWD,MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct \
#     post_training/run_stage_a_saved_output_calibration_margin_sft_expanse.sbatch
#
# Raw margins, trainable state, and scheduler logs are written under ignored
# paths and should not be committed. This job is an SFT/margin diagnostic, not
# DPO/RLVR.

set -euo pipefail

WORK="${WORK:-${SLURM_SUBMIT_DIR:-$PWD}}"
JOB_ID="${SLURM_JOB_ID:-manual}"
RUN_ID="${RUN_ID:-stage_a_saved_output_calibration_margin_sft_expanse_${JOB_ID}}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-0.5B-Instruct}"
PAIRS="${PAIRS:-post_training/stage_a_saved_output_calibration_probe_v1.jsonl}"
TRAIN_PAIRS="${TRAIN_PAIRS:-post_training/stage_a_saved_output_calibration_probe_train_v1.jsonl}"
HELDOUT_PAIRS="${HELDOUT_PAIRS:-post_training/stage_a_saved_output_calibration_probe_heldout_v1.jsonl}"
MANIFEST="${MANIFEST:-post_training/stage_a_saved_output_calibration_probe_manifest.json}"
OUT_DIR="${OUT_DIR:-post_training/runs/${RUN_ID}}"
MAX_STEPS="${MAX_STEPS:-20}"
BATCH_SIZE="${BATCH_SIZE:-1}"
LR="${LR:-1e-5}"
TRAIN_LAST_LAYERS="${TRAIN_LAST_LAYERS:-1}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
DEVICE="${DEVICE:-auto}"
CONDA_PREFIX_DIR="${CONDA_PREFIX_DIR:-${PROJECT_DIR:-${SCRATCH:-/scratch}/${USER}}/conda_envs/${ENV:-stage_a_transformers}}"
ALLOW_DOWNLOAD="${ALLOW_DOWNLOAD:-0}"
SCORE_BASE_MARGINS="${SCORE_BASE_MARGINS:-1}"
SCORE_TRAIN_MARGINS="${SCORE_TRAIN_MARGINS:-1}"
SCORE_BASE_CANDIDATES="${SCORE_BASE_CANDIDATES:-0}"
SCORE_TRAIN_CANDIDATES="${SCORE_TRAIN_CANDIDATES:-0}"
SCORE_TRAINED_CANDIDATES="${SCORE_TRAINED_CANDIDATES:-0}"
FOCUS_CHOSEN_PAIRS="${FOCUS_CHOSEN_PAIRS:-}"
FOCUS_REPEAT="${FOCUS_REPEAT:-1}"
FOCUS_ONLY="${FOCUS_ONLY:-0}"
PAIRWISE_MARGIN_WEIGHT="${PAIRWISE_MARGIN_WEIGHT:-1}"
PAIRWISE_MARGIN="${PAIRWISE_MARGIN:-0.05}"
MARGIN_LOGPROB_MODE="${MARGIN_LOGPROB_MODE:-mean}"
CANDIDATE_CE_WEIGHT="${CANDIDATE_CE_WEIGHT:-0}"
CANDIDATE_CE_MODE="${CANDIDATE_CE_MODE:-pair}"
CANDIDATE_CE_LOGPROB_MODE="${CANDIDATE_CE_LOGPROB_MODE:-mean}"
TARGET_FORMAT="${TARGET_FORMAT:-full}"
SCORE_TARGET_FORMATS="${SCORE_TARGET_FORMATS:-}"
CANDIDATE_POLICY="${CANDIDATE_POLICY:-train_observed_plus_rejected}"
CANDIDATE_TARGET_FORMAT="${CANDIDATE_TARGET_FORMAT:-full}"

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
BASE_MARGIN_ARG=""
if [ "${SCORE_BASE_MARGINS}" = "1" ]; then
  BASE_MARGIN_ARG="--score-base-margins"
fi
TRAIN_MARGIN_ARG=""
if [ "${SCORE_TRAIN_MARGINS}" = "1" ]; then
  TRAIN_MARGIN_ARG="--score-train-margins"
fi
BASE_CANDIDATE_ARG=""
if [ "${SCORE_BASE_CANDIDATES}" = "1" ]; then
  BASE_CANDIDATE_ARG="--score-base-candidates"
fi
TRAIN_CANDIDATE_ARG=""
if [ "${SCORE_TRAIN_CANDIDATES}" = "1" ]; then
  TRAIN_CANDIDATE_ARG="--score-train-candidates"
fi
TRAINED_CANDIDATE_ARG=""
if [ "${SCORE_TRAINED_CANDIDATES}" = "1" ]; then
  TRAINED_CANDIDATE_ARG="--score-trained-candidates"
fi
FOCUS_ARG=""
if [ -n "${FOCUS_CHOSEN_PAIRS}" ]; then
  FOCUS_ARG="--focus-chosen-pairs ${FOCUS_CHOSEN_PAIRS} --focus-repeat ${FOCUS_REPEAT}"
fi
FOCUS_ONLY_ARG=""
if [ "${FOCUS_ONLY}" = "1" ]; then
  FOCUS_ONLY_ARG="--focus-only"
fi

echo "python: $(command -v python) ($(python --version 2>&1))"
echo "work: ${WORK}"
echo "run_id: ${RUN_ID}"
echo "model: ${MODEL_ID}"
echo "train_pairs: ${TRAIN_PAIRS}"
echo "heldout_pairs: ${HELDOUT_PAIRS}"
echo "out_dir: ${OUT_DIR}"
echo "score_base_margins: ${SCORE_BASE_MARGINS}"
echo "score_train_margins: ${SCORE_TRAIN_MARGINS}"
echo "score_base_candidates: ${SCORE_BASE_CANDIDATES}"
echo "score_train_candidates: ${SCORE_TRAIN_CANDIDATES}"
echo "score_trained_candidates: ${SCORE_TRAINED_CANDIDATES}"
echo "focus_chosen_pairs: ${FOCUS_CHOSEN_PAIRS}"
echo "focus_repeat: ${FOCUS_REPEAT}"
echo "focus_only: ${FOCUS_ONLY}"
echo "pairwise_margin_weight: ${PAIRWISE_MARGIN_WEIGHT}"
echo "pairwise_margin: ${PAIRWISE_MARGIN}"
echo "margin_logprob_mode: ${MARGIN_LOGPROB_MODE}"
echo "candidate_ce_weight: ${CANDIDATE_CE_WEIGHT}"
echo "candidate_ce_mode: ${CANDIDATE_CE_MODE}"
echo "candidate_ce_logprob_mode: ${CANDIDATE_CE_LOGPROB_MODE}"
echo "target_format: ${TARGET_FORMAT}"
echo "score_target_formats: ${SCORE_TARGET_FORMATS}"
echo "candidate_policy: ${CANDIDATE_POLICY}"
echo "candidate_target_format: ${CANDIDATE_TARGET_FORMAT}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python -u post_training/run_stage_a_saved_output_calibration_margin_sft.py \
  --allow-model-load \
  --model "${MODEL_ID}" \
  --pairs "${PAIRS}" \
  --train-pairs "${TRAIN_PAIRS}" \
  --heldout-pairs "${HELDOUT_PAIRS}" \
  --manifest "${MANIFEST}" \
  --out-dir "${OUT_DIR}" \
  --run-id "${RUN_ID}" \
  --max-steps "${MAX_STEPS}" \
  --batch-size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --train-last-layers "${TRAIN_LAST_LAYERS}" \
  --max-length "${MAX_LENGTH}" \
  --device "${DEVICE}" \
  --pairwise-margin-weight "${PAIRWISE_MARGIN_WEIGHT}" \
  --pairwise-margin "${PAIRWISE_MARGIN}" \
  --margin-logprob-mode "${MARGIN_LOGPROB_MODE}" \
  --candidate-ce-weight "${CANDIDATE_CE_WEIGHT}" \
  --candidate-ce-mode "${CANDIDATE_CE_MODE}" \
  --candidate-ce-logprob-mode "${CANDIDATE_CE_LOGPROB_MODE}" \
  --target-format "${TARGET_FORMAT}" \
  --score-target-formats "${SCORE_TARGET_FORMATS}" \
  --candidate-policy "${CANDIDATE_POLICY}" \
  --candidate-target-format "${CANDIDATE_TARGET_FORMAT}" \
  ${BASE_MARGIN_ARG} \
  ${TRAIN_MARGIN_ARG} \
  ${BASE_CANDIDATE_ARG} \
  ${TRAIN_CANDIDATE_ARG} \
  ${TRAINED_CANDIDATE_ARG} \
  ${FOCUS_ARG} \
  ${FOCUS_ONLY_ARG} \
  ${DOWNLOAD_ARG}

echo "wrote ${OUT_DIR}/report.json"
echo "wrote ${OUT_DIR}/base_margins.jsonl if SCORE_BASE_MARGINS=1"
echo "wrote ${OUT_DIR}/train_margins.jsonl if SCORE_TRAIN_MARGINS=1"
echo "wrote ${OUT_DIR}/margins.jsonl"
echo "wrote ${OUT_DIR}/margin_report.json"
echo "wrote ${OUT_DIR}/margin_delta_report.json if SCORE_BASE_MARGINS=1"
echo "wrote ${OUT_DIR}/base_candidates.jsonl if SCORE_BASE_CANDIDATES=1"
echo "wrote ${OUT_DIR}/train_candidates.jsonl if SCORE_TRAIN_CANDIDATES=1"
echo "wrote ${OUT_DIR}/candidates.jsonl if SCORE_TRAINED_CANDIDATES=1"

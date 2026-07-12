#!/bin/bash
#SBATCH --job-name=ct_traj
#SBATCH --partition=<gpu-partition>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=h100:1                 # default = qwen14b (1 GPU). For llama70b override at submit:
#SBATCH --mem=96G                     #   sbatch --gpus=h100:4 --mem=384G --export=ALL,MODEL=llama70b run_ct_trajectory_expanse.sbatch
#SBATCH --time=2:00:00
# Set your site allocation at submit time if required, for example:
#   sbatch --account=<allocation> run_ct_trajectory_expanse.sbatch
#SBATCH --output=logs/ct_traj_%x_%j.log
#SBATCH --error=logs/ct_traj_%x_%j.err
#
# (c) Gate-1 prompting-only headroom for the NegBioDB-CT trajectory tasks, open models.
# Serves a model via vLLM WITH tool-use (--enable-auto-tool-choice --tool-call-parser), waits for
# readiness, then runs run_agent_open.py (OpenAI-compatible tool loop) over the 400 balanced tasks.
# This is the open-model counterpart of the Sonnet reference (0.95) -> the headroom SFT must close.
#
# Budget: qwen14b ~0.5 GPU-h (1xH100); llama70b ~1 GPU-h (4xH100). Well within the 1,800 cap.
#
# PREREQS (sync once):
#   - ${WORK} = this LLM_SFM_tool_deployment/ tree (rsync; carries negbiodb_ct/, llm_sfm_tool_deployment/, tasks_pilot.jsonl)
#   - ${REPO}/data/negbiodb_ct.db  (already part of the Expanse repo sync; see slurm_nairr/d1_*)
#   - models in ${MODEL_DIR}: Llama-3.3-70B-Instruct (present) + Qwen2.5-14B-Instruct (download if absent)
#   - conda env d_spec_vllm_cu124 (vllm 0.8.5 + openai) — activated by PATH directly, not `conda activate`
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SCRATCH:-/scratch}/${USER}}"
REPO="${REPO:-${PROJECT_DIR}/NegBioDB}"
WORK="${WORK:-${PROJECT_DIR}/ct_trajectory}"
CONDA_PREFIX_DIR="${PROJECT_DIR}/conda_envs/${ENV:-d_spec_vllm_cu124}"
MODEL_DIR="${MODEL_DIR:-${PROJECT_DIR}/models}"
MODEL="${MODEL:-qwen14b}"
PORT="${PORT:-$((8001 + SLURM_JOB_ID % 999))}"   # per-job port -> concurrent jobs on a shared node don't collide
N="${N:-400}"

case "${MODEL}" in
    qwen14b)                                                 # bf16, ~28GB -> 1xH100
        MODEL_PATH="${MODEL_DIR}/Qwen2.5-14B-Instruct"; HF_REPO="Qwen/Qwen2.5-14B-Instruct"; TP=1; PARSER="hermes"; SERVED="qwen14b"
        QUANT_ARG=""; TEMPLATE_ARG="" ;;                     # Qwen2.5 chat template is tool-aware by default
    qwen32b)                                                 # AWQ 4-bit, ~20GB -> 1xH100 (present on Expanse)
        MODEL_PATH="${MODEL_DIR}/Qwen2.5-32B-Instruct-AWQ"; HF_REPO="Qwen/Qwen2.5-32B-Instruct-AWQ"; TP=1; PARSER="hermes"; SERVED="qwen32b"
        QUANT_ARG="--quantization awq"; TEMPLATE_ARG="" ;;
    llama70b)                                                # bf16, ~140GB -> 4xH100 (NOT on Expanse; download first)
        MODEL_PATH="${MODEL_DIR}/Llama-3.3-70B-Instruct"; HF_REPO="meta-llama/Llama-3.3-70B-Instruct"; TP=4; PARSER="llama3_json"; SERVED="llama70b"
        # Llama-3.3 may need the tool chat template if the built-in one drops tool turns; override:
        #   export TOOL_TEMPLATE=/path/to/tool_chat_template_llama3.1_json.jinja
        QUANT_ARG=""; TEMPLATE_ARG="${TOOL_TEMPLATE:+--chat-template ${TOOL_TEMPLATE}}" ;;
    *) echo "Unknown MODEL=${MODEL}. Use qwen14b | qwen32b | llama70b"; exit 1 ;;
esac

# conda is NOT on PATH on Expanse compute nodes -> activate the env by PATH directly (no conda needed)
export PATH="${CONDA_PREFIX_DIR}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX_DIR}/lib:${LD_LIBRARY_PATH:-}"
hash -r
echo "python: $(command -v python) ($(python --version 2>&1))"
mkdir -p logs
export HF_HOME="/scratch/${USER}/job_${SLURM_JOB_ID}/hf_home"; mkdir -p "${HF_HOME}"

# read-only sqlite over lustre can deadlock on locks -> stage the DB on node-local scratch
LOCAL_DB="/scratch/${USER}/job_${SLURM_JOB_ID}/negbiodb_ct.db"
cp "${CT_DB_SRC:-${WORK}/negbiodb_ct.db}" "${LOCAL_DB}"

echo "=== CT trajectory (open-model Gate-1) ==="
echo "Model: ${MODEL} (${MODEL_PATH}) | TP=${TP} | parser=${PARSER} | N=${N}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# 0) ensure the model weights are COMPLETE -- download on-node if missing/partial (robust vs the
#    login-node OOM; HF_TRANSFER off avoids the Rust over-allocation; resumes any partial). A bare
#    config.json is NOT enough: the earlier partial had config.json but 0/8 shards -> vLLM load fails.
need_dl=0
ls "${MODEL_PATH}"/*.safetensors >/dev/null 2>&1 || need_dl=1
idx="${MODEL_PATH}/model.safetensors.index.json"
if [ -f "${idx}" ]; then
    for shard in $(grep -oE 'model-[0-9]+-of-[0-9]+\.safetensors' "${idx}" | sort -u); do
        [ -f "${MODEL_PATH}/${shard}" ] || need_dl=1
    done
fi
if [ "${need_dl}" = 1 ]; then
    echo "Model incomplete -> downloading ${HF_REPO} on node"
    HF_HUB_ENABLE_HF_TRANSFER=0 huggingface-cli download "${HF_REPO}" --local-dir "${MODEL_PATH}"
    echo "download done: $(du -sh "${MODEL_PATH}" 2>/dev/null | cut -f1)"
fi

# 1) serve vLLM with tool-use, in the background
python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_PATH}" --served-model-name "${SERVED}" \
    --host 127.0.0.1 --port "${PORT}" \
    --tensor-parallel-size "${TP}" --max-model-len 8192 \
    --gpu-memory-utilization 0.92 --dtype auto --trust-remote-code --enforce-eager \
    --enable-auto-tool-choice --tool-call-parser "${PARSER}" ${TEMPLATE_ARG} ${QUANT_ARG} \
    > "logs/vllm_${MODEL}_${SLURM_JOB_ID}.log" 2>&1 &
VLLM_PID=$!

# 2) wait for readiness (70B weight load can take ~10 min)
ready=0
for i in $(seq 1 120); do
    if curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
        echo "vLLM ready after ~$((i*10))s"; ready=1; break
    fi
    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "ERROR: vLLM exited during load -- see logs/vllm_${MODEL}_${SLURM_JOB_ID}.log"; exit 1
    fi
    sleep 10
done
[ "${ready}" = 1 ] || { echo "ERROR: vLLM not ready in 20 min"; kill "${VLLM_PID}" 2>/dev/null || true; exit 1; }

# 3) run the trajectory eval (prompting-only headroom)
cd "${WORK}/negbiodb_ct"
if [ -n "${TIER:-}" ]; then          # fuzzy-retrieval mode: resolver-backed tools, tier = exact|fuzzy|synonym
    python -u run_agent_open_fuzzy.py \
        --model "${SERVED}" --base-url "http://127.0.0.1:${PORT}/v1" \
        --tier "${TIER}" --n "${N}" --ct-db "${LOCAL_DB}" \
        --tasks "${WORK}/negbiodb_ct/tasks_fuzzy.jsonl" \
        --out "${WORK}/negbiodb_ct/fuzzy_${MODEL}_${TIER}.json"
else                                 # oracle mode (G1)
    python -u run_agent_open.py \
        --model "${SERVED}" --base-url "http://127.0.0.1:${PORT}/v1" \
        --n "${N}" --ct-db "${LOCAL_DB}" \
        --tasks "${WORK}/negbiodb_ct/tasks_pilot.jsonl" \
        --out "${WORK}/negbiodb_ct/open_${MODEL}.json"
fi

kill "${VLLM_PID}" 2>/dev/null || true
echo "=== DONE ${MODEL} -> open_${MODEL}.json ==="

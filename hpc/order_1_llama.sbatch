#!/bin/bash
#SBATCH --job-name=order_1_llama
#SBATCH --nodes=1
#SBATCH --partition=l40s
#SBATCH --gpus=1              
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G                 # Bumped for 7B parameter loading
#SBATCH --time=24:00:00         
#SBATCH --output=logs/%j_slurm_order_1_llama.out

set -euo pipefail

echo "Starting order_1_llama execution on $HOSTNAME at $(date)"

# --- 1. Storage & Environment Setup ---
USER_NAME="${SLURM_JOB_USER:-${USER:-${LOGNAME:-$(whoami)}}}"

# Persistent cache on scratch.hpc (project directory)
PERSISTENT_CACHE="$PWD/.scratch_cache"
mkdir -p logs logs_and_outputs_llama/order_1/logs "$PERSISTENT_CACHE/huggingface" "$PERSISTENT_CACHE/torch"

# Node-local storage for sockets and lock files
NODE_TMP="${SLURM_TMPDIR:-/tmp/${USER_NAME}_${SLURM_JOB_ID:-standalone}}"
mkdir -p "$NODE_TMP"

# Automatically clean up node-local temp space when the job exits
trap 'rm -rf "$NODE_TMP"' EXIT

# CPU Thread Control (Prevents thread oversubscription)
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

# Dynamic DeepSpeed Port
export MASTER_PORT=$((29500 + ${SLURM_JOB_ID:-0} % 10000))

# Hugging Face Token (Set this in your terminal beforehand if using gated Meta repos: export HF_TOKEN="hf_...")
export HF_TOKEN="${HF_TOKEN:-}"

# Export modern Hugging Face cache paths
export HF_HOME="$PERSISTENT_CACHE/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_ASSETS_CACHE="$HF_HOME/assets"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TORCH_HOME="$PERSISTENT_CACHE/torch"
export TMPDIR="$NODE_TMP"

# Pass ALL cache paths into the Apptainer container environment
export APPTAINERENV_HF_HOME="$HF_HOME"
export APPTAINERENV_HF_HUB_CACHE="$HF_HUB_CACHE"
export APPTAINERENV_HF_ASSETS_CACHE="$HF_ASSETS_CACHE"
export APPTAINERENV_TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE"
export APPTAINERENV_HF_DATASETS_CACHE="$HF_DATASETS_CACHE"
export APPTAINERENV_TORCH_HOME="$TORCH_HOME"
export APPTAINERENV_TMPDIR="$TMPDIR"

CONTAINER="./pytorch_2.4.0-cuda12.1-cudnn9-devel.sif"

# Patch scripts_llama/order_1.sh for single-GPU execution and set -e safety if needed
if [ -f "scripts_llama/order_1.sh" ]; then
    sed -i 's/CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7/CUDA_VISIBLE_DEVICES=0/g' scripts_llama/order_1.sh
    grep -q "set -e" scripts_llama/order_1.sh || sed -i '2i set -e' scripts_llama/order_1.sh
fi

# --- 2. Check-and-Download Base Model ---
MODEL_DIR="initial_model/llama"

# Ungated LLaMA-2 7B mirror (or meta-llama/Llama-2-7b-hf)
MODEL_ID="${MODEL_ID:-NousResearch/Llama-2-7b-hf}"

if [ ! -f "$MODEL_DIR/config.json" ] || { [ ! -f "$MODEL_DIR/model.safetensors.index.json" ] && [ ! -f "$MODEL_DIR/pytorch_model.bin.index.json" ] && [ ! -f "$MODEL_DIR/model.safetensors" ]; }; then
    echo "📥 Downloading $MODEL_ID into $MODEL_DIR..."
    mkdir -p "$MODEL_DIR"
    apptainer exec --nv "$CONTAINER" .venv/bin/python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = '$MODEL_ID'
save_path = '$MODEL_DIR'

print(f'  -> Downloading tokenizer from {model_id}...')
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.save_pretrained(save_path)

print(f'  -> Downloading Causal LM weights from {model_id}...')
model = AutoModelForCausalLM.from_pretrained(model_id)
model.save_pretrained(save_path)

print('✓ LLaMA base model successfully saved to ' + save_path)
"
else
    echo "✓ Valid LLaMA base model found at $MODEL_DIR. Skipping download."
fi

# --- 3. Execute Pipeline ---
echo "🚀 Launching scripts_llama/order_1.sh training pipeline..."

apptainer exec --nv "$CONTAINER" bash -c "
    set -e
    source .venv/bin/activate
    bash scripts_llama/order_1.sh
" 2>&1 | tee logs_and_outputs_llama/order_1/logs/train_and_infer.log

echo "Finished executing order_1_llama at $(date)"
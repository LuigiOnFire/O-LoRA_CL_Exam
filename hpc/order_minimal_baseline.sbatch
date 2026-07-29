#!/bin/bash
#SBATCH --job-name=order_min_t5
#SBATCH --nodes=1
#SBATCH --partition=l40s
#SBATCH --gpus=1              
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00                 # 2 hours is plenty for a 2-task run
#SBATCH --output=logs/%j_order_minimal_baseline.out

set -euo pipefail

echo "Starting minimal O-LoRA execution on $HOSTNAME at $(date)"

# --- Storage & Environment Setup ---
USER_NAME="${SLURM_JOB_USER:-${USER:-${LOGNAME:-$(whoami)}}}"

# Persistent cache on scratch.hpc (project directory)
PERSISTENT_CACHE="$PWD/.scratch_cache"
mkdir -p logs logs_and_outputs/order_minimal/logs "$PERSISTENT_CACHE/huggingface" "$PERSISTENT_CACHE/torch"

# Node-local storage for sockets and lock files (avoids network filesystem locks)
NODE_TMP="${SLURM_TMPDIR:-/tmp/${USER_NAME}_${SLURM_JOB_ID:-standalone}}"
mkdir -p "$NODE_TMP"

# Automatically clean up node-local temp space on exit
trap 'rm -rf "$NODE_TMP"' EXIT

# CPU Thread Control (Prevents host thread oversubscription)
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

# Dynamic DeepSpeed Port to prevent collisions on multi-tenant nodes
export MASTER_PORT=$((29500 + ${SLURM_JOB_ID:-0} % 10000))

# Export HuggingFace and PyTorch caches
export HF_HOME="$PERSISTENT_CACHE/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TORCH_HOME="$PERSISTENT_CACHE/torch"
export TMPDIR="$NODE_TMP"

# Pass environment variables directly into Apptainer container
export APPTAINERENV_HF_HOME="$HF_HOME"
export APPTAINERENV_TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE"
export APPTAINERENV_HF_DATASETS_CACHE="$HF_DATASETS_CACHE"
export APPTAINERENV_TORCH_HOME="$TORCH_HOME"
export APPTAINERENV_TMPDIR="$TMPDIR"
export APPTAINERENV_OMP_NUM_THREADS="$OMP_NUM_THREADS"
export APPTAINERENV_MASTER_PORT="$MASTER_PORT"
export APPTAINERENV_CUDA_VISIBLE_DEVICES="0"

CONTAINER="./pytorch_2.4.0-cuda12.1-cudnn9-devel.sif"

# --- Check-and-Download Base Model ---
MODEL_DIR="initial_model/t5-large"
if [ ! -f "$MODEL_DIR/config.json" ] || { [ ! -f "$MODEL_DIR/model.safetensors" ] && [ ! -f "$MODEL_DIR/pytorch_model.bin" ]; }; then
    echo "📥 Base model missing. Downloading google-t5/t5-large into $MODEL_DIR..."
    mkdir -p "$MODEL_DIR"
    apptainer exec --nv "$CONTAINER" .venv/bin/python -c "
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
model_id = 'google-t5/t5-large'
save_path = '$MODEL_DIR'

print('  -> Downloading tokenizer...')
AutoTokenizer.from_pretrained(model_id).save_pretrained(save_path)

print('  -> Downloading model weights...')
AutoModelForSeq2SeqLM.from_pretrained(model_id).save_pretrained(save_path)

print('✓ Base model successfully saved.')
"
else
    echo "✓ Base model found at $MODEL_DIR. Skipping download."
fi


# ---  Execute Pipeline ---
echo "🚀 Launching scripts/order_minimal.sh training pipeline..."
chmod +x scripts/order_minimal.sh

apptainer exec --nv "$CONTAINER" bash -c "
    set -e
    source .venv/bin/activate
    bash scripts/order_minimal_baseline.sh
" 2>&1 | tee logs_and_outputs/order_minimal/logs/train_and_infer.log

echo "Finished executing order_minimal at $(date)"
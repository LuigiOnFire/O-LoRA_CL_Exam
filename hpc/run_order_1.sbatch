#!/bin/bash
#SBATCH --job-name=order_1_run
#SBATCH --nodes=1
#SBATCH --partition=l40s
#SBATCH --gpus=1              
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00         
#SBATCH --output=logs/%j_slurm_order_1.out
#SBATCH --error=logs/%j_slurm_order_1.err

set -euo pipefail

echo "Starting order_1 execution on $HOSTNAME at $(date)"

# --- 1. Storage & Environment Setup ---
# Resolve user name safely under set -u
USER_NAME="${SLURM_JOB_USER:-${USER:-${LOGNAME:-$(whoami)}}}"

# Persistent cache on scratch.hpc (project directory)
PERSISTENT_CACHE="$PWD/.scratch_cache"
mkdir -p logs logs_and_outputs/order_1/logs "$PERSISTENT_CACHE/huggingface" "$PERSISTENT_CACHE/torch"

# Node-local storage for sockets and lock files (avoids network lock hangs)
NODE_TMP="${SLURM_TMPDIR:-/tmp/${USER_NAME}_${SLURM_JOB_ID:-standalone}}"
mkdir -p "$NODE_TMP"

# Automatically clean up node-local temp space when the job exits
trap 'rm -rf "$NODE_TMP"' EXIT

# CPU Thread Control (Prevents thread oversubscription on shared nodes)
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

# Dynamic DeepSpeed Port to prevent port collisions on multi-tenant nodes
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

# --- 2. Check-and-Download Base Model ---
MODEL_DIR="initial_model/t5-large"
# Check for both config AND model weights to ensure non-corrupt download
if [ ! -f "$MODEL_DIR/config.json" ] || [ ! -f "$MODEL_DIR/model.safetensors" -a ! -f "$MODEL_DIR/pytorch_model.bin" ]; then
    echo "📥 Downloading google-t5/t5-large into $MODEL_DIR..."
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
    echo "✓ Valid base model found at $MODEL_DIR. Skipping download."
fi

# --- 3. Execute Pipeline ---
echo "🚀 Launching order_1 training pipeline..."

# Run execution while mirroring logs to both Slurm output and local file
apptainer exec --nv "$CONTAINER" bash -c "
    set -e
    source .venv/bin/activate
    bash scripts/order_1.sh
" 2>&1 | tee logs_and_outputs/order_1/logs/train_and_infer.log

echo "Finished executing order_1 at $(date)"
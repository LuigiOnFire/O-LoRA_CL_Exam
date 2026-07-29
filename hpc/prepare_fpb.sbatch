#!/bin/bash
#SBATCH --job-name=prep_fpb
#SBATCH --nodes=1
#SBATCH --partition=l40s
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:15:00          # 15 minutes is plenty for dataset processing
#SBATCH --output=logs/%j_prepare_fpb.out

set -euo pipefail

echo "Starting Financial PhraseBank data prep on $HOSTNAME at $(date)"

# --- 1. Storage & Environment Setup ---
USER_NAME="${SLURM_JOB_USER:-${USER:-${LOGNAME:-$(whoami)}}}"

PERSISTENT_CACHE="$PWD/.scratch_cache"
mkdir -p logs data/financial_phrasebank "$PERSISTENT_CACHE/huggingface" "$PERSISTENT_CACHE/torch"

NODE_TMP="${SLURM_TMPDIR:-/tmp/${USER_NAME}_${SLURM_JOB_ID:-standalone}}"
mkdir -p "$NODE_TMP"
trap 'rm -rf "$NODE_TMP"' EXIT

# Export HuggingFace and PyTorch caches
export HF_HOME="$PERSISTENT_CACHE/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TORCH_HOME="$PERSISTENT_CACHE/torch"
export TMPDIR="$NODE_TMP"

export APPTAINERENV_HF_HOME="$HF_HOME"
export APPTAINERENV_TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE"
export APPTAINERENV_HF_DATASETS_CACHE="$HF_DATASETS_CACHE"
export APPTAINERENV_TORCH_HOME="$TORCH_HOME"
export APPTAINERENV_TMPDIR="$TMPDIR"

CONTAINER="./pytorch_2.4.0-cuda12.1-cudnn9-devel.sif"

# --- 2. Run Data Prep Script ---
# Make sure the script path matches where your script is stored!
SCRIPT_PATH="download/prepare_fpb.py"

if [ ! -f "$SCRIPT_PATH" ]; then
    # Fallback to root directory if not in download/
    SCRIPT_PATH="prepare_fpb.py"
fi

echo "Running $SCRIPT_PATH inside container..."

apptainer exec "$CONTAINER" bash -c "
    set -e
    source .venv/bin/activate
    python3 $SCRIPT_PATH
"

echo "✓ Data preparation complete at $(date)"
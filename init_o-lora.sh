#!/bin/bash
echo "🚀 Initializing O-Lora Workspace..."

# Create the logs directory before Slurm tries to use it
mkdir -p logs

# 3. Create the sbatch script on the fly
cat << 'EOF' > setup_o_lora.sbatch
#!/bin/bash
#SBATCH --job-name=setup_o_lora
#SBATCH --nodes=1
#SBATCH --partition=l40s
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%j_setup.out

echo "Starting O-Lora environment setup on $HOSTNAME"

export APPTAINER_CACHEDIR=/scratch.hpc/$USER/.apptainer_cache
export APPTAINER_TMPDIR=/scratch.hpc/$USER/.apptainer_tmp
mkdir -p $APPTAINER_CACHEDIR $APPTAINER_TMPDIR

# Protect home directory quota from pip cache bloat
export GLOBAL_CACHE=$PWD/.scratch_cache
mkdir -p $GLOBAL_CACHE
export APPTAINERENV_PIP_CACHE_DIR=$GLOBAL_CACHE/pip

echo "Pulling modern PyTorch Apptainer image..."
apptainer pull docker://pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel

echo "Creating .venv..."
apptainer run --nv ./pytorch_2.4.0-cuda12.1-cudnn9-devel.sif python -m venv .venv

apptainer run --nv ./pytorch_2.4.0-cuda12.1-cudnn9-devel.sif bash -c "
    set -e
    source .venv/bin/activate
    
    echo '1/5 Upgrading core build tools...'
    pip install --upgrade pip
    pip install ninja packaging 'numpy<2' 'pyarrow<15.0.0'

    echo '2/5 Installing extra evaluation & model dependencies...'
    pip install bitsandbytes matplotlib rouge-score rouge sentencepiece
    
    echo '3/5 Cleaning requirements.txt...'
    # Filter out packages we explicitly manage/pin so requirements.txt doesn't overwrite them
    grep -vEi '^(torch|sentencepiece|setuptools|numpy|pyarrow)(vision|audio)?(=|>|<|$)' requirements.txt > requirements_clean.txt
    
    echo '4/5 Installing remaining repo requirements & pinning setuptools...'
    pip install -r requirements_clean.txt
    
    echo '5/5 Installing DeepSpeed...'
    pip install deepspeed
    pip install --force-reinstall 'setuptools==69.5.1'

"

rm -rf $APPTAINER_CACHEDIR $APPTAINER_TMPDIR
echo "Setup complete! O-Lora is ready to run."
EOF

# 4. Submit to Slurm
echo "🐳 Submitting Apptainer and Python setup job to Slurm..."
sbatch setup_o_lora.sbatch

echo "🎉 Setup triggered! Use 'squeue -u $USER' to monitor progress."
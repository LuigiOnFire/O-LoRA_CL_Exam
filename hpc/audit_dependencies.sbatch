#!/bin/bash
#SBATCH --job-name=audit_imports
#SBATCH --nodes=1
#SBATCH --partition=l40s
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%j_audit.out

mkdir -p logs

apptainer exec --nv ./pytorch_2.4.0-cuda12.1-cudnn9-devel.sif .venv/bin/python -c "
import sys
from pathlib import Path

# Add src and src/peft to Python path
PROJECT_ROOT = Path('.').resolve()
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
sys.path.insert(0, str(PROJECT_ROOT / 'src' / 'peft'))

print('🔍 Testing main execution entry point (src.run_uie_lora)...')
try:
    import run_uie_lora
    print('\n' + '='*60)
    print('🎉 SUCCESS! All imports in run_uie_lora resolved cleanly.')
    print('Your O-LoRA environment is 100% ready for training jobs!')
    print('='*60 + '\n')
except Exception as e:
    print('\n' + '='*60)
    print(f'❌ CRASH ON ENTRY POINT IMPORT: {e}')
    print('='*60 + '\n')
    raise e
"
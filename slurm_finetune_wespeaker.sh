#!/bin/bash
#SBATCH -o wespeaker-finetune-%j.output
#SBATCH --gres=gpu:1               # Embedding training is fine with 1 GPU
#SBATCH -n 1                       # Number of tasks
#SBATCH -c 8                       # Number of CPUs per task

# 1. Environment Setup
if [[ -f "$HOME/.bashrc" ]]; then
    source "$HOME/.bashrc"
fi

if command -v conda >/dev/null 2>&1; then
    conda activate diarizen
fi

# 2. Path Setup
export PYTHONPATH=$PYTHONPATH:/home3/yihao/Research/Code/DiariZen

# 3. Execution
echo "Starting WeSpeaker fine-tuning on Slurm..."
echo "Job ID: $SLURM_JOB_ID"

python finetune_wespeaker.py

echo "Job finished at $(date)"

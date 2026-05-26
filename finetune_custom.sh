#!/bin/bash
#SBATCH -o diarizen-finetune-%j.output
#SBATCH --gres=gpu:2               # Number of GPUs
#SBATCH -n 1                       # Number of tasks
#SBATCH -c 8                       # Number of CPUs per task


# 1. Environment Setup (from your test.sh)
if [[ -f "$HOME/.bashrc" ]]; then
    source "$HOME/.bashrc"
fi

if command -v conda >/dev/null 2>&1; then
    conda activate diarizen
fi

# 2. Path Setup
export PYTHONPATH=$PYTHONPATH:/home3/yihao/Research/Code/DiariZen
CONFIG="conf/custom_finetune.toml"

# 3. Execution
echo "Starting DiariZen fine-tuning on Slurm..."
echo "Using config: $CONFIG"
echo "Job ID: $SLURM_JOB_ID"

# We use --num_processes 2 to match the --gres=gpu:2 above
accelerate launch \
    --num_processes 2 \
    --main_process_port 1134 \
    run_dual_opt.py -C $CONFIG -M train

echo "Job finished at $(date)"

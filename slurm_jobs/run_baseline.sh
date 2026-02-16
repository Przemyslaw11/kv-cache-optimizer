#!/bin/bash
#SBATCH --job-name=kvquant-baseline
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=04:00:00
#SBATCH --mem=128GB
#SBATCH --output=logs/baseline_%j.out
#SBATCH --error=logs/baseline_%j.err
#SBATCH -A plgoncotherapy-gpu-a100

# =============================================================================
# KVQuant Prefill — Baseline FP16 Prefill Measurement
# =============================================================================

echo "=== Job Info ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo "================"

# Environment setup
module purge
module load Python/3.10.4
module load CUDA/12.4.0
module list

# Ensure SCRATCH is defined (may not survive module purge on some clusters)
export SCRATCH="${SCRATCH:-/net/tscratch/people/$(whoami)}"
cd $SCRATCH/kv-cache-optimizer
source .venv/bin/activate

# Offline WandB
export WANDB_MODE=offline

# Verify GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}')"

# Run baseline measurement
python scripts/measure_prefill_baseline.py

echo "=== Job Complete: $(date) ==="

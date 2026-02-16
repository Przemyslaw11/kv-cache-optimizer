#!/bin/bash
#SBATCH --job-name=kvquant-throughput
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:8
#SBATCH --time=24:00:00
#SBATCH --mem=1024GB
#SBATCH --output=logs/throughput_%j.out
#SBATCH --error=logs/throughput_%j.err
#SBATCH -A <your_grant_name>-gpu-a100

# =============================================================================
# KVQuant Prefill — Throughput Benchmark Sweep
# =============================================================================

echo "=== Job Info ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start: $(date)"
echo "================"

# Environment setup
module purge
module load Python/3.10.4
module load CUDA/12.4.0
module list

cd $SCRATCH/kv-cache-optimizer
source .venv/bin/activate

export WANDB_MODE=offline

# Verify all GPUs
python -c "
import torch
n = torch.cuda.device_count()
print(f'GPUs available: {n}')
for i in range(n):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
"

# Run throughput benchmark
python experiments/benchmark_throughput.py

echo "=== Job Complete: $(date) ==="

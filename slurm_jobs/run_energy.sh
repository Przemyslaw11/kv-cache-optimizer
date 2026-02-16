#!/bin/bash
#SBATCH --job-name=kvquant-energy
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:8
#SBATCH --time=12:00:00
#SBATCH --mem=1024GB
#SBATCH --output=logs/energy_%j.out
#SBATCH --error=logs/energy_%j.err
#SBATCH -A plgoncotherapy-gpu-a100

# =============================================================================
# KVQuant Prefill — Energy Profiling (NVML + CodeCarbon)
# =============================================================================

echo "=== Job Info ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo "================"

module purge
module load Python/3.10.4
module load CUDA/12.4.0
module list

# Ensure SCRATCH is defined (may not survive module purge on some clusters)
export SCRATCH="${SCRATCH:-/net/tscratch/people/$(whoami)}"
cd $SCRATCH/kv-cache-optimizer
source .venv/bin/activate

export WANDB_MODE=offline

python experiments/profile_energy.py

echo "=== Job Complete: $(date) ==="

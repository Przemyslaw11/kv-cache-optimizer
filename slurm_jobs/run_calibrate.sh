#!/bin/bash
#SBATCH --job-name=calibrate
#SBATCH --partition=plgrid-gpu-a100
#SBATCH -A <your_grant_name>-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128GB
#SBATCH --time=04:00:00
#SBATCH --output=logs/calibrate_%j.out
#SBATCH --error=logs/calibrate_%j.err

# ============================================================
# Calibration job: compute NUQ codebooks + scaling factors
# from Wikitext-2 key activations on Llama-2-7B-32K.
#
# Expected runtime: ~30 min on 1x A100-40GB
# ============================================================

set -euo pipefail

echo "===== Job Info ====="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "GPUs: ${CUDA_VISIBLE_DEVICES:-none}"
echo "Start: $(date)"
echo "===================="

# Load modules
module load CUDA/12.4.0
module load Python/3.10.4

# Activate environment
cd "${SLURM_SUBMIT_DIR}"
source .venv/bin/activate

# Ensure directories exist
mkdir -p logs results

# Run calibration
python experiments/calibrate.py \
    --model_path models/Llama-2-7B-32K \
    --output_dir results/calibration/Llama-2-7B-32K \
    --nuq_datatype nuq3 \
    --num_samples 16 \
    --max_seq_len 2048 \
    --kmeans_iterations 100 \
    --seed 42

echo "===== Job Complete ====="
echo "End: $(date)"

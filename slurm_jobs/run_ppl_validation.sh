#!/bin/bash
#SBATCH --job-name=ppl-validation
#SBATCH --partition=plgrid-gpu-a100
#SBATCH -A <your_grant_name>-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128GB
#SBATCH --time=04:00:00
#SBATCH --output=logs/ppl_validation_%j.out
#SBATCH --error=logs/ppl_validation_%j.err

# ============================================================
# Perplexity validation: compare fp16 vs quantized KV cache
# on Wikitext-2 test set. Requires calibration artifacts from
# the calibrate job (run_calibrate.sh).
#
# Expected runtime: ~1h on 1x A100-40GB
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

# Configuration
MODEL="models/Llama-2-7B-32K"
CALIB_DIR="results/calibration/Llama-2-7B-32K"
NUQ="nuq3"
OUTLIER=0.01
BLOCK_SIZE=256
MAX_SEQ_LEN=2048
STRIDE=512

# Run with calibrated codebooks (if available)
if [ -d "${CALIB_DIR}" ]; then
    echo "Using calibrated codebooks from ${CALIB_DIR}"
    python scripts/run_ppl_validation.py \
        --model_path "${MODEL}" \
        --calibration_dir "${CALIB_DIR}" \
        --nuq_datatype "${NUQ}" \
        --outlier_fraction "${OUTLIER}" \
        --block_size "${BLOCK_SIZE}" \
        --max_seq_len "${MAX_SEQ_LEN}" \
        --stride "${STRIDE}" \
        --seed 42
else
    echo "No calibration dir found, using heuristic codebook"
    python scripts/run_ppl_validation.py \
        --model_path "${MODEL}" \
        --nuq_datatype "${NUQ}" \
        --outlier_fraction "${OUTLIER}" \
        --block_size "${BLOCK_SIZE}" \
        --max_seq_len "${MAX_SEQ_LEN}" \
        --stride "${STRIDE}" \
        --seed 42
fi

echo "===== Job Complete ====="
echo "End: $(date)"

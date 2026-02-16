#!/bin/bash
#SBATCH --job-name=kvquant-ppl-val
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128000M
#SBATCH --time=08:00:00
#SBATCH -A plgoncotherapy-gpu-a100
#SBATCH -p plgrid-gpu-a100
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

# =============================================================================
# KVQuant Prefill — Perplexity Validation
# Compare fp16 vs quantized KV cache PPL on Wikitext-2 test set.
# Requires calibration artifacts from run_calibrate.sh (optional — falls
# back to heuristic codebook if not available).
#
# Expected runtime: ~1h on 1x A100-40GB
# =============================================================================

set -euo pipefail

echo "=== Job Info ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: ${CUDA_VISIBLE_DEVICES:-none}"
echo "Start: $(date)"
echo "================"

# Environment setup
module purge
module load Python/3.10.4
module load CUDA/12.4.0
module list

# Redirect caches to scratch
export SCRATCH="/net/tscratch/people/$(whoami)"
export HF_HOME="$SCRATCH/.cache/huggingface"
export WANDB_MODE=offline

cd $SCRATCH/kv-cache-optimizer
source .venv/bin/activate

# Ensure directories exist
mkdir -p logs results

# Verify GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}')"

# Configuration
MODEL="models/Llama-2-7B-32K"
CALIB_DIR="results/calibration/Llama-2-7B-32K"
NUQ="nuq3"
OUTLIER=0.01
BLOCK_SIZE=256
MAX_SEQ_LEN=2048
STRIDE=512

# Check model exists
if [ ! -d "$MODEL" ] || [ -z "$(ls -A $MODEL 2>/dev/null)" ]; then
    echo "ERROR: Model not found at $MODEL. Run run_calibrate.sh first to download."
    exit 1
fi

# Run with calibrated codebooks (if available)
if [ -d "${CALIB_DIR}" ] && [ -n "$(ls -A ${CALIB_DIR} 2>/dev/null)" ]; then
    echo ">>> Using calibrated codebooks from ${CALIB_DIR}"
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
    echo ">>> No calibration dir found, using heuristic codebook"
    python scripts/run_ppl_validation.py \
        --model_path "${MODEL}" \
        --nuq_datatype "${NUQ}" \
        --outlier_fraction "${OUTLIER}" \
        --block_size "${BLOCK_SIZE}" \
        --max_seq_len "${MAX_SEQ_LEN}" \
        --stride "${STRIDE}" \
        --seed 42
fi

echo "=== Job Complete: $(date) ==="

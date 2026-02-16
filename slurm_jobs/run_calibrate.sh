#!/bin/bash
#SBATCH --job-name=kvquant-calibrate
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
# KVQuant Prefill — Calibration Pipeline
# Compute NUQ codebooks + scaling factors from Wikitext-2 key activations
# on Llama-2-7B-32K-Instruct.
#
# Expected runtime: ~30 min on 1x A100-40GB
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
mkdir -p logs results models

# Verify GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}')"

# Download model if not present
MODEL_DIR="models/Llama-2-7B-32K"
if [ ! -d "$MODEL_DIR" ] || [ -z "$(ls -A $MODEL_DIR 2>/dev/null)" ]; then
    echo ">>> Downloading model togethercomputer/Llama-2-7B-32K-Instruct..."
    python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'togethercomputer/Llama-2-7B-32K-Instruct',
    local_dir='$MODEL_DIR',
    local_dir_use_symlinks=False,
)
print('Download complete.')
"
else
    echo ">>> Model already present at $MODEL_DIR"
fi

# Run calibration
python experiments/calibrate.py \
    --model_path "$MODEL_DIR" \
    --output_dir results/calibration/Llama-2-7B-32K \
    --nuq_datatype nuq3 \
    --num_samples 16 \
    --max_seq_len 2048 \
    --kmeans_iterations 100 \
    --seed 42

echo "=== Job Complete: $(date) ==="

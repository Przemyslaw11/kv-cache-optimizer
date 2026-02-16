#!/bin/bash
# =============================================================================
# Athena Quick Setup Script
# Run this ONCE on the Athena login node after cloning the repository.
# =============================================================================

set -e

echo "=== KV Cache Optimizer — Athena Setup ==="
echo "Date: $(date)"
echo "User: $(whoami)"
echo "Host: $(hostname)"
echo ""

# 0. Redirect caches to scratch (home quota is 10GB)
export SCRATCH="/net/tscratch/people/$(whoami)"
export UV_CACHE_DIR="$SCRATCH/.cache/uv"
export PIP_CACHE_DIR="$SCRATCH/.cache/pip"
export HF_HOME="$SCRATCH/.cache/huggingface"
export WANDB_DIR="$SCRATCH/.cache/wandb"
export XDG_CACHE_HOME="$SCRATCH/.cache"
mkdir -p "$UV_CACHE_DIR" "$PIP_CACHE_DIR" "$HF_HOME" "$WANDB_DIR"
echo ">>> Caches redirected to $SCRATCH/.cache/"

# 1. Load modules
echo ">>> Loading modules..."
module purge
module load Python/3.10.4
module load CUDA/12.4.0
module list

# 2. Install uv (into user site-packages if not already available)
echo ""
echo ">>> Ensuring uv is available..."
pip install --quiet --upgrade pip
pip install --quiet uv

# 3. Install project dependencies (uv creates .venv automatically)
echo ""
echo ">>> Installing project dependencies with uv..."
uv sync --python "$(which python3)"

# 4. Create directories
echo ""
echo ">>> Creating project directories..."
mkdir -p results logs models

# 5. Verify installation
echo ""
echo ">>> Verifying installation..."
.venv/bin/python -c '
import sys, torch, transformers, scipy, kvquant, utils
print("Python:", sys.version)
print("PyTorch:", torch.__version__)
print("CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("Transformers:", transformers.__version__)
print("scipy:", scipy.__version__)
print("kvquant: OK")
print("utils: OK")

# Verify core classes can be imported
from kvquant import (
    BatchedKVQuantizer, PrefillQuantizedAttention,
    BlockedCSCMatrix, BlockedCSRMatrix,
    create_heuristic_codebook, quantize_to_nuq,
)
print("Core imports: OK")
print("All imports successful!")
'

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "  1. Activate the environment:  source .venv/bin/activate"
echo "  2. Edit slurm_jobs/*.sh to set your grant: #SBATCH -A <grant>-gpu-a100"
echo "  3. Download model: huggingface-cli download togethercomputer/Llama-2-7B-32K-Instruct --local-dir ./models/Llama-2-7B-32K"
echo "  4. Run baseline: sbatch slurm_jobs/run_baseline.sh"

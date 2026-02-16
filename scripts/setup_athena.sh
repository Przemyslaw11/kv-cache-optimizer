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

# 1. Load modules
echo ">>> Loading modules..."
module purge
module load Python/3.10.4
module load CUDA/12.4.0
module list

# 2. Create virtual environment
echo ""
echo ">>> Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# 3. Install uv
echo ""
echo ">>> Installing uv..."
pip install --upgrade pip
pip install uv

# 4. Install project dependencies
echo ""
echo ">>> Installing project dependencies with uv..."
uv sync

# 5. Create directories
echo ""
echo ">>> Creating project directories..."
mkdir -p results logs models

# 6. Verify installation
echo ""
echo ">>> Verifying installation..."
python -c "
import torch
import transformers
import pynvml
print(f'Python: {__import__(\"sys\").version}')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'Transformers: {transformers.__version__}')
print(f'pynvml: {pynvml.__version__}')
print('All imports successful!')
"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "  1. Edit slurm_jobs/*.sh to set your grant: #SBATCH -A <grant>-gpu-a100"
echo "  2. Download model: huggingface-cli download togethercomputer/Llama-2-7B-32K-Instruct --local-dir ./models/Llama-2-7B-32K"
echo "  3. Run baseline: sbatch slurm_jobs/run_baseline.sh"

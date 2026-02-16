# KV Cache Optimizer

**Efficient Prefill: Blocked Quantization for Batched LLM Inference**

Research codebase for ICML 2026 submission. Extends [KVQuant](https://github.com/SqueezeAILab/KVQuant) (NeurIPS 2024) to the **prefill phase** of LLM inference, implementing batched quantization, blocked memory allocation, and energy-validated analysis.

## Key Innovations

1. **Batched quantization** — per-channel NUQ quantization of variable-length prompts during prefill
2. **Blocked CSC/CSR allocation** — pre-allocated memory blocks with O(1) append
3. **Energy profiling** — hardware-validated (NVML + CodeCarbon) energy and carbon footprint analysis

## Repository Structure

```
kv-cache-optimizer/
├── kvquant/                    # Core library
│   ├── batched_quant.py        # BatchedKVQuantizer
│   ├── blocked_sparse.py       # BlockedCSCMatrix / BlockedCSRMatrix
│   ├── prefill_quant.py        # PrefillQuantizedAttention
│   ├── nuq.py                  # Non-uniform quantization datatypes
│   └── kernels/
│       ├── triton_kernels.py   # Triton CUDA kernels (optional)
│       └── pytorch_fused.py    # PyTorch loop-fused quantization
├── experiments/                # Evaluation & benchmarking scripts
├── utils/                      # Shared utilities
├── scripts/                    # Setup & helper scripts
├── slurm_jobs/                 # Slurm .sh templates for Athena
├── paper/                      # LaTeX source (ICML 2026 format)
├── results/                    # JSON experiment outputs
├── tests/                      # Unit & integration tests
└── logs/                       # Slurm job logs
```

## Setup

### Local Setup (for development/testing)

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Przemyslaw11/kv-cache-optimizer.git
    cd kv-cache-optimizer
    ```
2.  **Install uv (package manager):**
    ```bash
    pip install uv
    ```
3.  **Install all dependencies:**
    ```bash
    uv sync
    ```

### Athena Cluster Setup (Cyfronet)

Experiments run on the `plgrid-gpu-a100` partition of the Athena supercomputer (8× NVIDIA A100-40GB).

1.  **Log in to Athena:**
    ```bash
    ssh plgX@athena.cyfronet.pl
    ```

2.  **Navigate to `$SCRATCH` and clone the repository:**
    ```bash
    cd $SCRATCH
    git clone https://github.com/X/kv-cache-optimizer.git
    cd kv-cache-optimizer
    ```

3.  **Set up the Python environment:**
    ```bash
    module purge
    module load Python/3.10.4
    module load CUDA/12.4.0
    module list

    python3 -m venv venv
    source venv/bin/activate

    pip install uv
    uv sync
    ```

    **Key Runtime Environment Versions:**
    - Python: `3.10.4` (via `module load Python/3.10.4`)
    - CUDA Toolkit: `12.4.0` (via `module load CUDA/12.4.0`)
    - PyTorch: `2.4.0`
    - Transformers: `latest`
    - pynvml: `latest`
    - CodeCarbon: `latest`
    - wandb: `latest`

4.  **Download the model (from login node):**
    ```bash
    pip install huggingface-hub
    huggingface-cli login
    huggingface-cli download togethercomputer/Llama-2-7B-32K-Instruct \
        --local-dir ./models/Llama-2-7B-32K
    ```

5.  **Verify CUDA availability (in interactive GPU session):**
    ```bash
    srun --partition=plgrid-gpu-a100 --gres=gpu:1 --time=00:10:00 --pty /bin/bash -l
    module purge
    module load Python/3.10.4
    module load CUDA/12.4.0
    source venv/bin/activate
    python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}')"
    ```

## Running Experiments

All Slurm jobs use the `plgrid-gpu-a100` partition. **Edit Slurm scripts to set your grant account: `#SBATCH -A <your_grant_name>-gpu-a100`**.

### Submitting a Job

```bash
sbatch slurm_jobs/run_baseline.sh
```

### WandB (Offline Mode)

Slurm scripts run WandB in offline mode (`export WANDB_MODE=offline`). Sync after completion:
```bash
wandb sync --sync-all
```

## Development

### Linting & Formatting

```bash
uv run ruff check . --fix   # Lint
uv run ruff format .         # Format
```

### Testing

```bash
uv run pytest tests/ -v                     # Full suite
uv run pytest tests/ -v -m "not gpu"        # CPU-only tests
uv run pytest tests/test_batched_quant.py   # Single module
```

### Pre-commit Workflow

1. `uv run ruff check . --fix`
2. `uv run ruff format .`
3. `uv run pytest tests/ -v`
4. Commit only if all pass.

## Troubleshooting

- **Slow `pip install` on Athena:** Download wheels locally, transfer to `$SCRATCH`, install with `pip install --no-index --find-links=./wheels -r requirements.txt`.
- **`CUDA available: False`:** Normal on login node. Verify in an interactive GPU job.
- **Module/Environment Conflicts:** Always `module purge` before loading modules. Activate venv after loading modules.
- **NVML Errors:** Report persistent NVML errors on compute nodes to Cyfronet support.

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{X2026efficient,
  title={Efficient Prefill: Blocked Quantization for Batched LLM Inference},
  author={X},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2026}
}
```

## License

MIT

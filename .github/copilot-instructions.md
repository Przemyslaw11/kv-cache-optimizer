# Copilot Instructions — KV Cache Optimizer (Project C1)

## Project Context

This is a **research codebase** for an ICML 2026 submission: *"Efficient Prefill: Blocked Quantization for Batched LLM Inference"*. We extend [KVQuant](https://github.com/SqueezeAILab/KVQuant) (NeurIPS 2024) to the **prefill phase** of LLM inference, implementing batched quantization, blocked memory allocation, and energy-validated analysis. The target hardware is **8× NVIDIA A100-40GB** on the Athena supercomputer (Slurm-managed).

### Research Goal

Achieve **2–3× throughput improvement** for long-context prefill (16K+ tokens) with **<0.1 PPL degradation** on LongBench, while reducing energy per token by 40–50%. Three core innovations:

1. **Batched quantization** — per-channel NUQ quantization of variable-length prompts during prefill (not just generation)
2. **Blocked CSC/CSR allocation** — pre-allocated memory blocks with O(1) append, eliminating reallocation overhead
3. **Energy profiling** — hardware-validated (NVML + CodeCarbon) energy and carbon footprint analysis

## Repository Structure

```
kv-cache-optimizer/
├── kvquant/                    # Core library
│   ├── batched_quant.py        # BatchedKVQuantizer — variable-length batch quantization
│   ├── blocked_sparse.py       # BlockedCSCMatrix / BlockedCSRMatrix — zero-copy append
│   ├── prefill_quant.py        # PrefillQuantizedAttention — drop-in attention replacement
│   ├── nuq.py                  # Non-uniform quantization datatypes (from KVQuant)
│   └── kernels/
│       ├── triton_kernels.py   # Triton CUDA kernels
│       └── pytorch_fused.py    # PyTorch loop-fused quantization (preferred path)
├── experiments/                # Evaluation & benchmarking scripts
│   ├── evaluate_longbench.py   # LongBench (6 tasks) accuracy evaluation
│   ├── evaluate_ruler.py       # RULER (13 tasks) long-context evaluation
│   ├── evaluate_passkey.py     # Passkey retrieval across context lengths
│   ├── benchmark_throughput.py # Throughput sweep (prompt_len × batch_size)
│   ├── profile_energy.py       # NVML energy-per-token profiling
│   ├── ablation_*.py           # Ablation studies (block_size, sparsity, bitwidth)
│   └── pareto_analysis.py      # Pareto frontier: accuracy vs throughput vs energy
├── utils/                      # Shared utilities
│   ├── energy_monitor.py       # Lightweight NVML wrapper
│   └── energy_profiler.py      # DetailedEnergyProfiler with CSV logging
├── scripts/                    # Setup & helper scripts
├── slurm_jobs/                 # Slurm .sh templates for Athena
├── paper/                      # LaTeX source (ICML 2026 format)
├── results/                    # JSON experiment outputs
├── tests/                      # Unit & integration tests
└── logs/                       # Slurm job logs
```

## Code Conventions & Quality Standards

### Python Style

- **Python 3.10+**, type hints on all public function signatures.
- Follow **PEP 8** with 100-char line limit (to fit wide tensor shape comments).
- Use `torch.no_grad()` context manager around all inference code — never compute unnecessary gradients.
- Prefer **`torch.float16`** (not `bfloat16`) for consistency with KVQuant and A100 FP16 tensor cores.

### SOLID & DRY Principles

- **Single Responsibility:** Each module owns one concept — `batched_quant.py` handles quantization logic only, `blocked_sparse.py` handles memory layout only, `prefill_quant.py` composes them into the attention replacement.
- **Open/Closed:** New quantization schemes (e.g., `nuq2`, `nuq4`) must be addable without modifying `BatchedKVQuantizer` — use the `nuq_datatype` lookup table parameter, not hardcoded bit-widths.
- **Dependency Inversion:** Experiment scripts depend on abstractions (`load_quantized_model(...)`) not on concrete model-loading internals. Keep model loading centralized in `kvquant/` init or a `model_utils.py`.
- **DRY:** Common patterns like `get_sequence_lengths(attention_mask)`, throughput measurement, and energy profiling must live in `utils/` — never duplicate these across experiment scripts.

### Documentation

- Every class and public function must have a **Google-style docstring** with `Args:`, `Returns:`, and tensor shape annotations using the convention `[batch_size, num_heads, seq_len, head_dim]`.
- Complex algorithms (quantization, blocked append) must include inline comments explaining the **why**, not just the **what**.
- Each experiment script must start with a module-level docstring stating: what it measures, expected runtime, required GPU resources, and expected output files.

### Tensor Shape Convention

Always document tensor shapes in comments and docstrings using this format:

```python
# keys: [batch_size, num_heads, seq_len, head_dim]  — fp16
# quantized_keys: [batch_size, num_heads, seq_len, head_dim]  — uint8 (3-bit packed)
# attention_mask: [batch_size, seq_len]  — 1=real token, 0=padding
```

### Error Handling

- Use explicit shape assertions (`assert keys.shape == (B, H, S, D), f"Expected ... got {keys.shape}"`) at function boundaries.
- Catch `RuntimeError` for OOM conditions in benchmark scripts — log the failure and continue to the next configuration, never crash the full sweep.
- All experiment scripts must save intermediate results to JSON after each configuration completes (crash-resilient design).

## Key Implementation Patterns

### Blocked Sparse Matrix — Zero-Copy Append

The `BlockedCSCMatrix` pre-allocates memory in blocks of `block_size` tokens (default 256). Appending a token is O(1). Only call `to_standard_csc()` when you need to perform matrix operations (attention computation). This is the critical performance pattern — never bypass it with naive `torch.cat` in a loop.

### Variable-Length Batch Handling

All prefill code must correctly handle **padded batches** with variable-length sequences. Always:
1. Extract actual lengths via `seq_lengths = attention_mask.sum(dim=1)`
2. Quantize only real tokens (indices `0:seq_len`), skip padding
3. When padding back for batched operations, use zeros for quantized indices

### Quantization Pipeline Order

The correct quantization pipeline is:
1. **Normalize** keys by pre-computed per-channel scaling factors
2. **Detect outliers** (top 1% by absolute value per channel)
3. **Quantize non-outliers** via NUQ lookup table (`torch.argmin` on distances)
4. **Store outliers** in blocked sparse matrix (CSC for keys, CSR for values)
5. **Dequantize** = NUQ lookup + add sparse outliers + rescale

### Fused vs. Naive Path

Prefer the **PyTorch loop-fused** path (`pytorch_fused.py`) over custom Triton kernels unless profiling shows the fused path is the bottleneck. The fused path vectorizes quantization across all heads and tokens in a single pass, avoiding intermediate fp16 materialization.

## Testing Strategy

- **Unit tests** (`tests/`): Quantization round-trip error < 0.05, blocked sparse correctness (compare `to_standard_csc()` output against naive `torch.cat` reference).
- **Integration tests**: Single-sample perplexity degradation < 0.2 PPL on Wikitext-2.
- **Statistical validation**: All accuracy claims require **5-seed runs** with paired t-test (p > 0.05 = no significant degradation).
- Run tests before every commit. Use `pytest tests/ -v` locally.

## Experiment Execution

### Benchmark Scripts

All benchmark scripts in `experiments/` follow this pattern:
1. Accept `--config` CLI argument (e.g., `len16384_batch16`)
2. Run warmup (5–10 iterations), then measure (100 iterations with `torch.cuda.synchronize()`)
3. Record `latency_sec`, `throughput_tokens_per_sec`, `peak_memory_gb`
4. Save results to `results/<experiment>_<config>.json`
5. Support crash-recovery by checking for existing result files before re-running

### Slurm Jobs

Use the templates in `slurm_jobs/`. Key settings for Athena:
- Partition: `plgrid-gpu-a100`
- 8 GPUs: `--gres=gpu:a100:8`
- Memory: `--mem=1024GB`
- Always redirect stdout/stderr to `logs/`

### Reproducibility

- Set `torch.manual_seed(42)` and `torch.cuda.manual_seed_all(42)` at the start of every experiment.
- Log all hyperparameters to WandB (`project="kvquant-prefill"`).
- Pin dependency versions in `requirements.txt`.

## Key Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| PyTorch | 2.4.0 | Core framework |
| Triton | 2.1.0 | Custom CUDA kernels |
| transformers | latest | Model loading (Llama-2-7B-32K) |
| pynvml | latest | GPU energy measurement |
| codecarbon | latest | Carbon emissions tracking |
| wandb | latest | Experiment logging |
| scipy | latest | Statistical tests |
| matplotlib | latest | Publication-quality figures |

## What NOT to Do

- **Never** quantize padding tokens — they waste computation and corrupt per-channel statistics.
- **Never** use `torch.cat` in a loop to grow sparse matrices — use `BlockedCSCMatrix.append()`.
- **Never** move tensors between CPU and GPU in hot paths — all quantization runs on GPU.
- **Never** hardcode model dimensions — read `config.num_attention_heads`, `config.hidden_size`, etc.
- **Never** skip `torch.cuda.synchronize()` before timing measurements — you'll get incorrect latencies.
- **Never** overwrite result files — use unique filenames with config/seed in the name.

## Publication Targets

- **Primary:** ICML 2026 (deadline Jan 31, 2026) — 8 pages + unlimited appendix
- **Backup:** NeurIPS 2026, MLSys 2026, EMNLP 2026
- Figures: 300 DPI, colorblind-friendly palette, matplotlib with `plt.tight_layout()`

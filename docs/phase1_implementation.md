# Phase 1 Implementation Report — Setup & Baseline

**Status:** ✅ Complete  
**Date completed:** February 16, 2026  
**Phase duration:** Weeks 1–2 (per execution plan)  
**Branch:** `main`

---

## 1. Summary

Phase 1 established the full project infrastructure, implemented the core quantization
library (`kvquant/`), created energy monitoring utilities (`utils/`), baseline measurement
scripts, Slurm job templates, and a comprehensive test suite. All 56 unit tests pass,
linting is clean (ruff), and the lockfile is committed for reproducibility.

---

## 2. Deliverables Checklist

| Deliverable | File(s) | Status |
|---|---|---|
| Project config & deps | `pyproject.toml`, `uv.lock` | ✅ |
| Athena setup script | `scripts/setup_athena.sh` | ✅ |
| NUQ codebook (heuristic + K-means) | `kvquant/nuq.py` | ✅ |
| Fused PyTorch quantization kernels | `kvquant/kernels/pytorch_fused.py` | ✅ |
| BatchedKVQuantizer | `kvquant/batched_quant.py` | ✅ |
| BlockedCSCMatrix / BlockedCSRMatrix | `kvquant/blocked_sparse.py` | ✅ |
| PrefillQuantizedAttention | `kvquant/prefill_quant.py` | ✅ |
| Energy monitoring utilities | `utils/energy_monitor.py`, `utils/energy_profiler.py` | ✅ |
| Baseline measurement script | `scripts/measure_prefill_baseline.py` | ✅ |
| Slurm job templates (4) | `slurm_jobs/run_{baseline,throughput,energy,longbench}.sh` | ✅ |
| Experiment script stubs (6) | `experiments/*.py` | ✅ |
| Unit tests (56 tests) | `tests/test_{batched_quant,blocked_sparse,prefill_quant}.py` | ✅ |
| Public API exports | `kvquant/__init__.py`, `utils/__init__.py` | ✅ |
| README with setup instructions | `README.md` | ✅ |

---

## 3. Architecture Overview

### 3.1 Module Dependency Graph

```
PrefillQuantizedAttention  (kvquant/prefill_quant.py)
    │
    ├── BatchedKVQuantizer  (kvquant/batched_quant.py)
    │       │
    │       ├── fused_quantize / fused_quantize_batch  (kvquant/kernels/pytorch_fused.py)
    │       ├── fused_dequantize / fused_dequantize_batch
    │       ├── BlockedCSCMatrix  (kvquant/blocked_sparse.py)  ← Key outliers
    │       ├── BlockedCSRMatrix  (kvquant/blocked_sparse.py)  ← Value outliers
    │       └── get_cached_codebook  (kvquant/nuq.py)
    │
    └── create_heuristic_codebook  (kvquant/nuq.py)

Utils (independent):
    ├── EnergyMonitor       (utils/energy_monitor.py)   ← single-session NVML wrapper
    └── DetailedEnergyProfiler (utils/energy_profiler.py) ← CSV + JSON multi-config profiler
```

### 3.2 Quantization Pipeline (implemented in `pytorch_fused.py`)

```
Input tensor [H, S, D] (fp16)
    │
    ├─ Step 1: Normalize per-channel  ──→  tensor / scaling_factors[:, None, :]
    │
    ├─ Step 2: Detect outliers        ──→  top outlier_fraction by abs value per (H, D)
    │                                       ──→ outlier_mask [H, S, D] (bool)
    │                                       ──→ outlier_values [H, S, D] (fp16, original scale)
    │
    ├─ Step 3: Quantize non-outliers  ──→  nearest NUQ centroid via argmin on distances
    │                                       ──→ quantized_indices [H, S, D] (uint8)
    │
    └─ Output: (quantized_indices, outlier_mask, outlier_values, scaling_factors)
```

Dequantization reverses this: codebook lookup → rescale → overwrite outlier positions.

### 3.3 Blocked Sparse Storage

- **BlockedCSCMatrix** — for key outliers. Each column = one token along `seq_len`.
- **BlockedCSRMatrix** — for value outliers. Each row = one token along `seq_len`.
- Memory is pre-allocated in blocks of `block_size` tokens (default 256, configurable).
- `append_token()` is O(1). New blocks are allocated transparently when full.
- `to_dense()` and `to_standard_csc()`/`to_standard_csr()` are O(n) — only called
  when attention computation requires the data.

### 3.4 Tensor Shape Conventions

All tensors follow the convention `[batch_size, num_heads, seq_len, head_dim]`:

```python
# keys / values:           [B, H, S, D]  — fp16
# quantized_indices:       [B, H, S, D]  — uint8
# outlier_mask:            [B, H, S, D]  — bool
# outlier_values:          [B, H, S, D]  — fp16
# scaling_factors:         [H, D]        — fp16
# codebook:                [num_levels]   — fp16
# attention_mask:          [B, S]         — int64 (1=real, 0=padding)
```

---

## 4. Public API Reference

### 4.1 `BatchedKVQuantizer`

```python
from kvquant import BatchedKVQuantizer, create_heuristic_codebook

codebook = create_heuristic_codebook("nuq3", device="cuda")
scales = torch.ones(32, 128, dtype=torch.float16, device="cuda")

quantizer = BatchedKVQuantizer(
    codebook=codebook,        # or just "nuq3" for heuristic
    scaling_factors=scales,   # [num_heads, head_dim], or None for unit scales
    outlier_fraction=0.01,    # top 1% as outliers
    block_size=256,           # for blocked sparse matrices
)

# Quantize a batched key tensor
cache = quantizer.quantize_keys(keys, attention_mask)   # → QuantizedKVCache
cache = quantizer.quantize_values(values, attention_mask)

# Dequantize back to fp16
reconstructed = quantizer.dequantize(cache)  # → [B, H, S, D] fp16

# Extract outliers into sparse storage
csc = quantizer.extract_key_outliers_to_sparse(cache, head_idx=0)   # → BlockedCSCMatrix
csr = quantizer.extract_value_outliers_to_sparse(cache, head_idx=0) # → BlockedCSRMatrix
```

### 4.2 `PrefillQuantizedAttention`

```python
from kvquant import PrefillQuantizedAttention

attn = PrefillQuantizedAttention(
    num_heads=32, head_dim=128, codebook="nuq3",
    scaling_factors=None, outlier_fraction=0.01, block_size=256,
)

# Prefill: quantize full KV cache in one batched pass
output = attn.forward(query, key, value, attention_mask, is_prefill=True)

# Generation: quantize single new token
output = attn.forward(query, key, value, attention_mask, is_prefill=False)

# Access cached quantized KV
key_cache = attn.key_cache     # QuantizedKVCache or None
value_cache = attn.value_cache

# Measure quantization error
metrics = attn.compute_round_trip_error(original_keys, attention_mask)
# → {"mse": float, "max_abs_error": float, "relative_error": float}
```

### 4.3 `BlockedCSCMatrix` / `BlockedCSRMatrix`

```python
from kvquant import BlockedCSCMatrix, BlockedCSRMatrix

csc = BlockedCSCMatrix(num_rows=128, block_size=256, device="cuda")
csc.append_token(row_indices, values)       # O(1) append
dense = csc.to_dense()                       # [num_rows, total_columns]
col_offsets, row_idx, vals = csc.to_standard_csc()

csr = BlockedCSRMatrix(num_cols=128, block_size=256, device="cuda")
csr.append_token(col_indices, values)
dense = csr.to_dense()                       # [total_rows, num_cols]
```

### 4.4 NUQ Utilities

```python
from kvquant import (
    create_heuristic_codebook,  # quick prototyping
    compute_nuq_codebook,       # K-means from calibration data
    quantize_to_nuq,            # values → uint8 indices
    dequantize_from_nuq,        # uint8 indices → fp16 values
    get_nuq_bitwidth,           # "nuq3" → 3
    get_num_levels,             # "nuq3" → 8
)
```

### 4.5 Energy Utilities

```python
from utils import EnergyMonitor, DetailedEnergyProfiler

# Simple single-session monitoring
monitor = EnergyMonitor(gpu_id=0)
monitor.start()
# ... inference ...
result = monitor.stop()  # → EnergyMeasurement dataclass

# Multi-config profiling with CSV + JSON output
profiler = DetailedEnergyProfiler(output_dir="results/energy", gpu_id=0)
with profiler.profile("nuq3_len4096_batch4") as ctx:
    # ... inference ...
    ctx.set_tokens(4096 * 4)
profiler.save_summary()  # → results/energy/energy_summary.json
```

---

## 5. Test Suite

### 5.1 Test Inventory (56 tests)

| File | Class | Tests | Coverage |
|---|---|---|---|
| `test_batched_quant.py` | `TestNUQCodebook` | 9 | Codebook creation, round-trip, dtype, range |
| | `TestFusedQuantize` | 5 | Shapes, round-trip MSE, outlier preservation, scales |
| | `TestBatchedKVQuantizer` | 9 | Shapes, MSE, variable-length, sparse extraction |
| | `TestHelpers` | 2 | `get_sequence_lengths` |
| `test_blocked_sparse.py` | `TestBlockedCSCMatrix` | 11 | Empty, append, boundaries, CSC format, dense vs cat |
| | `TestBlockedCSRMatrix` | 8 | Empty, append, boundaries, CSR format, dense vs cat |
| `test_prefill_quant.py` | `TestPrefillQuantizedAttention` | 9 | Forward shapes, NaN check, fp16 closeness, cache |
| | | **56 total** | |

### 5.2 Key Correctness Guarantees

- **Round-trip MSE < 0.1** for fused quantize → dequantize with 1% outliers.
- **Outlier preservation**: Injected extreme values (±50, ±100) are detected as
  outliers and exactly preserved after round-trip.
- **Padding correctness**: Padding positions have zero quantized indices and do not
  affect real-token quantization.
- **Sparse ↔ dense equivalence**: `BlockedCSCMatrix.to_dense()` exactly matches
  a naive `torch.cat` reference over 50 random tokens.
- **Block boundary handling**: Appending more tokens than `block_size` auto-allocates
  new blocks transparently.

### 5.3 Running Tests

```bash
# Full suite (CPU-only, no GPU needed)
.venv/bin/pytest tests/ -v

# Single module
.venv/bin/pytest tests/test_batched_quant.py -v

# GPU-only tests (requires CUDA)
.venv/bin/pytest tests/ -v -m gpu

# With coverage
.venv/bin/pytest tests/ --cov=kvquant --cov-report=term-missing
```

---

## 6. Infrastructure

### 6.1 Environment

- **Python:** 3.10.4 (via `module load Python/3.10.4`)
- **CUDA:** 12.4.0 (via `module load CUDA/12.4.0`)
- **PyTorch:** 2.4.0 (pinned in `pyproject.toml`)
- **Package manager:** uv (lockfile committed as `uv.lock`)
- **NVML:** `nvidia-ml-py>=12.560.30` (replaces deprecated `pynvml` package)
- **Linting:** ruff (lint + format in one tool)
- **Testing:** pytest with `@pytest.mark.gpu` markers

### 6.2 Slurm Templates

All templates in `slurm_jobs/` require editing the grant name:
```
#SBATCH -A <your_grant_name>-gpu-a100
```

| Script | GPUs | Time | Purpose |
|---|---|---|---|
| `run_baseline.sh` | 1× A100 | 4h | FP16 prefill baseline |
| `run_throughput.sh` | 8× A100 | 24h | Throughput sweep |
| `run_energy.sh` | 8× A100 | 12h | NVML energy profiling |
| `run_longbench.sh` | 8× A100 | 12h | LongBench evaluation |

### 6.3 Pre-commit Workflow

```bash
.venv/bin/ruff check . --fix    # 1. Lint (auto-fix safe issues)
.venv/bin/ruff format .          # 2. Format
.venv/bin/pytest tests/ -v       # 3. Test
git add -A && git commit         # 4. Commit only if all pass
```

---

## 7. Design Decisions & Rationale

### 7.1 PyTorch-first, Triton-optional

The fused quantization path (`pytorch_fused.py`) vectorizes across all heads and tokens
using standard PyTorch ops. No Python loops in the single-sample path. The batched path
loops only over the batch dimension (necessary for variable-length sequences). Triton
kernels (`triton_kernels.py`) remain an empty stub — they are a Phase 4 optimization
that should only be attempted after profiling confirms the PyTorch path is the bottleneck.

### 7.2 Outlier detection: per (head, dim) quantile along seq_len

The outlier threshold is computed as the `(1 - outlier_fraction)` quantile of absolute
normalized values **per channel** (i.e., per `(head, dim)` pair, across the `seq_len`
axis). This follows KVQuant's per-channel approach: different channels have different
magnitude distributions, so a global threshold would misclassify values.

### 7.3 BlockedCSRMatrix reuses `_Block` dataclass

Both `BlockedCSCMatrix` and `BlockedCSRMatrix` share the same `_Block` dataclass. In
`BlockedCSRMatrix`, the `row_indices` field stores column indices and `col_offsets`
stores per-row counts. This avoids code duplication at the cost of slightly confusing
field names internally — but the public API (`append_token(col_indices, values)`) uses
correct naming.

### 7.4 `QuantizedKVCache` as a dataclass (not a class with methods)

The quantized cache is a plain dataclass holding tensors. All logic lives in
`BatchedKVQuantizer` (quantize/dequantize/extract). This follows Dependency Inversion:
experiment scripts depend on `BatchedKVQuantizer`'s interface, not on cache internals.

---

## 8. Known Limitations (to address in Phase 2+)

1. **Batched quantization loops over batch dimension** — the `fused_quantize_batch`
   function iterates over `B` in Python. For large batch sizes, a fully vectorized
   path (or a Triton kernel) may be needed. Profile first.

2. **Generation path is a placeholder** — `_generation_quantize` in
   `PrefillQuantizedAttention` performs a full quantize-dequantize on the entire input
   rather than incrementally appending to the cached sparse matrices. This must be
   completed in Phase 2.

3. **No real model integration yet** — all tests use synthetic tensors. Phase 2 should
   integrate with `transformers` model internals (hook into Llama attention layers).

4. **Calibration-based codebooks untested on real data** — `compute_nuq_codebook` is
   validated on synthetic bimodal data. Real calibration from Wikitext-2 / Llama
   activations is a Phase 2 task.

5. **Experiment scripts are stubs** — `experiments/*.py` contain only docstrings.
   Implementation starts in Phase 5 (accuracy) and Phase 6 (throughput).

6. **No WandB integration yet** — logging setup is prepared (offline mode in Slurm),
   but actual `wandb.log()` calls are not in any script.

---

## 9. Phase 2 Readiness Checklist

Phase 2 goal: **Implement variable-length batch quantization with real model
integration** (Week 3 per execution plan).

Before starting Phase 2, ensure:

- [x] All 56 Phase 1 tests pass
- [x] Ruff lint + format clean
- [x] `uv.lock` committed
- [x] Grant name set in Slurm templates
- [x] Model downloaded to `./models/Llama-2-7B-32K`
- [ ] Baseline measurements collected via `sbatch slurm_jobs/run_baseline.sh`

### Phase 2 tasks to implement:

1. **Real model integration** — Hook `PrefillQuantizedAttention` into Llama-2-7B-32K
   attention layers (replace `LlamaAttention.forward` or use model hooks).
2. **Calibration pipeline** — Run `compute_nuq_codebook` and
   `estimate_scaling_factors` on real Wikitext-2 key activations.
3. **Incremental generation cache** — Extend `_generation_quantize` to append single
   tokens to existing `BlockedCSCMatrix`/`BlockedCSRMatrix` instances.
4. **Wikitext-2 perplexity validation** — Measure PPL degradation with quantized KV
   cache (target: < 0.2 PPL over fp16 baseline).
5. **Extended tests** — Add GPU-marked integration tests that run on real model
   outputs (synthetic → real activation distributions).

### Key files to modify in Phase 2:

| File | What to add |
|---|---|
| `kvquant/prefill_quant.py` | Incremental generation cache management |
| `kvquant/batched_quant.py` | (minor) optimize batch loop if profiling shows need |
| `kvquant/nuq.py` | Save/load calibrated codebooks for Llama-2-7B-32K |
| `experiments/` (new) | `calibrate.py` — run calibration on Wikitext-2 |
| `tests/test_prefill_quant.py` | GPU integration tests with real model |
| `scripts/` (new) | `run_ppl_validation.py` — perplexity check |

---

## 10. File Inventory

```
kv-cache-optimizer/
├── pyproject.toml                          # Project config, deps, ruff, pytest
├── uv.lock                                # Pinned dependency lockfile
├── README.md                              # Setup & usage instructions
├── LICENSE                                # MIT license
├── .gitignore                             # Comprehensive ignores
├── C1_KV_Cache_Prefill_Execution_Plan.md  # Full 8-12 week roadmap
│
├── kvquant/                               # Core library (5 modules)
│   ├── __init__.py                        # Public API exports
│   ├── nuq.py                            # NUQ codebook: create, compute, save, load, quant, dequant
│   ├── batched_quant.py                  # BatchedKVQuantizer, QuantizedKVCache, get_sequence_lengths
│   ├── blocked_sparse.py                 # BlockedCSCMatrix, BlockedCSRMatrix, _Block
│   ├── prefill_quant.py                  # PrefillQuantizedAttention
│   └── kernels/
│       ├── __init__.py                   # Re-exports pytorch_fused functions
│       ├── pytorch_fused.py              # Fused quant/dequant/estimate_scaling_factors
│       └── triton_kernels.py             # Stub (Phase 4)
│
├── utils/                                 # Shared utilities
│   ├── __init__.py                        # Exports EnergyMonitor, DetailedEnergyProfiler
│   ├── energy_monitor.py                 # NVML wrapper, EnergyMeasurement
│   └── energy_profiler.py                # Background-thread profiler, CSV + JSON output
│
├── tests/                                 # 56 unit tests
│   ├── __init__.py
│   ├── test_batched_quant.py             # 28 tests: NUQ, fused kernels, BatchedKVQuantizer
│   ├── test_blocked_sparse.py            # 19 tests: CSC + CSR correctness
│   └── test_prefill_quant.py             # 9 tests: PrefillQuantizedAttention integration
│
├── experiments/                           # Stubs (Phases 5–8)
│   ├── benchmark_throughput.py
│   ├── evaluate_longbench.py
│   ├── evaluate_passkey.py
│   ├── evaluate_ruler.py
│   ├── pareto_analysis.py
│   └── profile_energy.py
│
├── scripts/
│   ├── setup_athena.sh                   # One-time Athena env setup
│   └── measure_prefill_baseline.py       # FP16 baseline latency measurement
│
├── slurm_jobs/
│   ├── run_baseline.sh                   # 1 GPU, 4h
│   ├── run_throughput.sh                 # 8 GPUs, 24h
│   ├── run_energy.sh                     # 8 GPUs, 12h
│   └── run_longbench.sh                  # 8 GPUs, 12h
│
├── docs/
│   └── phase1_implementation.md          # This document
│
├── paper/                                 # LaTeX (ICML 2026)
├── results/                               # JSON outputs (gitignored)
├── models/                                # Model weights (gitignored)
└── logs/                                  # Slurm logs (gitignored)
```

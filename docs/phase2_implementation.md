# Phase 2 Implementation Report — Batched Quantization & Model Integration

**Status:** ✅ Complete  
**Date completed:** February 16, 2026  
**Phase duration:** Week 3 (per execution plan)  
**Branch:** `main`

---

## 1. Summary

Phase 2 implements the four core deliverables required to transition from synthetic-tensor
testing (Phase 1) to real model integration:

1. **Model integration layer** (`kvquant/model_utils.py`) — hook `PrefillQuantizedAttention`
   into any Llama-style HuggingFace model via `QuantizedModelWrapper.patch()`.
2. **Calibration pipeline** (`experiments/calibrate.py`) — compute per-layer NUQ codebooks
   and scaling factors from Wikitext-2 key activations via K-means.
3. **Incremental generation cache** — `_generation_quantize` in `PrefillQuantizedAttention`
   now appends single tokens to the existing quantized cache (prefill → generation
   transition is seamless).
4. **Perplexity validation script** (`scripts/run_ppl_validation.py`) — compare fp16 vs.
   quantized KV cache PPL on Wikitext-2 (target: < 0.2 PPL degradation).

Additionally:
- **24 new tests** (80 total, up from 56), including `_append_to_cache` correctness,
  multi-step generation, `QuantizedModelWrapper` patch/unpatch lifecycle, per-layer
  codebook save/load round-trip, `ActivationCollector` hook management, and GPU
  integration tests.
- **Per-layer NUQ persistence** (`nuq.py`) — save/load per-layer codebooks and scaling
  factors to/from disk for reproducibility.
- **Slurm job templates** for calibration and perplexity validation on Athena.

---

## 2. Deliverables Checklist

| Deliverable | File(s) | Status |
|---|---|---|
| Model integration module | `kvquant/model_utils.py` | ✅ |
| ActivationCollector (hook-based KV capture) | `kvquant/model_utils.py` | ✅ |
| QuantizedModelWrapper (patch/unpatch) | `kvquant/model_utils.py` | ✅ |
| Centralized model loading (`load_model`) | `kvquant/model_utils.py` | ✅ |
| Sliding-window perplexity eval | `kvquant/model_utils.py` | ✅ |
| Calibration pipeline script | `experiments/calibrate.py` | ✅ |
| Incremental generation cache | `kvquant/prefill_quant.py` | ✅ |
| `_append_to_cache` helper | `kvquant/prefill_quant.py` | ✅ |
| Per-layer codebook save/load | `kvquant/nuq.py` | ✅ |
| Per-layer scaling factor save/load | `kvquant/nuq.py` | ✅ |
| Perplexity validation script | `scripts/run_ppl_validation.py` | ✅ |
| Slurm: calibrate job | `slurm_jobs/run_calibrate.sh` | ✅ |
| Slurm: PPL validation job | `slurm_jobs/run_ppl_validation.sh` | ✅ |
| Generation cache tests (10 tests) | `tests/test_generation_cache.py` | ✅ |
| Model integration tests (14 tests) | `tests/test_model_integration.py` | ✅ |
| Updated public API exports | `kvquant/__init__.py` | ✅ |
| Phase 2 documentation | `docs/phase2_implementation.md` | ✅ |

---

## 3. Architecture Updates

### 3.1 New Module: `kvquant/model_utils.py`

```
QuantizedModelWrapper        ←— Top-level entry point for model patching
    │
    ├── ActivationCollector  ←— Hook-based KV activation capture
    │       │
    │       └── _make_hook() ←— Forward hook to intercept (K, V, past_kv)
    │
    ├── load_model()         ←— Centralized HF model + tokenizer loading
    │
    └── evaluate_perplexity() ←— Sliding-window PPL on Wikitext-2
```

### 3.2 Updated Module: `kvquant/prefill_quant.py`

```
PrefillQuantizedAttention
    │
    ├── _prefill_quantize()     ←— (unchanged) Full batch quant during prefill
    │
    ├── _generation_quantize()  ←— (UPDATED) Incremental single-token append
    │       │
    │       └── _append_to_cache()  ←— (NEW) Concatenate QuantizedKVCache along seq_len
    │
    └── forward()               ←— Dispatches to prefill or generation path
```

### 3.3 Updated Module: `kvquant/nuq.py`

New functions for per-layer artifact persistence:

| Function | Purpose |
|---|---|
| `save_per_layer_codebooks()` | Save `{layer_idx: codebook}` to `codebook_layer<N>.pt` |
| `load_per_layer_codebooks()` | Load all `codebook_layer*.pt` from a directory |
| `save_per_layer_scaling_factors()` | Save `{layer_idx: scales}` to `scales_layer<N>.pt` |
| `load_per_layer_scaling_factors()` | Load all `scales_layer*.pt` from a directory |

### 3.4 Incremental Generation Cache Design

During generation, new tokens are quantized individually and appended to the
existing cache via `_append_to_cache()`:

```python
def _append_to_cache(existing: QuantizedKVCache, new: QuantizedKVCache) -> QuantizedKVCache:
    """Concatenate along dim=2 (seq_len), update seq_lengths."""
    return QuantizedKVCache(
        quantized_indices=torch.cat([existing.quantized_indices, new.quantized_indices], dim=2),
        outlier_mask=torch.cat([existing.outlier_mask, new.outlier_mask], dim=2),
        outlier_values=torch.cat([existing.outlier_values, new.outlier_values], dim=2),
        scaling_factors=existing.scaling_factors,
        codebook=existing.codebook,
        seq_lengths=existing.seq_lengths + new.seq_lengths,
    )
```

The generation flow:
1. Quantize only the new token: `quantizer.quantize_keys(key[:,:,-1:,:], mask)`
2. Append to existing cache: `_append_to_cache(self._key_cache, new_cache)`
3. Dequantize the full cache for attention: `quantizer.dequantize(self._key_cache)`

This avoids re-quantizing previously quantized tokens.

---

## 4. New Public API Reference

### 4.1 `QuantizedModelWrapper`

```python
from kvquant import QuantizedModelWrapper, load_model

model, tokenizer = load_model("models/Llama-2-7B-32K")

# Option A: Heuristic codebook (no calibration needed)
wrapper = QuantizedModelWrapper(model, codebook="nuq3")

# Option B: Per-layer calibrated codebooks
from kvquant import load_per_layer_codebooks, load_per_layer_scaling_factors
codebooks = load_per_layer_codebooks("results/calibration/Llama-2-7B-32K")
scales = load_per_layer_scaling_factors("results/calibration/Llama-2-7B-32K")
wrapper = QuantizedModelWrapper(model, codebook=codebooks, scaling_factors=scales)

# Patch and use
wrapper.patch()
output = model(**inputs)  # Now uses quantized KV cache
wrapper.unpatch()         # Restore original model
```

### 4.2 `ActivationCollector`

```python
from kvquant import ActivationCollector

collector = ActivationCollector(model, max_samples=16)
collector.install_hooks()
with torch.no_grad():
    model(**calibration_inputs)
collector.increment_sample_count()
collector.remove_hooks()

keys_layer0 = collector.get_keys(layer_idx=0)  # [samples, H, S, D]
values_layer0 = collector.get_values(layer_idx=0)
```

### 4.3 `evaluate_perplexity`

```python
from kvquant import evaluate_perplexity

result = evaluate_perplexity(model, tokenizer, max_seq_len=2048, stride=512)
print(f"PPL: {result['perplexity']:.4f}")
```

### 4.4 Per-layer Codebook Persistence

```python
from kvquant import save_per_layer_codebooks, load_per_layer_codebooks

# Save
codebooks = {0: cb_layer0, 1: cb_layer1, ...}
save_per_layer_codebooks(codebooks, "results/calibration/Llama-2-7B-32K")

# Load
codebooks = load_per_layer_codebooks("results/calibration/Llama-2-7B-32K")
```

---

## 5. Experiment Scripts

### 5.1 Calibration (`experiments/calibrate.py`)

```bash
python experiments/calibrate.py \
    --model_path models/Llama-2-7B-32K \
    --output_dir results/calibration/Llama-2-7B-32K \
    --nuq_datatype nuq3 \
    --num_samples 16 \
    --max_seq_len 2048 \
    --kmeans_iterations 100 \
    --seed 42
```

Pipeline:
1. Load model + tokenizer
2. Load 16 Wikitext-2 calibration samples (2048 tokens each)
3. Install activation hooks on all attention layers
4. Run 16 forward passes to collect key activations
5. Compute per-layer scaling factors (mean absolute value per channel)
6. Compute per-layer NUQ codebooks (K-means on flattened key values)
7. Save codebooks, scales, and summary JSON to output directory

### 5.2 Perplexity Validation (`scripts/run_ppl_validation.py`)

```bash
python scripts/run_ppl_validation.py \
    --model_path models/Llama-2-7B-32K \
    --calibration_dir results/calibration/Llama-2-7B-32K \
    --nuq_datatype nuq3 \
    --outlier_fraction 0.01 \
    --max_seq_len 2048
```

Runs:
1. FP16 baseline perplexity on Wikitext-2 test set
2. Patched model with quantized KV cache
3. Quantized perplexity on same test set
4. Reports degradation and pass/fail (< 0.2 threshold)

---

## 6. Test Suite

### 6.1 Test Inventory (80 tests — 24 new)

| File | Class | Tests | New? |
|---|---|---|---|
| `test_batched_quant.py` | `TestNUQCodebook` | 9 | |
| | `TestFusedQuantize` | 5 | |
| | `TestBatchedKVQuantizer` | 9 | |
| | `TestHelpers` | 2 | |
| `test_blocked_sparse.py` | `TestBlockedCSCMatrix` | 11 | |
| | `TestBlockedCSRMatrix` | 8 | |
| `test_prefill_quant.py` | `TestPrefillQuantizedAttention` | 9 | |
| `test_generation_cache.py` | `TestAppendToCache` | 5 | ✅ |
| | `TestIncrementalGeneration` | 5 | ✅ |
| `test_model_integration.py` | `TestPerLayerCodebooks` | 4 | ✅ |
| | `TestPerLayerScalingFactors` | 3 | ✅ |
| | `TestQuantizedModelWrapper` | 4 | ✅ |
| | `TestActivationCollector` | 3 | ✅ |
| | `TestGPUIntegration` | 3 | ✅ (gpu) |
| | **Total** | **80** | **27** |

### 6.2 Key Correctness Guarantees (Phase 2 additions)

- **Cache append correctness**: `_append_to_cache` preserves existing data and
  correctly increments `seq_lengths`.
- **Multi-step generation**: 5-step generation after prefill produces correct
  shapes and no NaN/Inf at each step.
- **Prefill → generation transition**: Cache grows from `S_prefill` to
  `S_prefill + N` after N generation steps.
- **Codebook persistence round-trip**: Save → load per-layer codebooks matches
  original tensors exactly (`torch.allclose`).
- **Patch/unpatch lifecycle**: Patching replaces forward, unpatching restores
  original class method; forward produces valid output after unpatch.

---

## 7. Slurm Job Templates

| Script | GPUs | Time | Purpose |
|---|---|---|---|
| `run_calibrate.sh` | 1× A100 | 4h | NUQ calibration from Wikitext-2 |
| `run_ppl_validation.sh` | 1× A100 | 4h | FP16 vs quantized PPL comparison |

---

## 8. Known Limitations (to address in Phase 3+)

1. **`_generation_quantize` uses `torch.cat` per token** — the incremental cache
   appends via `torch.cat` along `dim=2`, which is O(n) per step. For very long
   generation (>1K tokens), this may become a bottleneck. Phase 3 should integrate
   with `BlockedCSCMatrix`/`BlockedCSRMatrix` for O(1) append.

2. **`QuantizedModelWrapper.patch` intercepts at output level** — the current hook
   quantizes the KV cache *after* attention is computed with fp16 values, then stores
   the quantized version. For true inference savings, the attention computation
   itself should use quantized KV. This requires a deeper forward replacement
   (Phase 4 Triton kernels).

3. **Calibration collects activations via `past_key_value` output** — this requires
   `use_cache=True`, which may not be available for all model architectures. The
   collector handles this gracefully by checking for the expected output format.

4. **No multi-GPU calibration** — the calibration script runs on a single GPU.
   For larger models, multi-GPU support (via `device_map="auto"`) is available
   for inference but the activation collection happens on whichever device the
   attention output lands on.

---

## 9. Phase 3 Readiness Checklist

Phase 3 goal: **Blocked memory allocation optimization** (Weeks 4–5).

Before starting Phase 3, ensure:

- [x] All 80 Phase 2 tests pass
- [x] Ruff lint + format clean
- [x] `uv.lock` committed
- [ ] Calibration run completed (`sbatch slurm_jobs/run_calibrate.sh`)
- [ ] PPL validation confirms < 0.2 degradation

### Phase 3 tasks to implement:

1. **Integrate blocked sparse matrices into generation cache** — replace
   `torch.cat` with `BlockedCSCMatrix.append_token()` / `BlockedCSRMatrix.append_token()`.
2. **Benchmark memory allocation overhead** — compare blocked vs. naive allocation
   for long sequences (16K+ tokens).
3. **Block size ablation** — sweep block_size ∈ {64, 128, 256, 512, 1024}.
4. **Memory savings validation** — measure peak GPU memory with blocked vs. naive.

### Key files to modify in Phase 3:

| File | What to add |
|---|---|
| `kvquant/prefill_quant.py` | Replace `_append_to_cache` with blocked sparse append |
| `kvquant/batched_quant.py` | Optimize batch loop if profiling shows need |
| `experiments/ablation_block_size.py` | Block size ablation script |
| `tests/test_blocked_sparse.py` | Extended tests for generation append |

---

## 10. File Inventory (Phase 2 additions)

```
kv-cache-optimizer/
├── kvquant/
│   ├── model_utils.py          # NEW: Model integration (load, patch, collect, PPL)
│   ├── prefill_quant.py        # MODIFIED: Incremental generation cache
│   ├── nuq.py                  # MODIFIED: Per-layer codebook/scale save/load
│   └── __init__.py             # MODIFIED: New public API exports
│
├── experiments/
│   └── calibrate.py            # NEW: NUQ calibration pipeline
│
├── scripts/
│   └── run_ppl_validation.py   # NEW: Perplexity validation script
│
├── slurm_jobs/
│   ├── run_calibrate.sh        # NEW: Slurm calibration job
│   └── run_ppl_validation.sh   # NEW: Slurm PPL validation job
│
├── tests/
│   ├── test_generation_cache.py   # NEW: 10 tests for incremental generation
│   └── test_model_integration.py  # NEW: 14 tests for model integration
│
└── docs/
    └── phase2_implementation.md   # NEW: This document
```

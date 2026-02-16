"""KVQuant Prefill — Blocked Quantization for Batched LLM Inference.

Public API
----------
Core classes:
    - ``BatchedKVQuantizer``  — variable-length batch quantization engine.
    - ``QuantizedKVCache``    — container for quantized indices + outliers.
    - ``PrefillQuantizedAttention`` — drop-in attention replacement.

Model integration (Phase 2):
    - ``QuantizedModelWrapper``  — patch HF models with quantized attention.
    - ``ActivationCollector``    — collect KV activations for calibration.
    - ``load_model``             — centralized model loading.
    - ``evaluate_perplexity``    — sliding-window perplexity evaluation.

Sparse storage:
    - ``BlockedCSCMatrix`` — for key outliers (column = token).
    - ``BlockedCSRMatrix`` — for value outliers (row = token).

NUQ utilities:
    - ``create_heuristic_codebook`` — quick prototyping codebook.
    - ``compute_nuq_codebook``      — calibration-based codebook.
    - ``quantize_to_nuq``           — nearest-centroid quantization.
    - ``dequantize_from_nuq``       — codebook index lookup.
    - ``save_per_layer_codebooks``  — save per-layer codebooks to disk.
    - ``load_per_layer_codebooks``  — load per-layer codebooks from disk.
    - ``save_per_layer_scaling_factors`` — save per-layer scales to disk.
    - ``load_per_layer_scaling_factors`` — load per-layer scales from disk.

Fused kernels (PyTorch path):
    - ``fused_quantize``         — single-sample fused quant.
    - ``fused_quantize_batch``   — batched fused quant.
    - ``fused_dequantize``       — single-sample dequant.
    - ``fused_dequantize_batch`` — batched dequant.
    - ``estimate_scaling_factors`` — calibration scale estimation.

Helpers:
    - ``get_sequence_lengths`` — extract real lengths from attention mask.
"""

from kvquant.batched_quant import (
    BatchedKVQuantizer,
    QuantizedKVCache,
    get_sequence_lengths,
)
from kvquant.blocked_sparse import BlockedCSCMatrix, BlockedCSRMatrix
from kvquant.kernels.pytorch_fused import (
    estimate_scaling_factors,
    fused_dequantize,
    fused_dequantize_batch,
    fused_quantize,
    fused_quantize_batch,
)
from kvquant.model_utils import (
    ActivationCollector,
    QuantizedModelWrapper,
    evaluate_perplexity,
    load_model,
)
from kvquant.nuq import (
    compute_nuq_codebook,
    create_heuristic_codebook,
    dequantize_from_nuq,
    get_num_levels,
    get_nuq_bitwidth,
    load_per_layer_codebooks,
    load_per_layer_scaling_factors,
    quantize_to_nuq,
    save_per_layer_codebooks,
    save_per_layer_scaling_factors,
)
from kvquant.prefill_quant import PrefillQuantizedAttention

__all__ = [
    "ActivationCollector",
    "BatchedKVQuantizer",
    "BlockedCSCMatrix",
    "BlockedCSRMatrix",
    "PrefillQuantizedAttention",
    "QuantizedKVCache",
    "QuantizedModelWrapper",
    "compute_nuq_codebook",
    "create_heuristic_codebook",
    "dequantize_from_nuq",
    "estimate_scaling_factors",
    "evaluate_perplexity",
    "fused_dequantize",
    "fused_dequantize_batch",
    "fused_quantize",
    "fused_quantize_batch",
    "get_num_levels",
    "get_nuq_bitwidth",
    "get_sequence_lengths",
    "load_model",
    "load_per_layer_codebooks",
    "load_per_layer_scaling_factors",
    "quantize_to_nuq",
    "save_per_layer_codebooks",
    "save_per_layer_scaling_factors",
]

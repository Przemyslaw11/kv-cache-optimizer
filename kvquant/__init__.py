"""KVQuant Prefill — Blocked Quantization for Batched LLM Inference.

Public API
----------
Core classes:
    - ``BatchedKVQuantizer``  — variable-length batch quantization engine.
    - ``QuantizedKVCache``    — container for quantized indices + outliers.
    - ``PrefillQuantizedAttention`` — drop-in attention replacement.

Sparse storage:
    - ``BlockedCSCMatrix`` — for key outliers (column = token).
    - ``BlockedCSRMatrix`` — for value outliers (row = token).

NUQ utilities:
    - ``create_heuristic_codebook`` — quick prototyping codebook.
    - ``compute_nuq_codebook``      — calibration-based codebook.
    - ``quantize_to_nuq``           — nearest-centroid quantization.
    - ``dequantize_from_nuq``       — codebook index lookup.

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
from kvquant.nuq import (
    compute_nuq_codebook,
    create_heuristic_codebook,
    dequantize_from_nuq,
    get_num_levels,
    get_nuq_bitwidth,
    quantize_to_nuq,
)
from kvquant.prefill_quant import PrefillQuantizedAttention

__all__ = [
    # Core
    "BatchedKVQuantizer",
    # Sparse storage
    "BlockedCSCMatrix",
    "BlockedCSRMatrix",
    "PrefillQuantizedAttention",
    "QuantizedKVCache",
    "compute_nuq_codebook",
    # NUQ
    "create_heuristic_codebook",
    "dequantize_from_nuq",
    "estimate_scaling_factors",
    "fused_dequantize",
    "fused_dequantize_batch",
    # Fused kernels
    "fused_quantize",
    "fused_quantize_batch",
    "get_num_levels",
    "get_nuq_bitwidth",
    # Helpers
    "get_sequence_lengths",
    "quantize_to_nuq",
]

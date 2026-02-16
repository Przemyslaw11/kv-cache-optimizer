"""KVQuant CUDA kernel implementations.

Default path: ``pytorch_fused`` (vectorized PyTorch ops).
Optional: ``triton_kernels`` (Triton CUDA kernels, Phase 4+).
"""

from kvquant.kernels.pytorch_fused import (
    estimate_scaling_factors,
    fused_dequantize,
    fused_dequantize_batch,
    fused_quantize,
    fused_quantize_batch,
)

__all__ = [
    "estimate_scaling_factors",
    "fused_dequantize",
    "fused_dequantize_batch",
    "fused_quantize",
    "fused_quantize_batch",
]

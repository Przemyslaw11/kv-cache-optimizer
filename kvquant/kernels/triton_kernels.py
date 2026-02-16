"""Triton CUDA kernels for quantized attention (optional optimization).

These kernels are an optional optimization layer. The PyTorch fused path
(pytorch_fused.py) is the default and must be correct before Triton is used.
"""

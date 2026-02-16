"""PyTorch loop-fused quantization kernels (preferred default path).

Vectorizes quantization across all heads and tokens in a single pass,
avoiding intermediate fp16 materialization. This is the primary implementation
path — Triton kernels are only used after this path is validated.
"""

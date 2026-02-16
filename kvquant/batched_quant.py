"""Batched KV cache quantization for the prefill phase.

This module implements variable-length batch quantization of Keys and Values
during the prefill phase, extending KVQuant's per-channel NUQ quantization
from single-token generation to multi-token prefill.
"""

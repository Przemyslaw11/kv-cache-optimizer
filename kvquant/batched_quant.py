"""Batched KV cache quantization for the prefill phase.

This module implements variable-length batch quantization of Keys and Values
during the prefill phase, extending KVQuant's per-channel NUQ quantization
from single-token generation to multi-token prefill.

Key design decisions:
    - Uses **pre-computed** scaling factors (from calibration) — no online
      recomputation during inference.
    - Uses the **PyTorch fused path** by default (``kernels/pytorch_fused.py``).
    - Correctly handles **padded batches** with variable-length sequences.
    - Stores outliers in ``BlockedCSCMatrix`` (keys) / ``BlockedCSRMatrix``
      (values) for O(1) append.
    - New quantization schemes (nuq2/3/4) are addable without modifying
      this class — controlled via the ``nuq_datatype`` codebook parameter.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

from kvquant.blocked_sparse import BlockedCSCMatrix, BlockedCSRMatrix
from kvquant.kernels.pytorch_fused import (
    fused_dequantize,
    fused_dequantize_batch,
    fused_quantize,
    fused_quantize_batch,
)
from kvquant.nuq import get_cached_codebook

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class QuantizedKVCache:
    """Container for a quantized KV cache for a single layer.

    Attributes:
        quantized_indices: Quantized index tensor
            ``[batch_size, num_heads, seq_len, head_dim]`` — uint8.
        outlier_mask: Boolean mask for outlier positions
            ``[batch_size, num_heads, seq_len, head_dim]``.
        outlier_values: Original fp16 values at outlier positions
            ``[batch_size, num_heads, seq_len, head_dim]``.
        scaling_factors: Per-channel scaling factors
            ``[num_heads, head_dim]`` — fp16.
        codebook: NUQ codebook ``[num_levels]`` — fp16.
        seq_lengths: Actual sequence length per batch element ``[batch_size]``.
    """

    quantized_indices: torch.Tensor
    outlier_mask: torch.Tensor
    outlier_values: torch.Tensor
    scaling_factors: torch.Tensor
    codebook: torch.Tensor
    seq_lengths: torch.Tensor


# ---------------------------------------------------------------------------
# BatchedKVQuantizer
# ---------------------------------------------------------------------------


class BatchedKVQuantizer:
    """Quantize KV cache during the prefill phase for batched variable-length prompts.

    This quantizer:
        1. Normalizes per-channel using pre-computed scaling factors.
        2. Detects outliers (configurable fraction, default 1 %).
        3. Quantizes non-outliers via nearest NUQ centroid lookup.
        4. Optionally stores outliers in blocked sparse matrices.

    The quantizer is stateless w.r.t. the cache — it produces quantized
    tensors that can be stored independently (e.g. in ``QuantizedKVCache``
    or in blocked sparse matrices managed by the caller).

    Args:
        codebook: Pre-computed NUQ codebook ``[num_levels]`` (fp16).
            Can also pass a string like ``"nuq3"`` for a heuristic codebook.
        scaling_factors: Per-channel scaling factors ``[num_heads, head_dim]``
            (fp16).  If ``None``, a unit scale is assumed (useful for tests).
        outlier_fraction: Fraction of values treated as outliers per channel.
            Default 0.01 (1 %).
        block_size: Block size for blocked sparse matrices.  Only used when
            ``store_outliers_sparse=True``.  Default 256.

    Example:
        >>> codebook = create_heuristic_codebook("nuq3", device="cuda")
        >>> scales = torch.ones(32, 128, dtype=torch.float16, device="cuda")
        >>> quantizer = BatchedKVQuantizer(codebook, scales)
        >>> result = quantizer.quantize_keys(keys, attention_mask)
    """

    def __init__(
        self,
        codebook: torch.Tensor | str,
        scaling_factors: torch.Tensor | None = None,
        outlier_fraction: float = 0.01,
        block_size: int = 256,
    ) -> None:
        # Handle string codebook shorthand
        if isinstance(codebook, str):
            self.codebook = get_cached_codebook(codebook)
        else:
            self.codebook = codebook

        self.scaling_factors = scaling_factors
        self.outlier_fraction = outlier_fraction
        self.block_size = block_size

        logger.info(
            "BatchedKVQuantizer: codebook=%d levels, outlier_fraction=%.3f, block_size=%d",
            self.codebook.numel(),
            outlier_fraction,
            block_size,
        )

    # ------------------------------------------------------------------
    # Single-sample quantization (unbatched, for one [H, S, D] tensor)
    # ------------------------------------------------------------------

    def quantize_single(
        self,
        tensor: torch.Tensor,
        scaling_factors: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize a single ``[num_heads, seq_len, head_dim]`` tensor.

        Args:
            tensor: Keys or values ``[H, S, D]`` — fp16.
            scaling_factors: Override per-channel scales ``[H, D]``.
                Falls back to ``self.scaling_factors``.

        Returns:
            Tuple of ``(quantized_indices, outlier_mask, outlier_values)``.
        """
        scales = self._resolve_scales(tensor, scaling_factors)
        codebook = self.codebook.to(device=tensor.device)

        q_idx, o_mask, o_vals, _ = fused_quantize(tensor, codebook, scales, self.outlier_fraction)
        return q_idx, o_mask, o_vals

    def dequantize_single(
        self,
        quantized_indices: torch.Tensor,
        outlier_mask: torch.Tensor,
        outlier_values: torch.Tensor,
        scaling_factors: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Dequantize a single ``[H, S, D]`` quantized tensor.

        Args:
            quantized_indices: ``[H, S, D]`` — uint8.
            outlier_mask: ``[H, S, D]`` — bool.
            outlier_values: ``[H, S, D]`` — fp16.
            scaling_factors: Override per-channel scales ``[H, D]``.

        Returns:
            Reconstructed tensor ``[H, S, D]`` — fp16.
        """
        H, _S, D = quantized_indices.shape
        if scaling_factors is None:
            scaling_factors = self.scaling_factors
        if scaling_factors is None:
            scaling_factors = torch.ones(H, D, dtype=torch.float16, device=quantized_indices.device)
        codebook = self.codebook.to(device=quantized_indices.device)
        scales = scaling_factors.to(device=quantized_indices.device, dtype=torch.float16)

        return fused_dequantize(quantized_indices, codebook, scales, outlier_mask, outlier_values)

    # ------------------------------------------------------------------
    # Batched quantization (for prefill: [B, H, S, D] with attention mask)
    # ------------------------------------------------------------------

    def quantize_keys(
        self,
        keys: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> QuantizedKVCache:
        """Quantize key tensors for a batched prefill.

        Only real tokens (determined by ``attention_mask``) are quantized;
        padding positions remain as zero indices.

        Args:
            keys: ``[batch_size, num_heads, seq_len, head_dim]`` — fp16.
            attention_mask: ``[batch_size, seq_len]`` — 1=real, 0=padding.

        Returns:
            ``QuantizedKVCache`` with quantized indices and outlier info.
        """
        return self._quantize_batch(keys, attention_mask)

    def quantize_values(
        self,
        values: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> QuantizedKVCache:
        """Quantize value tensors for a batched prefill.

        Args:
            values: ``[batch_size, num_heads, seq_len, head_dim]`` — fp16.
            attention_mask: ``[batch_size, seq_len]`` — 1=real, 0=padding.

        Returns:
            ``QuantizedKVCache`` with quantized indices and outlier info.
        """
        return self._quantize_batch(values, attention_mask)

    def dequantize(self, cache: QuantizedKVCache) -> torch.Tensor:
        """Dequantize a ``QuantizedKVCache`` back to fp16.

        Args:
            cache: Quantized cache returned by ``quantize_keys`` or
                ``quantize_values``.

        Returns:
            Dequantized tensor ``[B, H, S, D]`` — fp16.
        """
        codebook = cache.codebook.to(device=cache.quantized_indices.device)
        scales = cache.scaling_factors.to(
            device=cache.quantized_indices.device, dtype=torch.float16
        )

        # Rebuild attention mask from seq_lengths for the batch dequantize
        B, _H, S, _D = cache.quantized_indices.shape
        attention_mask = torch.zeros(B, S, dtype=torch.long, device=cache.quantized_indices.device)
        for b in range(B):
            sl = cache.seq_lengths[b].item()
            attention_mask[b, :sl] = 1

        return fused_dequantize_batch(
            cache.quantized_indices,
            codebook,
            scales,
            cache.outlier_mask,
            cache.outlier_values,
            attention_mask,
        )

    # ------------------------------------------------------------------
    # Blocked sparse outlier extraction
    # ------------------------------------------------------------------

    def extract_key_outliers_to_sparse(
        self,
        cache: QuantizedKVCache,
        head_idx: int,
    ) -> BlockedCSCMatrix:
        """Extract outliers for a single head into a BlockedCSCMatrix.

        Keys use CSC format: each column = one token position along seq_len,
        each row = one head_dim element.

        Args:
            cache: Quantized cache from ``quantize_keys``.
            head_idx: Which attention head to extract.

        Returns:
            ``BlockedCSCMatrix`` containing outlier values for this head.
        """
        B, _H, _S, D = cache.quantized_indices.shape
        device = cache.quantized_indices.device

        csc = BlockedCSCMatrix(
            num_rows=D,
            block_size=self.block_size,
            device=device,
            dtype=torch.float16,
        )

        for b in range(B):
            seq_len = cache.seq_lengths[b].item()
            for t in range(seq_len):
                mask = cache.outlier_mask[b, head_idx, t, :]  # [D]
                if mask.any():
                    row_indices = torch.where(mask)[0].to(dtype=torch.int32)
                    values = cache.outlier_values[b, head_idx, t, mask]
                    csc.append_token(row_indices, values)
                else:
                    # Append an empty token (no outliers)
                    csc.append_token(
                        torch.zeros(0, dtype=torch.int32, device=device),
                        torch.zeros(0, dtype=torch.float16, device=device),
                    )

        return csc

    def extract_value_outliers_to_sparse(
        self,
        cache: QuantizedKVCache,
        head_idx: int,
    ) -> BlockedCSRMatrix:
        """Extract outliers for a single head into a BlockedCSRMatrix.

        Values use CSR format: each row = one token position along seq_len,
        each column = one head_dim element.

        Args:
            cache: Quantized cache from ``quantize_values``.
            head_idx: Which attention head to extract.

        Returns:
            ``BlockedCSRMatrix`` containing outlier values for this head.
        """
        B, _H, _S, D = cache.quantized_indices.shape
        device = cache.quantized_indices.device

        csr = BlockedCSRMatrix(
            num_cols=D,
            block_size=self.block_size,
            device=device,
            dtype=torch.float16,
        )

        for b in range(B):
            seq_len = cache.seq_lengths[b].item()
            for t in range(seq_len):
                mask = cache.outlier_mask[b, head_idx, t, :]  # [D]
                if mask.any():
                    col_indices = torch.where(mask)[0].to(dtype=torch.int32)
                    values = cache.outlier_values[b, head_idx, t, mask]
                    csr.append_token(col_indices, values)
                else:
                    csr.append_token(
                        torch.zeros(0, dtype=torch.int32, device=device),
                        torch.zeros(0, dtype=torch.float16, device=device),
                    )

        return csr

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _quantize_batch(
        self,
        tensor: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> QuantizedKVCache:
        """Internal batched quantization dispatcher."""
        assert tensor.ndim == 4, f"Expected [B, H, S, D], got {tensor.shape}"
        B, H, S, D = tensor.shape
        device = tensor.device

        # Resolve scaling factors with correct shape [H, D]
        if self.scaling_factors is not None:
            scales = self.scaling_factors.to(device=device, dtype=torch.float16)
        else:
            scales = torch.ones(H, D, dtype=torch.float16, device=device)
        codebook = self.codebook.to(device=device)

        q_idx, o_mask, o_vals, _ = fused_quantize_batch(
            tensor, codebook, scales, attention_mask, self.outlier_fraction
        )

        if attention_mask is not None:
            seq_lengths = attention_mask.sum(dim=1).long()
        else:
            seq_lengths = torch.full((B,), S, dtype=torch.long, device=device)

        return QuantizedKVCache(
            quantized_indices=q_idx,
            outlier_mask=o_mask,
            outlier_values=o_vals,
            scaling_factors=scales,
            codebook=codebook,
            seq_lengths=seq_lengths,
        )

    def _resolve_scales(
        self,
        tensor: torch.Tensor,
        override: torch.Tensor | None,
    ) -> torch.Tensor:
        """Resolve scaling factors with fallback to unit scales."""
        if override is not None:
            return override.to(device=tensor.device, dtype=torch.float16)
        if self.scaling_factors is not None:
            return self.scaling_factors.to(device=tensor.device, dtype=torch.float16)
        # Fallback: unit scales
        if tensor.ndim == 3:
            H, _S, D = tensor.shape
        elif tensor.ndim == 4:
            _B, H, _S, D = tensor.shape
        else:
            H, D = tensor.shape
        return torch.ones(H, D, dtype=torch.float16, device=tensor.device)

    def __repr__(self) -> str:
        return (
            f"BatchedKVQuantizer(codebook_size={self.codebook.numel()}, "
            f"outlier_fraction={self.outlier_fraction}, "
            f"block_size={self.block_size})"
        )


# ---------------------------------------------------------------------------
# Convenience: sequence length extraction
# ---------------------------------------------------------------------------


def get_sequence_lengths(attention_mask: torch.Tensor) -> torch.Tensor:
    """Extract actual sequence lengths from an attention mask.

    Args:
        attention_mask: ``[batch_size, seq_len]`` — 1=real token, 0=padding.

    Returns:
        ``[batch_size]`` — actual length of each sequence.
    """
    return attention_mask.sum(dim=1).long()

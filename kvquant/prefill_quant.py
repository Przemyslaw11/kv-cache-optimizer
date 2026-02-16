"""Drop-in quantized attention replacement for the prefill phase.

Composes ``BatchedKVQuantizer`` and blocked sparse matrices into a complete
attention module that quantizes KV cache during prefill.

This module provides ``PrefillQuantizedAttention`` — the top-level class
that experiment scripts and model patches should use.  It orchestrates:

    1. Quantization of keys/values via ``BatchedKVQuantizer``.
    2. Sparse outlier storage via ``BlockedCSCMatrix`` / ``BlockedCSRMatrix``.
    3. Dequantization for attention score computation.
    4. Standard scaled dot-product attention.

Usage:
    >>> attn = PrefillQuantizedAttention(
    ...     num_heads=32, head_dim=128, codebook=codebook, scaling_factors=scales
    ... )
    >>> output = attn.forward(query, key, value, attention_mask, is_prefill=True)
"""

from __future__ import annotations

import logging
import math

import torch

from kvquant.batched_quant import BatchedKVQuantizer, QuantizedKVCache
from kvquant.nuq import create_heuristic_codebook

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cache append helper
# ---------------------------------------------------------------------------


def _append_to_cache(
    existing: QuantizedKVCache,
    new: QuantizedKVCache,
) -> QuantizedKVCache:
    """Append new quantized tokens to an existing cache along the seq_len axis.

    Concatenates ``quantized_indices``, ``outlier_mask``, and
    ``outlier_values`` along ``dim=2`` (seq_len) and updates
    ``seq_lengths`` accordingly.

    Args:
        existing: Previously stored quantized cache.
        new: Freshly quantized cache for the new token(s).

    Returns:
        Merged ``QuantizedKVCache`` covering the full sequence.
    """
    return QuantizedKVCache(
        quantized_indices=torch.cat([existing.quantized_indices, new.quantized_indices], dim=2),
        outlier_mask=torch.cat([existing.outlier_mask, new.outlier_mask], dim=2),
        outlier_values=torch.cat([existing.outlier_values, new.outlier_values], dim=2),
        scaling_factors=existing.scaling_factors,  # shared across sequence
        codebook=existing.codebook,
        seq_lengths=existing.seq_lengths + new.seq_lengths,
    )


class PrefillQuantizedAttention:
    """Quantized attention module supporting both prefill and generation phases.

    During **prefill**, keys and values are quantized in a single batched
    pass (all tokens at once).  During **generation**, keys and values are
    quantized one token at a time and appended to the cache.

    Args:
        num_heads: Number of attention heads.
        head_dim: Dimension of each attention head.
        codebook: NUQ codebook tensor ``[num_levels]`` or a string like
            ``"nuq3"`` to use a heuristic codebook.
        scaling_factors: Per-channel scaling factors ``[num_heads, head_dim]``.
            If ``None``, unit scales are used (suitable for testing).
        outlier_fraction: Fraction of values treated as outliers (default 0.01).
        block_size: Block size for blocked sparse matrices (default 256).

    Example:
        >>> attn = PrefillQuantizedAttention(
        ...     num_heads=32, head_dim=128, codebook="nuq3",
        ... )
        >>> q = torch.randn(2, 32, 64, 128, dtype=torch.float16)
        >>> k = torch.randn(2, 32, 64, 128, dtype=torch.float16)
        >>> v = torch.randn(2, 32, 64, 128, dtype=torch.float16)
        >>> mask = torch.ones(2, 64, dtype=torch.long)
        >>> out = attn.forward(q, k, v, mask, is_prefill=True)
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        codebook: torch.Tensor | str = "nuq3",
        scaling_factors: torch.Tensor | None = None,
        outlier_fraction: float = 0.01,
        block_size: int = 256,
    ) -> None:
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.outlier_fraction = outlier_fraction
        self.block_size = block_size

        # Resolve codebook
        if isinstance(codebook, str):
            self._codebook = create_heuristic_codebook(codebook)
        else:
            self._codebook = codebook

        self._scaling_factors = scaling_factors

        # Build the quantizer
        self.quantizer = BatchedKVQuantizer(
            codebook=self._codebook,
            scaling_factors=self._scaling_factors,
            outlier_fraction=outlier_fraction,
            block_size=block_size,
        )

        # Cached quantized KV (set after prefill)
        self._key_cache: QuantizedKVCache | None = None
        self._value_cache: QuantizedKVCache | None = None

        logger.info(
            "PrefillQuantizedAttention: heads=%d, head_dim=%d, "
            "codebook=%d levels, outlier=%.3f, block_size=%d",
            num_heads,
            head_dim,
            self._codebook.numel(),
            outlier_fraction,
            block_size,
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        is_prefill: bool = False,
    ) -> torch.Tensor:
        """Forward pass with quantized KV cache.

        Args:
            query: ``[batch_size, num_heads, seq_len_q, head_dim]`` — fp16.
            key: ``[batch_size, num_heads, seq_len_kv, head_dim]`` — fp16.
            value: ``[batch_size, num_heads, seq_len_kv, head_dim]`` — fp16.
            attention_mask: ``[batch_size, seq_len_kv]`` — 1=real, 0=padding.
            is_prefill: If ``True``, quantize the full KV cache in one pass.
                If ``False``, quantize the last token only (generation).

        Returns:
            Attention output ``[batch_size, num_heads, seq_len_q, head_dim]``
            — fp16.
        """
        assert query.ndim == 4, f"Expected [B, H, S, D], got {query.shape}"
        assert key.ndim == 4, f"Expected [B, H, S, D], got {key.shape}"
        assert value.ndim == 4, f"Expected [B, H, S, D], got {value.shape}"

        if is_prefill:
            dequantized_key, dequantized_value = self._prefill_quantize(key, value, attention_mask)
        else:
            dequantized_key, dequantized_value = self._generation_quantize(
                key, value, attention_mask
            )

        # Standard scaled dot-product attention
        output = self._compute_attention(query, dequantized_key, dequantized_value, attention_mask)
        return output

    # ------------------------------------------------------------------
    # Quantize-then-dequantize paths
    # ------------------------------------------------------------------

    def _prefill_quantize(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize entire KV cache during prefill, then dequantize for attention.

        The quantized cache is stored in ``self._key_cache`` and
        ``self._value_cache`` for potential reuse in generation.
        """
        self._key_cache = self.quantizer.quantize_keys(key, attention_mask)
        self._value_cache = self.quantizer.quantize_values(value, attention_mask)

        dequantized_key = self.quantizer.dequantize(self._key_cache)
        dequantized_value = self.quantizer.dequantize(self._value_cache)

        return dequantized_key, dequantized_value

    def _generation_quantize(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize a single new token during generation and append to cache.

        If a prefill cache exists (``self._key_cache`` is not None), the new
        token's quantized data is appended to the existing cache tensors,
        giving an incrementally growing quantized KV cache. If no cache
        exists yet, a fresh cache is created from the provided tensors.

        The appended cache is then dequantized in full for the attention
        computation. This ensures the generation path is functionally
        identical to re-quantizing the entire KV sequence, but avoids
        redundant work on previously-quantized tokens.

        Args:
            key: ``[B, H, 1, D]`` or ``[B, H, S_new, D]`` — new token(s).
            value: ``[B, H, 1, D]`` or ``[B, H, S_new, D]`` — new token(s).
            attention_mask: ``[B, S_total]`` covering the full sequence
                (prefill + new tokens), or ``None``.

        Returns:
            Tuple of dequantized key/value tensors covering the full
            cache length: ``[B, H, S_total, D]``.
        """
        _B, _H, S_new, _D = key.shape

        # Quantize only the new token(s)
        new_mask: torch.Tensor | None = None
        if attention_mask is not None:
            # The last S_new columns of the mask correspond to the new tokens
            new_mask = attention_mask[:, -S_new:]
        cache_k = self.quantizer.quantize_keys(key, new_mask)
        cache_v = self.quantizer.quantize_values(value, new_mask)

        if self._key_cache is not None and self._value_cache is not None:
            # Append new token(s) to existing cache
            self._key_cache = _append_to_cache(self._key_cache, cache_k)
            self._value_cache = _append_to_cache(self._value_cache, cache_v)
        else:
            # No existing cache — use the new cache directly
            self._key_cache = cache_k
            self._value_cache = cache_v

        # Dequantize the full cache for attention computation
        dequant_k = self.quantizer.dequantize(self._key_cache)
        dequant_v = self.quantizer.dequantize(self._value_cache)

        return dequant_k, dequant_v

    # ------------------------------------------------------------------
    # Attention computation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Standard scaled dot-product attention.

        Args:
            query: ``[B, H, S_q, D]`` — fp16.
            key: ``[B, H, S_kv, D]`` — fp16.
            value: ``[B, H, S_kv, D]`` — fp16.
            attention_mask: ``[B, S_kv]`` — 1=real, 0=padding.

        Returns:
            ``[B, H, S_q, D]`` — fp16.
        """
        _B, _H, _S_q, D = query.shape
        scale = 1.0 / math.sqrt(D)

        # QK^T: [B, H, S_q, S_kv]
        attn_weights = torch.matmul(query.float(), key.float().transpose(-2, -1)) * scale

        # Apply attention mask (convert to additive mask)
        if attention_mask is not None:
            # attention_mask: [B, S_kv] → [B, 1, 1, S_kv]
            mask = attention_mask.unsqueeze(1).unsqueeze(2).to(dtype=attn_weights.dtype)
            attn_weights = attn_weights.masked_fill(mask == 0, float("-inf"))

        attn_weights = torch.softmax(attn_weights, dim=-1)

        # Handle NaN from all-masked rows
        attn_weights = attn_weights.nan_to_num(0.0)

        # Weighted sum: [B, H, S_q, D]
        output = torch.matmul(attn_weights, value.float())
        return output.to(dtype=torch.float16)

    # ------------------------------------------------------------------
    # Cache access
    # ------------------------------------------------------------------

    @property
    def key_cache(self) -> QuantizedKVCache | None:
        """Access the quantized key cache (set after prefill)."""
        return self._key_cache

    @property
    def value_cache(self) -> QuantizedKVCache | None:
        """Access the quantized value cache (set after prefill)."""
        return self._value_cache

    def clear_cache(self) -> None:
        """Clear the cached quantized KV tensors."""
        self._key_cache = None
        self._value_cache = None

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def compute_round_trip_error(
        self,
        original: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, float]:
        """Compute round-trip quantization error for a given tensor.

        Quantizes and immediately dequantizes, then measures the error.

        Args:
            original: ``[B, H, S, D]`` — fp16 keys or values.
            attention_mask: ``[B, S]`` — optional.

        Returns:
            Dictionary with ``"mse"``, ``"max_abs_error"``, ``"relative_error"``.
        """
        cache = self.quantizer.quantize_keys(original, attention_mask)
        reconstructed = self.quantizer.dequantize(cache)

        # Only measure on real tokens
        if attention_mask is not None:
            mask_4d = attention_mask.unsqueeze(1).unsqueeze(-1)  # [B, 1, S, 1]
            mask_4d = mask_4d.expand_as(original).bool()
            orig_masked = original[mask_4d].float()
            recon_masked = reconstructed[mask_4d].float()
        else:
            orig_masked = original.float().flatten()
            recon_masked = reconstructed.float().flatten()

        diff = orig_masked - recon_masked
        mse = (diff**2).mean().item()
        max_abs = diff.abs().max().item()
        orig_norm = orig_masked.norm().item()
        rel_err = diff.norm().item() / max(orig_norm, 1e-8)

        return {
            "mse": mse,
            "max_abs_error": max_abs,
            "relative_error": rel_err,
        }

    def __repr__(self) -> str:
        return (
            f"PrefillQuantizedAttention(num_heads={self.num_heads}, "
            f"head_dim={self.head_dim}, "
            f"codebook={self._codebook.numel()} levels, "
            f"outlier_fraction={self.outlier_fraction})"
        )

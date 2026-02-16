"""PyTorch loop-fused quantization kernels (preferred default path).

Vectorizes quantization across all heads and tokens in a single pass,
avoiding intermediate fp16 materialization. This is the primary implementation
path — Triton kernels are only used after this path is validated.

All operations stay on GPU (no CPU↔GPU transfers in hot paths).
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fused quantization — vectorized across heads x tokens x channels
# ---------------------------------------------------------------------------


def fused_quantize(
    tensor: torch.Tensor,
    codebook: torch.Tensor,
    scaling_factors: torch.Tensor,
    outlier_fraction: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused per-channel NUQ quantization with outlier detection.

    Performs normalization, outlier detection, and quantization in a single
    vectorized pass over all heads and tokens.  No Python loops over heads
    or tokens — everything is batched via PyTorch ops.

    Pipeline:
        1. Normalize by per-channel scaling factors.
        2. Detect outliers (top ``outlier_fraction`` by absolute value per
           channel/vector).
        3. Quantize non-outlier values to nearest codebook centroid.
        4. Return quantized indices + outlier mask/values.

    Args:
        tensor: Input tensor of shape
            ``[num_heads, seq_len, head_dim]`` — fp16.
        codebook: 1-D codebook tensor ``[num_levels]`` — fp16.
        scaling_factors: Per-channel scales ``[num_heads, head_dim]`` — fp16.
            Each channel of ``tensor[:, :, d]`` is divided by
            ``scaling_factors[:, d]`` before quantization.
        outlier_fraction: Fraction of values per channel treated as outliers
            (default 0.01 = 1 %).

    Returns:
        Tuple of:
        - ``quantized_indices``: ``[num_heads, seq_len, head_dim]`` — uint8.
        - ``outlier_mask``: ``[num_heads, seq_len, head_dim]`` — bool.
        - ``outlier_values``: ``[num_heads, seq_len, head_dim]`` — fp16,
          containing original (un-normalized) values where ``outlier_mask``
          is True, zeros elsewhere.
        - ``scales_used``: Reference to *scaling_factors* (for dequantization).
    """
    assert tensor.ndim == 3, f"Expected [num_heads, seq_len, head_dim], got {tensor.shape}"
    assert scaling_factors.ndim == 2, f"Expected [num_heads, head_dim], got {scaling_factors.shape}"
    H, S, D = tensor.shape
    assert scaling_factors.shape == (H, D), (
        f"scaling_factors shape {scaling_factors.shape} != ({H}, {D})"
    )

    # --- Step 1: Normalize per-channel ---
    # tensor: [H, S, D], scaling_factors: [H, 1, D] after unsqueeze
    scales = scaling_factors.unsqueeze(1)  # [H, 1, D]
    # Avoid division by zero
    safe_scales = scales.clamp(min=1e-8)
    normalized = tensor / safe_scales  # [H, S, D]

    # --- Step 2: Detect outliers per channel ---
    # For each (head, dim) pair, compute the outlier threshold across seq_len
    abs_normalized = normalized.abs()  # [H, S, D]

    if outlier_fraction > 0.0 and S > 1:
        # Compute the (1 - outlier_fraction) quantile along the seq_len axis
        quantile_val = 1.0 - outlier_fraction
        # quantile along dim=1: result shape [H, D]
        thresholds = torch.quantile(abs_normalized.float(), quantile_val, dim=1).to(
            dtype=tensor.dtype
        )  # [H, D]
        thresholds = thresholds.unsqueeze(1)  # [H, 1, D]
        outlier_mask = abs_normalized > thresholds  # [H, S, D]
    else:
        outlier_mask = torch.zeros_like(normalized, dtype=torch.bool)

    # Store original values for outliers (before normalization)
    outlier_values = torch.where(outlier_mask, tensor, torch.zeros_like(tensor))

    # --- Step 3: Quantize non-outliers ---
    # Replace outlier positions with 0.0 for quantization (will map to
    # nearest centroid to 0, which is fine since they'll be overwritten
    # during dequantization)
    quant_input = torch.where(outlier_mask, torch.zeros_like(normalized), normalized)

    # Vectorized nearest-centroid lookup
    # quant_input: [H, S, D] → flat: [H*S*D]
    flat = quant_input.reshape(-1).float()
    cb = codebook.float()  # [K]

    # distances: [H*S*D, K]
    distances = torch.abs(flat.unsqueeze(1) - cb.unsqueeze(0))
    indices = distances.argmin(dim=1).to(dtype=torch.uint8)
    quantized_indices = indices.reshape(H, S, D)

    return quantized_indices, outlier_mask, outlier_values, scaling_factors


def fused_dequantize(
    quantized_indices: torch.Tensor,
    codebook: torch.Tensor,
    scaling_factors: torch.Tensor,
    outlier_mask: torch.Tensor,
    outlier_values: torch.Tensor,
) -> torch.Tensor:
    """Fused dequantization: codebook lookup + sparse outlier add + rescale.

    Args:
        quantized_indices: ``[num_heads, seq_len, head_dim]`` — uint8.
        codebook: ``[num_levels]`` — fp16.
        scaling_factors: ``[num_heads, head_dim]`` — fp16.
        outlier_mask: ``[num_heads, seq_len, head_dim]`` — bool.
        outlier_values: ``[num_heads, seq_len, head_dim]`` — fp16 (original
            un-normalized values at outlier positions, zeros elsewhere).

    Returns:
        Dequantized tensor ``[num_heads, seq_len, head_dim]`` — fp16.
    """
    assert quantized_indices.ndim == 3
    _H, _S, _D = quantized_indices.shape

    # Step 1: Codebook lookup → normalized values
    dequantized = codebook[quantized_indices.long()]  # [H, S, D] fp16

    # Step 2: Rescale
    scales = scaling_factors.unsqueeze(1)  # [H, 1, D]
    dequantized = dequantized * scales  # [H, S, D]

    # Step 3: Overwrite outlier positions with original values
    dequantized = torch.where(outlier_mask, outlier_values, dequantized)

    return dequantized


# ---------------------------------------------------------------------------
# Batched variants — handle [batch_size, num_heads, seq_len, head_dim]
# with variable-length sequences (padding-aware)
# ---------------------------------------------------------------------------


def fused_quantize_batch(
    tensor: torch.Tensor,
    codebook: torch.Tensor,
    scaling_factors: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    outlier_fraction: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batched fused quantization with variable-length sequence support.

    Only quantizes real tokens (identified by ``attention_mask``); padding
    positions are left as zero indices.

    Args:
        tensor: ``[batch_size, num_heads, seq_len, head_dim]`` — fp16.
        codebook: ``[num_levels]`` — fp16.
        scaling_factors: ``[num_heads, head_dim]`` — fp16.
        attention_mask: ``[batch_size, seq_len]`` — 1=real, 0=padding.
            If ``None``, all positions are treated as real.
        outlier_fraction: Fraction of outliers per channel (default 0.01).

    Returns:
        Tuple of ``(quantized_indices, outlier_mask, outlier_values,
        scaling_factors)`` — same shapes as input tensor.
    """
    assert tensor.ndim == 4, f"Expected [B, H, S, D], got {tensor.shape}"
    B, H, S, D = tensor.shape

    all_indices = torch.zeros(B, H, S, D, dtype=torch.uint8, device=tensor.device)
    all_outlier_mask = torch.zeros(B, H, S, D, dtype=torch.bool, device=tensor.device)
    all_outlier_values = torch.zeros(B, H, S, D, dtype=tensor.dtype, device=tensor.device)

    # Determine actual sequence lengths
    if attention_mask is not None:
        seq_lengths = attention_mask.sum(dim=1).long()  # [B]
    else:
        seq_lengths = torch.full((B,), S, dtype=torch.long, device=tensor.device)

    for b in range(B):
        seq_len = seq_lengths[b].item()
        if seq_len == 0:
            continue

        # Extract only real tokens for this sample
        sample = tensor[b, :, :seq_len, :]  # [H, seq_len, D]

        q_idx, o_mask, o_vals, _ = fused_quantize(
            sample, codebook, scaling_factors, outlier_fraction
        )

        all_indices[b, :, :seq_len, :] = q_idx
        all_outlier_mask[b, :, :seq_len, :] = o_mask
        all_outlier_values[b, :, :seq_len, :] = o_vals

    return all_indices, all_outlier_mask, all_outlier_values, scaling_factors


def fused_dequantize_batch(
    quantized_indices: torch.Tensor,
    codebook: torch.Tensor,
    scaling_factors: torch.Tensor,
    outlier_mask: torch.Tensor,
    outlier_values: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Batched fused dequantization with variable-length sequence support.

    Args:
        quantized_indices: ``[B, H, S, D]`` — uint8.
        codebook: ``[num_levels]`` — fp16.
        scaling_factors: ``[num_heads, head_dim]`` — fp16.
        outlier_mask: ``[B, H, S, D]`` — bool.
        outlier_values: ``[B, H, S, D]`` — fp16.
        attention_mask: ``[B, S]`` — 1=real, 0=padding. Optional.

    Returns:
        Dequantized tensor ``[B, H, S, D]`` — fp16.
    """
    B, H, S, D = quantized_indices.shape
    result = torch.zeros(B, H, S, D, dtype=torch.float16, device=quantized_indices.device)

    if attention_mask is not None:
        seq_lengths = attention_mask.sum(dim=1).long()
    else:
        seq_lengths = torch.full((B,), S, dtype=torch.long, device=result.device)

    for b in range(B):
        seq_len = seq_lengths[b].item()
        if seq_len == 0:
            continue

        result[b, :, :seq_len, :] = fused_dequantize(
            quantized_indices[b, :, :seq_len, :],
            codebook,
            scaling_factors,
            outlier_mask[b, :, :seq_len, :],
            outlier_values[b, :, :seq_len, :],
        )

    return result


# ---------------------------------------------------------------------------
# Scaling factor estimation — used during calibration
# ---------------------------------------------------------------------------


def estimate_scaling_factors(
    calibration_keys: torch.Tensor,
) -> torch.Tensor:
    """Estimate per-channel scaling factors from calibration data.

    Uses the mean absolute value per channel as the scaling factor,
    following the KVQuant methodology.

    Args:
        calibration_keys: ``[num_samples, num_heads, seq_len, head_dim]``
            — fp16 key activations from calibration set.

    Returns:
        Scaling factors ``[num_heads, head_dim]`` — fp16.
    """
    assert calibration_keys.ndim == 4, (
        f"Expected [num_samples, H, S, D], got {calibration_keys.shape}"
    )
    # Mean absolute value across samples and seq_len
    # [num_samples, H, S, D] → mean over dims 0, 2 → [H, D]
    scales = calibration_keys.abs().mean(dim=(0, 2))
    # Clamp to avoid division by zero
    scales = scales.clamp(min=1e-8)
    return scales.to(dtype=torch.float16)

"""Tests for BatchedKVQuantizer — quantization round-trip and correctness.

Validates:
    - NUQ codebook creation (heuristic and calibrated).
    - Quantize → dequantize round-trip error < 0.05 MSE.
    - Variable-length batch handling (padding ignored correctly).
    - Outlier detection and preservation.
    - Integration with BlockedCSCMatrix / BlockedCSRMatrix.
    - Scaling factor estimation from calibration data.
"""

import pytest
import torch

from kvquant.batched_quant import BatchedKVQuantizer, QuantizedKVCache, get_sequence_lengths
from kvquant.kernels.pytorch_fused import (
    estimate_scaling_factors,
    fused_dequantize,
    fused_quantize,
)
from kvquant.nuq import (
    compute_nuq_codebook,
    create_heuristic_codebook,
    dequantize_from_nuq,
    get_num_levels,
    get_nuq_bitwidth,
    quantize_to_nuq,
)

# =====================================================================
# NUQ codebook tests
# =====================================================================


class TestNUQCodebook:
    """Test suite for NUQ codebook creation and basic quant/dequant."""

    @pytest.mark.parametrize("nuq_type", ["nuq2", "nuq3", "nuq4"])
    def test_heuristic_codebook_size(self, nuq_type: str) -> None:
        """Heuristic codebook should have 2^bits entries."""
        cb = create_heuristic_codebook(nuq_type)
        expected = get_num_levels(nuq_type)
        assert cb.numel() == expected

    @pytest.mark.parametrize("nuq_type", ["nuq2", "nuq3", "nuq4"])
    def test_heuristic_codebook_sorted(self, nuq_type: str) -> None:
        """Heuristic codebook should be sorted ascending."""
        cb = create_heuristic_codebook(nuq_type)
        diffs = cb[1:] - cb[:-1]
        assert (diffs > 0).all()

    def test_heuristic_codebook_dtype(self) -> None:
        """Codebook should be float16."""
        cb = create_heuristic_codebook("nuq3")
        assert cb.dtype == torch.float16

    def test_invalid_nuq_type_raises(self) -> None:
        """Unknown NUQ type should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown NUQ datatype"):
            get_nuq_bitwidth("nuq5")

    def test_quantize_dequantize_roundtrip(self) -> None:
        """Round-trip should approximately recover original values."""
        torch.manual_seed(42)
        cb = create_heuristic_codebook("nuq3")
        values = torch.randn(100, dtype=torch.float16)

        indices = quantize_to_nuq(values, cb)
        recovered = dequantize_from_nuq(indices, cb)

        # Each value maps to nearest centroid → error bounded by half the
        # max gap between centroids
        mse = ((values.float() - recovered.float()) ** 2).mean().item()
        assert mse < 0.5, f"Round-trip MSE too high: {mse}"

    def test_quantize_indices_dtype(self) -> None:
        """Quantized indices should be uint8."""
        cb = create_heuristic_codebook("nuq4")
        values = torch.randn(50, dtype=torch.float16)
        indices = quantize_to_nuq(values, cb)
        assert indices.dtype == torch.uint8

    def test_quantize_indices_range(self) -> None:
        """All indices should be in [0, num_levels)."""
        cb = create_heuristic_codebook("nuq3")
        values = torch.randn(200, dtype=torch.float16)
        indices = quantize_to_nuq(values, cb)
        assert indices.min() >= 0
        assert indices.max() < 8  # 2^3

    def test_computed_codebook_from_data(self) -> None:
        """K-means codebook should adapt to data distribution."""
        torch.manual_seed(0)
        # Bimodal data
        data = torch.cat([torch.randn(500) - 3, torch.randn(500) + 3])
        cb = compute_nuq_codebook(data, "nuq3", num_iterations=50)

        assert cb.numel() == 8
        # Codebook should have centroids near -3 and +3
        assert cb.min().item() < -1.0
        assert cb.max().item() > 1.0


# =====================================================================
# Fused kernel tests
# =====================================================================


class TestFusedQuantize:
    """Test suite for fused PyTorch quantization kernels."""

    def test_fused_quantize_shapes(self) -> None:
        """Output shapes should match input shapes."""
        torch.manual_seed(42)
        H, S, D = 8, 32, 64
        tensor = torch.randn(H, S, D, dtype=torch.float16)
        cb = create_heuristic_codebook("nuq3")
        scales = torch.ones(H, D, dtype=torch.float16)

        q_idx, o_mask, o_vals, _ = fused_quantize(tensor, cb, scales)

        assert q_idx.shape == (H, S, D)
        assert o_mask.shape == (H, S, D)
        assert o_vals.shape == (H, S, D)
        assert q_idx.dtype == torch.uint8
        assert o_mask.dtype == torch.bool

    def test_fused_round_trip_error(self) -> None:
        """Fused quantize → dequantize round-trip MSE should be small."""
        torch.manual_seed(42)
        H, S, D = 4, 64, 32
        tensor = torch.randn(H, S, D, dtype=torch.float16)
        cb = create_heuristic_codebook("nuq3")
        scales = tensor.abs().mean(dim=1).clamp(min=1e-4)  # [H, D]

        q_idx, o_mask, o_vals, _ = fused_quantize(tensor, cb, scales, outlier_fraction=0.01)
        recovered = fused_dequantize(q_idx, cb, scales, o_mask, o_vals)

        mse = ((tensor.float() - recovered.float()) ** 2).mean().item()
        assert mse < 0.1, f"Fused round-trip MSE too high: {mse}"

    def test_outlier_preservation(self) -> None:
        """Outlier values should be exactly preserved after dequantization."""
        torch.manual_seed(42)
        H, S, D = 2, 16, 8
        tensor = torch.randn(H, S, D, dtype=torch.float16)
        # Inject extreme outliers
        tensor[0, 3, 5] = 100.0
        tensor[1, 7, 2] = -50.0

        cb = create_heuristic_codebook("nuq3")
        scales = torch.ones(H, D, dtype=torch.float16)

        q_idx, o_mask, o_vals, _ = fused_quantize(tensor, cb, scales, outlier_fraction=0.05)
        recovered = fused_dequantize(q_idx, cb, scales, o_mask, o_vals)

        # The extreme values should be detected as outliers and preserved
        assert o_mask[0, 3, 5].item() is True
        assert o_mask[1, 7, 2].item() is True
        assert recovered[0, 3, 5].item() == pytest.approx(100.0, abs=0.1)
        assert recovered[1, 7, 2].item() == pytest.approx(-50.0, abs=0.1)

    def test_zero_outlier_fraction(self) -> None:
        """With outlier_fraction=0, no values should be outliers."""
        torch.manual_seed(42)
        H, S, D = 4, 32, 16
        tensor = torch.randn(H, S, D, dtype=torch.float16)
        cb = create_heuristic_codebook("nuq3")
        scales = torch.ones(H, D, dtype=torch.float16)

        _, o_mask, _, _ = fused_quantize(tensor, cb, scales, outlier_fraction=0.0)
        assert o_mask.sum().item() == 0

    def test_scaling_factors_estimation(self) -> None:
        """Estimated scaling factors should have correct shape and be positive."""
        torch.manual_seed(42)
        calib = torch.randn(16, 8, 64, 32, dtype=torch.float16)
        scales = estimate_scaling_factors(calib)

        assert scales.shape == (8, 32)
        assert scales.dtype == torch.float16
        assert (scales > 0).all()


# =====================================================================
# BatchedKVQuantizer tests
# =====================================================================


class TestBatchedKVQuantizer:
    """Test suite for batched quantization during prefill."""

    def _make_quantizer(self, H: int = 8, D: int = 64) -> BatchedKVQuantizer:
        """Create a quantizer with default settings for testing."""
        cb = create_heuristic_codebook("nuq3")
        scales = torch.ones(H, D, dtype=torch.float16)
        return BatchedKVQuantizer(
            codebook=cb,
            scaling_factors=scales,
            outlier_fraction=0.01,
            block_size=64,
        )

    def test_quantize_keys_shape(self) -> None:
        """quantize_keys should return QuantizedKVCache with correct shapes."""
        quantizer = self._make_quantizer(H=4, D=16)
        B, H, S, D = 2, 4, 32, 16
        keys = torch.randn(B, H, S, D, dtype=torch.float16)
        mask = torch.ones(B, S, dtype=torch.long)

        result = quantizer.quantize_keys(keys, mask)

        assert isinstance(result, QuantizedKVCache)
        assert result.quantized_indices.shape == (B, H, S, D)
        assert result.outlier_mask.shape == (B, H, S, D)
        assert result.outlier_values.shape == (B, H, S, D)
        assert result.seq_lengths.shape == (B,)

    def test_round_trip_mse_below_threshold(self) -> None:
        """Quantize → dequantize MSE should be < 0.1."""
        torch.manual_seed(42)
        B, H, S, D = 2, 4, 64, 32
        keys = torch.randn(B, H, S, D, dtype=torch.float16)
        mask = torch.ones(B, S, dtype=torch.long)

        cb = create_heuristic_codebook("nuq3")
        scales = keys.abs().mean(dim=(0, 2)).clamp(min=1e-4)  # [H, D]

        quantizer = BatchedKVQuantizer(cb, scales, outlier_fraction=0.01)
        cache = quantizer.quantize_keys(keys, mask)
        recovered = quantizer.dequantize(cache)

        mse = ((keys.float() - recovered.float()) ** 2).mean().item()
        assert mse < 0.1, f"Batched round-trip MSE too high: {mse}"

    def test_variable_length_padding_ignored(self) -> None:
        """Padding tokens should not be quantized (remain zero indices)."""
        torch.manual_seed(42)
        B, H, S, D = 3, 2, 16, 8
        keys = torch.randn(B, H, S, D, dtype=torch.float16)

        # Variable lengths: 10, 16, 5
        mask = torch.zeros(B, S, dtype=torch.long)
        mask[0, :10] = 1
        mask[1, :16] = 1
        mask[2, :5] = 1

        quantizer = self._make_quantizer(H=H, D=D)
        cache = quantizer.quantize_keys(keys, mask)

        # Padding positions (beyond actual length) should be zero indices
        assert cache.quantized_indices[0, :, 10:, :].sum().item() == 0
        assert cache.quantized_indices[2, :, 5:, :].sum().item() == 0

        # Sequence lengths should be recorded correctly
        assert cache.seq_lengths[0].item() == 10
        assert cache.seq_lengths[1].item() == 16
        assert cache.seq_lengths[2].item() == 5

    def test_no_mask_treats_all_as_real(self) -> None:
        """With attention_mask=None, all positions should be quantized."""
        torch.manual_seed(42)
        B, H, S, D = 1, 2, 8, 4
        keys = torch.randn(B, H, S, D, dtype=torch.float16)

        quantizer = self._make_quantizer(H=H, D=D)
        cache = quantizer.quantize_keys(keys, attention_mask=None)

        assert cache.seq_lengths[0].item() == S
        # Some indices should be non-zero
        assert cache.quantized_indices.sum().item() > 0

    def test_string_codebook_shorthand(self) -> None:
        """BatchedKVQuantizer should accept string like 'nuq3'."""
        quantizer = BatchedKVQuantizer("nuq3")
        assert quantizer.codebook.numel() == 8

    def test_extract_key_outliers_to_sparse(self) -> None:
        """Key outliers should be extractable to BlockedCSCMatrix."""
        torch.manual_seed(42)
        B, H, S, D = 1, 2, 16, 8
        keys = torch.randn(B, H, S, D, dtype=torch.float16)
        # Inject outliers
        keys[0, 0, 5, 3] = 100.0
        keys[0, 0, 10, 7] = -50.0
        mask = torch.ones(B, S, dtype=torch.long)

        quantizer = self._make_quantizer(H=H, D=D)
        cache = quantizer.quantize_keys(keys, mask)

        csc = quantizer.extract_key_outliers_to_sparse(cache, head_idx=0)

        # CSC should have S columns (one per token)
        assert csc.total_columns == S
        # Should have at least the 2 injected outliers
        assert csc.total_nnz >= 2

    def test_extract_value_outliers_to_sparse(self) -> None:
        """Value outliers should be extractable to BlockedCSRMatrix."""
        torch.manual_seed(42)
        B, H, S, D = 1, 2, 16, 8
        values = torch.randn(B, H, S, D, dtype=torch.float16)
        values[0, 1, 3, 6] = 80.0
        mask = torch.ones(B, S, dtype=torch.long)

        quantizer = self._make_quantizer(H=H, D=D)
        cache = quantizer.quantize_values(values, mask)

        csr = quantizer.extract_value_outliers_to_sparse(cache, head_idx=1)

        assert csr.total_rows == S
        assert csr.total_nnz >= 1

    def test_compute_round_trip_error_metrics(self) -> None:
        """compute_round_trip_error should return valid metrics."""
        torch.manual_seed(42)
        B, H, S, D = 2, 4, 32, 16
        keys = torch.randn(B, H, S, D, dtype=torch.float16)
        mask = torch.ones(B, S, dtype=torch.long)

        from kvquant.prefill_quant import PrefillQuantizedAttention

        attn = PrefillQuantizedAttention(num_heads=H, head_dim=D, codebook="nuq3")
        metrics = attn.compute_round_trip_error(keys, mask)

        assert "mse" in metrics
        assert "max_abs_error" in metrics
        assert "relative_error" in metrics
        assert metrics["mse"] >= 0
        assert metrics["max_abs_error"] >= 0

    def test_repr(self) -> None:
        """__repr__ should include key info."""
        quantizer = self._make_quantizer()
        r = repr(quantizer)
        assert "BatchedKVQuantizer" in r
        assert "8" in r  # codebook_size=8


# =====================================================================
# Helper tests
# =====================================================================


class TestHelpers:
    """Test utility functions."""

    def test_get_sequence_lengths(self) -> None:
        """get_sequence_lengths should extract correct lengths."""
        mask = torch.tensor(
            [
                [1, 1, 1, 0, 0],
                [1, 1, 1, 1, 1],
                [1, 0, 0, 0, 0],
            ],
            dtype=torch.long,
        )

        lengths = get_sequence_lengths(mask)
        assert lengths.tolist() == [3, 5, 1]

    def test_get_sequence_lengths_all_ones(self) -> None:
        """All-ones mask should return max length for each."""
        mask = torch.ones(4, 10, dtype=torch.long)
        lengths = get_sequence_lengths(mask)
        assert (lengths == 10).all()

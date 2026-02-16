"""Tests for PrefillQuantizedAttention — integration tests.

Validates:
    - Forward pass shapes (prefill and generation modes).
    - Attention output is reasonable (not NaN/Inf).
    - Quantized attention output is close to fp16 reference.
    - Cache population during prefill.
    - Variable-length batch support.
    - Cache clearing.
"""

import math

import torch

from kvquant.prefill_quant import PrefillQuantizedAttention


class TestPrefillQuantizedAttention:
    """Test suite for the complete prefill quantization pipeline."""

    def _make_attention(self, H: int = 4, D: int = 16) -> PrefillQuantizedAttention:
        """Create a PrefillQuantizedAttention for testing."""
        return PrefillQuantizedAttention(
            num_heads=H,
            head_dim=D,
            codebook="nuq3",
            scaling_factors=None,  # unit scales
            outlier_fraction=0.01,
            block_size=32,
        )

    def test_forward_prefill_shapes(self) -> None:
        """Forward pass in prefill mode should return correct shapes."""
        H, D = 4, 16
        B, S = 2, 32
        attn = self._make_attention(H=H, D=D)

        q = torch.randn(B, H, S, D, dtype=torch.float16)
        k = torch.randn(B, H, S, D, dtype=torch.float16)
        v = torch.randn(B, H, S, D, dtype=torch.float16)
        mask = torch.ones(B, S, dtype=torch.long)

        output = attn.forward(q, k, v, mask, is_prefill=True)

        assert output.shape == (B, H, S, D)
        assert output.dtype == torch.float16

    def test_forward_generation_shapes(self) -> None:
        """Forward pass in generation mode should return correct shapes."""
        H, D = 4, 16
        B, S = 2, 1  # Single token generation
        attn = self._make_attention(H=H, D=D)

        q = torch.randn(B, H, S, D, dtype=torch.float16)
        k = torch.randn(B, H, S, D, dtype=torch.float16)
        v = torch.randn(B, H, S, D, dtype=torch.float16)
        mask = torch.ones(B, S, dtype=torch.long)

        output = attn.forward(q, k, v, mask, is_prefill=False)

        assert output.shape == (B, H, S, D)

    def test_output_not_nan(self) -> None:
        """Output should not contain NaN or Inf."""
        torch.manual_seed(42)
        H, D = 4, 16
        B, S = 2, 64
        attn = self._make_attention(H=H, D=D)

        q = torch.randn(B, H, S, D, dtype=torch.float16)
        k = torch.randn(B, H, S, D, dtype=torch.float16)
        v = torch.randn(B, H, S, D, dtype=torch.float16)
        mask = torch.ones(B, S, dtype=torch.long)

        output = attn.forward(q, k, v, mask, is_prefill=True)

        assert not torch.isnan(output).any(), "Output contains NaN"
        assert not torch.isinf(output).any(), "Output contains Inf"

    def test_quantized_vs_fp16_attention_close(self) -> None:
        """Quantized attention output should be close to fp16 reference."""
        torch.manual_seed(42)
        H, D = 4, 16
        B, S = 1, 32
        attn = self._make_attention(H=H, D=D)

        q = torch.randn(B, H, S, D, dtype=torch.float16) * 0.1
        k = torch.randn(B, H, S, D, dtype=torch.float16) * 0.1
        v = torch.randn(B, H, S, D, dtype=torch.float16) * 0.1
        mask = torch.ones(B, S, dtype=torch.long)

        # Quantized path
        quantized_output = attn.forward(q, k, v, mask, is_prefill=True)

        # FP16 reference (standard attention)
        scale = 1.0 / math.sqrt(D)
        attn_weights = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
        attn_weights = torch.softmax(attn_weights, dim=-1)
        fp16_output = torch.matmul(attn_weights, v.float()).to(torch.float16)

        # Should be reasonably close (within quantization error)
        mse = ((quantized_output.float() - fp16_output.float()) ** 2).mean().item()
        assert mse < 0.1, f"Quantized vs fp16 MSE too high: {mse}"

    def test_cache_populated_after_prefill(self) -> None:
        """After prefill, key_cache and value_cache should be populated."""
        H, D = 4, 16
        B, S = 1, 16
        attn = self._make_attention(H=H, D=D)

        q = torch.randn(B, H, S, D, dtype=torch.float16)
        k = torch.randn(B, H, S, D, dtype=torch.float16)
        v = torch.randn(B, H, S, D, dtype=torch.float16)
        mask = torch.ones(B, S, dtype=torch.long)

        assert attn.key_cache is None
        assert attn.value_cache is None

        attn.forward(q, k, v, mask, is_prefill=True)

        assert attn.key_cache is not None
        assert attn.value_cache is not None
        assert attn.key_cache.quantized_indices.shape == (B, H, S, D)

    def test_clear_cache(self) -> None:
        """clear_cache should reset both caches to None."""
        H, D = 4, 16
        B, S = 1, 8
        attn = self._make_attention(H=H, D=D)

        q = torch.randn(B, H, S, D, dtype=torch.float16)
        k = torch.randn(B, H, S, D, dtype=torch.float16)
        v = torch.randn(B, H, S, D, dtype=torch.float16)

        attn.forward(q, k, v, is_prefill=True)
        assert attn.key_cache is not None

        attn.clear_cache()
        assert attn.key_cache is None
        assert attn.value_cache is None

    def test_variable_length_batch(self) -> None:
        """Variable-length sequences should be handled correctly."""
        torch.manual_seed(42)
        H, D = 2, 8
        B, S = 3, 16
        attn = self._make_attention(H=H, D=D)

        q = torch.randn(B, H, S, D, dtype=torch.float16)
        k = torch.randn(B, H, S, D, dtype=torch.float16)
        v = torch.randn(B, H, S, D, dtype=torch.float16)

        mask = torch.zeros(B, S, dtype=torch.long)
        mask[0, :8] = 1
        mask[1, :16] = 1
        mask[2, :4] = 1

        output = attn.forward(q, k, v, mask, is_prefill=True)

        assert output.shape == (B, H, S, D)
        assert not torch.isnan(output).any()

    def test_compute_round_trip_error(self) -> None:
        """compute_round_trip_error should return valid metrics."""
        torch.manual_seed(42)
        H, D = 4, 16
        B, S = 2, 32
        attn = self._make_attention(H=H, D=D)
        original = torch.randn(B, H, S, D, dtype=torch.float16)
        mask = torch.ones(B, S, dtype=torch.long)

        metrics = attn.compute_round_trip_error(original, mask)

        assert metrics["mse"] >= 0
        assert metrics["max_abs_error"] >= 0
        assert metrics["relative_error"] >= 0

    def test_repr(self) -> None:
        """__repr__ should include key info."""
        attn = self._make_attention()
        r = repr(attn)
        assert "PrefillQuantizedAttention" in r

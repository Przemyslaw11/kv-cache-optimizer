"""Tests for incremental generation cache in PrefillQuantizedAttention.

Validates the Phase 2 incremental cache management:
    - Cache grows correctly when tokens are appended during generation.
    - Prefill → generation transition works seamlessly.
    - Multi-step generation produces correct shapes.
    - Cache clearing between sequences.
    - _append_to_cache merges quantized caches correctly.
"""

import torch

from kvquant.batched_quant import QuantizedKVCache
from kvquant.nuq import create_heuristic_codebook
from kvquant.prefill_quant import PrefillQuantizedAttention, _append_to_cache


class TestAppendToCache:
    """Tests for the _append_to_cache helper."""

    def _make_cache(self, B: int, H: int, S: int, D: int) -> QuantizedKVCache:
        """Create a synthetic QuantizedKVCache for testing."""
        cb = create_heuristic_codebook("nuq3")
        return QuantizedKVCache(
            quantized_indices=torch.randint(0, 8, (B, H, S, D), dtype=torch.uint8),
            outlier_mask=torch.zeros(B, H, S, D, dtype=torch.bool),
            outlier_values=torch.zeros(B, H, S, D, dtype=torch.float16),
            scaling_factors=torch.ones(H, D, dtype=torch.float16),
            codebook=cb,
            seq_lengths=torch.full((B,), S, dtype=torch.long),
        )

    def test_append_increases_seq_len(self) -> None:
        """Appending should increase the seq_len dimension."""
        B, H, D = 2, 4, 16
        existing = self._make_cache(B, H, 32, D)
        new = self._make_cache(B, H, 1, D)

        merged = _append_to_cache(existing, new)

        assert merged.quantized_indices.shape == (B, H, 33, D)
        assert merged.outlier_mask.shape == (B, H, 33, D)
        assert merged.outlier_values.shape == (B, H, 33, D)

    def test_append_updates_seq_lengths(self) -> None:
        """Merged cache should have updated seq_lengths."""
        B, H, D = 2, 4, 16
        existing = self._make_cache(B, H, 10, D)
        new = self._make_cache(B, H, 1, D)

        merged = _append_to_cache(existing, new)

        assert merged.seq_lengths.tolist() == [11, 11]

    def test_append_preserves_existing_data(self) -> None:
        """Existing quantized indices should be preserved after append."""
        B, H, D = 1, 2, 8
        existing = self._make_cache(B, H, 5, D)
        new = self._make_cache(B, H, 1, D)

        original_data = existing.quantized_indices.clone()
        merged = _append_to_cache(existing, new)

        assert torch.equal(merged.quantized_indices[:, :, :5, :], original_data)

    def test_append_preserves_codebook_and_scales(self) -> None:
        """Merged cache should use the existing codebook and scaling factors."""
        B, H, D = 1, 2, 8
        existing = self._make_cache(B, H, 5, D)
        new = self._make_cache(B, H, 1, D)

        merged = _append_to_cache(existing, new)

        assert torch.equal(merged.codebook, existing.codebook)
        assert torch.equal(merged.scaling_factors, existing.scaling_factors)

    def test_append_multiple_tokens(self) -> None:
        """Appending multiple tokens at once should work correctly."""
        B, H, D = 1, 4, 16
        existing = self._make_cache(B, H, 20, D)
        new = self._make_cache(B, H, 5, D)

        merged = _append_to_cache(existing, new)

        assert merged.quantized_indices.shape[2] == 25
        assert merged.seq_lengths.tolist() == [25]


class TestIncrementalGeneration:
    """Test the incremental generation cache in PrefillQuantizedAttention."""

    def _make_attention(self, H: int = 4, D: int = 16) -> PrefillQuantizedAttention:
        """Create a PrefillQuantizedAttention for testing."""
        return PrefillQuantizedAttention(
            num_heads=H,
            head_dim=D,
            codebook="nuq3",
            scaling_factors=None,
            outlier_fraction=0.01,
            block_size=32,
        )

    def test_prefill_then_generate_single_token(self) -> None:
        """After prefill, generating a single token should grow the cache."""
        torch.manual_seed(42)
        H, D = 4, 16
        B, S_prefill = 1, 32
        attn = self._make_attention(H=H, D=D)

        # Prefill
        q = torch.randn(B, H, S_prefill, D, dtype=torch.float16)
        k = torch.randn(B, H, S_prefill, D, dtype=torch.float16)
        v = torch.randn(B, H, S_prefill, D, dtype=torch.float16)
        mask = torch.ones(B, S_prefill, dtype=torch.long)

        attn.forward(q, k, v, mask, is_prefill=True)

        assert attn.key_cache is not None
        assert attn.key_cache.quantized_indices.shape[2] == S_prefill

        # Generate one token
        q_gen = torch.randn(B, H, 1, D, dtype=torch.float16)
        k_gen = torch.randn(B, H, 1, D, dtype=torch.float16)
        v_gen = torch.randn(B, H, 1, D, dtype=torch.float16)
        mask_gen = torch.ones(B, S_prefill + 1, dtype=torch.long)

        output = attn.forward(q_gen, k_gen, v_gen, mask_gen, is_prefill=False)

        # Cache should have grown
        assert attn.key_cache.quantized_indices.shape[2] == S_prefill + 1
        assert attn.value_cache.quantized_indices.shape[2] == S_prefill + 1

        # Output shape should match query length
        assert output.shape == (B, H, 1, D)

    def test_multi_step_generation(self) -> None:
        """Multiple generation steps should incrementally grow the cache."""
        torch.manual_seed(42)
        H, D = 2, 8
        B, S_prefill = 1, 16
        num_gen_steps = 5
        attn = self._make_attention(H=H, D=D)

        # Prefill
        q = torch.randn(B, H, S_prefill, D, dtype=torch.float16)
        k = torch.randn(B, H, S_prefill, D, dtype=torch.float16)
        v = torch.randn(B, H, S_prefill, D, dtype=torch.float16)
        mask = torch.ones(B, S_prefill, dtype=torch.long)

        attn.forward(q, k, v, mask, is_prefill=True)

        # Generate multiple tokens
        for step in range(num_gen_steps):
            current_len = S_prefill + step + 1
            q_gen = torch.randn(B, H, 1, D, dtype=torch.float16)
            k_gen = torch.randn(B, H, 1, D, dtype=torch.float16)
            v_gen = torch.randn(B, H, 1, D, dtype=torch.float16)
            mask_gen = torch.ones(B, current_len, dtype=torch.long)

            output = attn.forward(q_gen, k_gen, v_gen, mask_gen, is_prefill=False)

            expected_cache_len = S_prefill + step + 1
            assert attn.key_cache.quantized_indices.shape[2] == expected_cache_len
            assert output.shape == (B, H, 1, D)
            assert not torch.isnan(output).any(), f"NaN at step {step}"

    def test_generation_without_prefill(self) -> None:
        """Generation without prior prefill should create a fresh cache."""
        torch.manual_seed(42)
        H, D = 4, 16
        B = 1
        attn = self._make_attention(H=H, D=D)

        assert attn.key_cache is None

        # Generate directly (no prefill)
        q = torch.randn(B, H, 1, D, dtype=torch.float16)
        k = torch.randn(B, H, 1, D, dtype=torch.float16)
        v = torch.randn(B, H, 1, D, dtype=torch.float16)
        mask = torch.ones(B, 1, dtype=torch.long)

        output = attn.forward(q, k, v, mask, is_prefill=False)

        assert attn.key_cache is not None
        assert attn.key_cache.quantized_indices.shape[2] == 1
        assert output.shape == (B, H, 1, D)

    def test_clear_and_new_prefill(self) -> None:
        """After clearing cache, a new prefill should start fresh."""
        torch.manual_seed(42)
        H, D = 2, 8
        B, S = 1, 16
        attn = self._make_attention(H=H, D=D)

        # First sequence
        q = torch.randn(B, H, S, D, dtype=torch.float16)
        k = torch.randn(B, H, S, D, dtype=torch.float16)
        v = torch.randn(B, H, S, D, dtype=torch.float16)
        mask = torch.ones(B, S, dtype=torch.long)

        attn.forward(q, k, v, mask, is_prefill=True)
        assert attn.key_cache.quantized_indices.shape[2] == S

        # Clear and start new sequence
        attn.clear_cache()
        assert attn.key_cache is None

        S2 = 8
        q2 = torch.randn(B, H, S2, D, dtype=torch.float16)
        k2 = torch.randn(B, H, S2, D, dtype=torch.float16)
        v2 = torch.randn(B, H, S2, D, dtype=torch.float16)
        mask2 = torch.ones(B, S2, dtype=torch.long)

        attn.forward(q2, k2, v2, mask2, is_prefill=True)
        assert attn.key_cache.quantized_indices.shape[2] == S2

    def test_generation_output_not_nan(self) -> None:
        """Generated tokens should not produce NaN outputs."""
        torch.manual_seed(42)
        H, D = 4, 16
        B, S = 2, 32
        attn = self._make_attention(H=H, D=D)

        q = torch.randn(B, H, S, D, dtype=torch.float16)
        k = torch.randn(B, H, S, D, dtype=torch.float16)
        v = torch.randn(B, H, S, D, dtype=torch.float16)
        mask = torch.ones(B, S, dtype=torch.long)

        attn.forward(q, k, v, mask, is_prefill=True)

        # Generate 3 tokens
        for i in range(3):
            q_gen = torch.randn(B, H, 1, D, dtype=torch.float16)
            k_gen = torch.randn(B, H, 1, D, dtype=torch.float16)
            v_gen = torch.randn(B, H, 1, D, dtype=torch.float16)
            total_len = S + i + 1
            mask_gen = torch.ones(B, total_len, dtype=torch.long)

            output = attn.forward(q_gen, k_gen, v_gen, mask_gen, is_prefill=False)
            assert not torch.isnan(output).any(), f"NaN at generation step {i}"
            assert not torch.isinf(output).any(), f"Inf at generation step {i}"

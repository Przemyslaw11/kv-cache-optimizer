"""Tests for model integration utilities (kvquant/model_utils.py).

Validates:
    - Per-layer codebook and scaling factor save/load round-trip.
    - QuantizedModelWrapper patch/unpatch lifecycle.
    - ActivationCollector layer detection and hook management.
    - _append_to_cache correctness (tested in test_generation_cache.py).

Note: Tests requiring a real HuggingFace model are marked with
``@pytest.mark.gpu`` and are skipped in CI without GPUs.
"""

import tempfile
from pathlib import Path

import pytest
import torch

from kvquant.nuq import (
    create_heuristic_codebook,
    load_per_layer_codebooks,
    load_per_layer_scaling_factors,
    save_per_layer_codebooks,
    save_per_layer_scaling_factors,
)

# =====================================================================
# Per-layer codebook save/load tests
# =====================================================================


class TestPerLayerCodebooks:
    """Tests for per-layer codebook persistence."""

    def test_save_and_load_round_trip(self) -> None:
        """Saved codebooks should be loadable and match originals."""
        codebooks = {
            0: create_heuristic_codebook("nuq3"),
            1: create_heuristic_codebook("nuq3"),
            2: create_heuristic_codebook("nuq4"),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            save_per_layer_codebooks(codebooks, tmpdir)
            loaded = load_per_layer_codebooks(tmpdir)

        assert len(loaded) == 3
        for idx in codebooks:
            assert torch.allclose(codebooks[idx], loaded[idx])

    def test_load_nonexistent_dir_raises(self) -> None:
        """Loading from nonexistent directory should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_per_layer_codebooks("/nonexistent/path")

    def test_save_creates_directory(self) -> None:
        """Saving should create the output directory if it doesn't exist."""
        codebooks = {0: create_heuristic_codebook("nuq3")}

        with tempfile.TemporaryDirectory() as tmpdir:
            nested = Path(tmpdir) / "a" / "b" / "c"
            save_per_layer_codebooks(codebooks, nested)
            assert nested.exists()
            loaded = load_per_layer_codebooks(nested)
            assert len(loaded) == 1

    def test_empty_directory_returns_empty_dict(self) -> None:
        """Loading from empty directory should return empty dict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            loaded = load_per_layer_codebooks(tmpdir)
            assert loaded == {}


# =====================================================================
# Per-layer scaling factor save/load tests
# =====================================================================


class TestPerLayerScalingFactors:
    """Tests for per-layer scaling factor persistence."""

    def test_save_and_load_round_trip(self) -> None:
        """Saved scaling factors should be loadable and match originals."""
        scales = {
            0: torch.randn(32, 128, dtype=torch.float16),
            1: torch.randn(32, 128, dtype=torch.float16),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            save_per_layer_scaling_factors(scales, tmpdir)
            loaded = load_per_layer_scaling_factors(tmpdir)

        assert len(loaded) == 2
        for idx in scales:
            assert torch.allclose(scales[idx], loaded[idx])

    def test_load_nonexistent_dir_raises(self) -> None:
        """Loading from nonexistent directory should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_per_layer_scaling_factors("/nonexistent/path")

    def test_scaling_factors_shape_preserved(self) -> None:
        """Loaded scaling factors should have the same shape as saved."""
        H, D = 8, 64
        scales = {0: torch.rand(H, D, dtype=torch.float16)}

        with tempfile.TemporaryDirectory() as tmpdir:
            save_per_layer_scaling_factors(scales, tmpdir)
            loaded = load_per_layer_scaling_factors(tmpdir)

        assert loaded[0].shape == (H, D)
        assert loaded[0].dtype == torch.float16


# =====================================================================
# QuantizedModelWrapper tests (with mock model)
# =====================================================================


class _MockConfig:
    """Minimal mock of a HuggingFace model config."""

    def __init__(self, num_heads: int = 4, hidden_size: int = 64) -> None:
        self.num_attention_heads = num_heads
        self.hidden_size = hidden_size


class _MockAttention(torch.nn.Module):
    """Minimal mock attention module."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(64, 64)  # dummy parameter

    def forward(self, hidden_states, **kwargs):
        # Return (attn_output, attn_weights, past_key_value)
        B, S, D = hidden_states.shape
        output = hidden_states
        attn_weights = None
        # Fake past_key_value
        num_heads = 4
        head_dim = D // num_heads
        key_states = hidden_states.view(B, S, num_heads, head_dim).transpose(1, 2)
        value_states = key_states.clone()
        past_kv = (key_states, value_states)
        return (output, attn_weights, past_kv)


class _MockLayer(torch.nn.Module):
    """Minimal mock transformer layer."""

    def __init__(self) -> None:
        super().__init__()
        self.self_attn = _MockAttention()


class _MockModel(torch.nn.Module):
    """Minimal mock of a Llama-style model with 2 layers."""

    def __init__(self) -> None:
        super().__init__()
        self.config = _MockConfig(num_heads=4, hidden_size=64)
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([_MockLayer(), _MockLayer()])


class TestQuantizedModelWrapper:
    """Tests for QuantizedModelWrapper patch/unpatch lifecycle."""

    def test_patch_sets_is_patched(self) -> None:
        """Patching should set is_patched to True."""
        from kvquant.model_utils import QuantizedModelWrapper

        model = _MockModel()
        wrapper = QuantizedModelWrapper(model, codebook="nuq3")

        assert not wrapper.is_patched
        wrapper.patch()
        assert wrapper.is_patched

    def test_unpatch_restores_original(self) -> None:
        """Unpatching should restore original forward methods."""
        from kvquant.model_utils import QuantizedModelWrapper

        model = _MockModel()
        # Before patching, forward should NOT be in __dict__ (uses class method)
        assert "forward" not in model.model.layers[0].self_attn.__dict__

        wrapper = QuantizedModelWrapper(model, codebook="nuq3")
        wrapper.patch()

        # Forward method should be replaced (now in instance __dict__)
        assert "forward" in model.model.layers[0].self_attn.__dict__

        wrapper.unpatch()
        assert not wrapper.is_patched

        # After unpatch, calling forward should produce valid output
        hidden = torch.randn(1, 4, 64)
        result = model.model.layers[0].self_attn(hidden)
        assert isinstance(result, tuple)
        assert len(result) == 3  # (output, weights, past_kv)

    def test_double_patch_warns(self) -> None:
        """Patching an already-patched model should warn."""
        from kvquant.model_utils import QuantizedModelWrapper

        model = _MockModel()
        wrapper = QuantizedModelWrapper(model, codebook="nuq3")
        wrapper.patch()
        # Second patch should not raise
        wrapper.patch()
        assert wrapper.is_patched
        wrapper.unpatch()

    def test_patch_with_per_layer_codebooks(self) -> None:
        """Patching with per-layer codebooks should work."""
        from kvquant.model_utils import QuantizedModelWrapper

        model = _MockModel()
        codebooks = {
            0: create_heuristic_codebook("nuq3"),
            1: create_heuristic_codebook("nuq4"),
        }
        wrapper = QuantizedModelWrapper(model, codebook=codebooks)
        wrapper.patch()
        assert wrapper.is_patched
        wrapper.unpatch()


# =====================================================================
# ActivationCollector tests (with mock model)
# =====================================================================


class TestActivationCollector:
    """Tests for the ActivationCollector hook management."""

    def test_detect_num_layers(self) -> None:
        """Should auto-detect the number of transformer layers."""
        from kvquant.model_utils import ActivationCollector

        model = _MockModel()
        collector = ActivationCollector(model)
        assert collector.num_layers == 2

    def test_install_and_remove_hooks(self) -> None:
        """Hooks should be installable and removable without error."""
        from kvquant.model_utils import ActivationCollector

        model = _MockModel()
        collector = ActivationCollector(model)
        collector.install_hooks()
        assert len(collector._hooks) == 2
        collector.remove_hooks()
        assert len(collector._hooks) == 0

    def test_clear_releases_data(self) -> None:
        """clear() should release all collected activations."""
        from kvquant.model_utils import ActivationCollector

        model = _MockModel()
        collector = ActivationCollector(model)
        collector.install_hooks()

        # Run a mock forward
        hidden = torch.randn(1, 4, 64)
        model.model.layers[0].self_attn(hidden)
        collector.increment_sample_count()

        collector.remove_hooks()
        collector.clear()
        assert collector.get_keys(0) is None
        assert collector.get_values(0) is None


# =====================================================================
# GPU integration tests (require CUDA and real model)
# =====================================================================


@pytest.mark.gpu
class TestGPUIntegration:
    """GPU integration tests with real model outputs.

    These tests require a CUDA-capable GPU and are skipped in CI
    without GPU support. They validate quantization on realistic
    activation distributions from actual model internals.
    """

    def test_quantize_on_gpu_device(self) -> None:
        """Quantization should work correctly on CUDA tensors."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        from kvquant.batched_quant import BatchedKVQuantizer

        device = "cuda"
        B, H, S, D = 2, 4, 64, 32
        keys = torch.randn(B, H, S, D, dtype=torch.float16, device=device)
        mask = torch.ones(B, S, dtype=torch.long, device=device)

        cb = create_heuristic_codebook("nuq3", device=device)
        scales = torch.ones(H, D, dtype=torch.float16, device=device)
        quantizer = BatchedKVQuantizer(cb, scales, outlier_fraction=0.01)

        cache = quantizer.quantize_keys(keys, mask)
        recovered = quantizer.dequantize(cache)

        mse = ((keys.float() - recovered.float()) ** 2).mean().item()
        assert mse < 0.1, f"GPU round-trip MSE too high: {mse}"
        assert recovered.device.type == "cuda"

    def test_prefill_attention_on_gpu(self) -> None:
        """PrefillQuantizedAttention should work on GPU."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        from kvquant.prefill_quant import PrefillQuantizedAttention

        device = "cuda"
        H, D = 4, 16
        B, S = 2, 32
        attn = PrefillQuantizedAttention(
            num_heads=H,
            head_dim=D,
            codebook="nuq3",
        )

        q = torch.randn(B, H, S, D, dtype=torch.float16, device=device)
        k = torch.randn(B, H, S, D, dtype=torch.float16, device=device)
        v = torch.randn(B, H, S, D, dtype=torch.float16, device=device)
        mask = torch.ones(B, S, dtype=torch.long, device=device)

        output = attn.forward(q, k, v, mask, is_prefill=True)

        assert output.shape == (B, H, S, D)
        assert output.device.type == "cuda"
        assert not torch.isnan(output).any()

    def test_generation_on_gpu_after_prefill(self) -> None:
        """Generation after prefill should work on GPU."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        from kvquant.prefill_quant import PrefillQuantizedAttention

        device = "cuda"
        H, D = 4, 16
        B, S = 1, 32
        attn = PrefillQuantizedAttention(
            num_heads=H,
            head_dim=D,
            codebook="nuq3",
        )

        # Prefill
        q = torch.randn(B, H, S, D, dtype=torch.float16, device=device)
        k = torch.randn(B, H, S, D, dtype=torch.float16, device=device)
        v = torch.randn(B, H, S, D, dtype=torch.float16, device=device)
        mask = torch.ones(B, S, dtype=torch.long, device=device)
        attn.forward(q, k, v, mask, is_prefill=True)

        # Generate
        q_gen = torch.randn(B, H, 1, D, dtype=torch.float16, device=device)
        k_gen = torch.randn(B, H, 1, D, dtype=torch.float16, device=device)
        v_gen = torch.randn(B, H, 1, D, dtype=torch.float16, device=device)
        mask_gen = torch.ones(B, S + 1, dtype=torch.long, device=device)

        output = attn.forward(q_gen, k_gen, v_gen, mask_gen, is_prefill=False)

        assert output.shape == (B, H, 1, D)
        assert output.device.type == "cuda"
        assert attn.key_cache.quantized_indices.shape[2] == S + 1

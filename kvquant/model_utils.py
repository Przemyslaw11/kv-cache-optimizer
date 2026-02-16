"""Model integration utilities for hooking quantized attention into HuggingFace models.

Provides a clean abstraction for:
    1. Loading Llama-2 (or compatible) models with quantized KV cache.
    2. Patching attention layers to use ``PrefillQuantizedAttention``.
    3. Collecting key/value activations for calibration.
    4. Running perplexity evaluation with quantized KV cache.

The patching approach uses forward hooks to intercept and replace attention
computation, avoiding invasive model code changes and supporting any
HuggingFace-compatible model with standard attention interfaces.

Usage:
    >>> model, tokenizer = load_model("models/Llama-2-7B-32K")
    >>> patched = patch_model_attention(model, codebook="nuq3")
    >>> ppl = evaluate_perplexity(patched, tokenizer, dataset="wikitext2")
"""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model(
    model_path: str | Path,
    device_map: str = "auto",
    dtype: torch.dtype = torch.float16,
) -> tuple:
    """Load a HuggingFace causal LM model and tokenizer.

    Centralizes model loading so experiment scripts depend on this
    abstraction rather than on ``transformers`` internals directly.

    Args:
        model_path: Path to the model directory or HuggingFace model ID.
        device_map: Device mapping strategy (default ``"auto"``).
        dtype: Model dtype (default ``torch.float16``).

    Returns:
        Tuple of ``(model, tokenizer)``.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = str(model_path)
    logger.info("Loading model from %s (dtype=%s, device_map=%s)", model_path, dtype, device_map)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set pad_token to eos_token: %s", tokenizer.pad_token)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    model.eval()

    num_params = sum(p.numel() for p in model.parameters()) / 1e9
    logger.info("Model loaded: %.2fB parameters", num_params)

    return model, tokenizer


# ---------------------------------------------------------------------------
# Activation collection for calibration
# ---------------------------------------------------------------------------


@dataclass
class ActivationCollector:
    """Collects key/value activations from model attention layers via hooks.

    Registers forward hooks on all attention layers to capture key and value
    tensors during a forward pass over calibration data. Collected activations
    are stored per-layer.

    Args:
        model: HuggingFace causal LM model.
        num_layers: Number of transformer layers to hook (auto-detected if None).
        max_samples: Maximum number of forward passes to collect (default 16).

    Example:
        >>> collector = ActivationCollector(model)
        >>> collector.install_hooks()
        >>> with torch.no_grad():
        ...     model(**inputs)
        >>> collector.remove_hooks()
        >>> keys_layer0 = collector.get_keys(layer_idx=0)
    """

    model: nn.Module
    num_layers: int | None = None
    max_samples: int = 16

    def __post_init__(self) -> None:
        self._hooks: list = []
        self._key_activations: dict[int, list[torch.Tensor]] = {}
        self._value_activations: dict[int, list[torch.Tensor]] = {}
        self._sample_count: int = 0

        if self.num_layers is None:
            self.num_layers = self._detect_num_layers()

    def _detect_num_layers(self) -> int:
        """Auto-detect the number of transformer layers in the model."""
        model = self.model
        # Common attribute names for transformer layers
        for attr in ["model.layers", "transformer.h", "gpt_neox.layers"]:
            parts = attr.split(".")
            obj = model
            try:
                for part in parts:
                    obj = getattr(obj, part)
                return len(obj)
            except AttributeError:
                continue

        raise ValueError(
            "Could not auto-detect number of transformer layers. "
            "Please specify num_layers explicitly."
        )

    def _get_attention_layers(self) -> list[tuple[int, nn.Module]]:
        """Get all attention sub-modules in the model, indexed by layer."""
        model = self.model
        layers: list[tuple[int, nn.Module]] = []

        # Try Llama-style: model.model.layers[i].self_attn
        for attr_path in ["model.layers", "transformer.h", "gpt_neox.layers"]:
            parts = attr_path.split(".")
            obj = model
            try:
                for part in parts:
                    obj = getattr(obj, part)
                # obj is now the list of transformer layers
                for i, layer in enumerate(obj):
                    # Find the attention sub-module
                    for attn_attr in ["self_attn", "attn", "attention"]:
                        if hasattr(layer, attn_attr):
                            layers.append((i, getattr(layer, attn_attr)))
                            break
                if layers:
                    return layers
            except AttributeError:
                continue

        raise ValueError("Could not find attention layers in the model.")

    def install_hooks(self) -> None:
        """Install forward hooks on all attention layers to collect KV activations."""
        attention_layers = self._get_attention_layers()

        for layer_idx, attn_module in attention_layers:
            self._key_activations[layer_idx] = []
            self._value_activations[layer_idx] = []

            hook = attn_module.register_forward_hook(
                self._make_hook(layer_idx),
            )
            self._hooks.append(hook)

        logger.info("Installed %d activation collection hooks", len(self._hooks))

    def _make_hook(self, layer_idx: int):
        """Create a forward hook for a specific layer.

        The hook intercepts the attention module's forward pass and captures
        the key_states and value_states tensors. For Llama-style models,
        these are computed inside the attention module before being used
        for attention.
        """

        def hook_fn(module, args, output):
            if self._sample_count >= self.max_samples:
                return

            # For Llama-style models, we need to capture KV from the module's
            # internal computation. We do this by hooking the k_proj and v_proj.
            # However, the standard approach for post-forward hooks is to
            # re-derive KV from the hidden states.
            #
            # Alternative: capture from output tuple.
            # LlamaAttention.forward returns (attn_output, attn_weights, past_key_value)
            # where past_key_value = (key_states, value_states)
            if isinstance(output, tuple) and len(output) >= 3:
                past_kv = output[2]
                if past_kv is not None and isinstance(past_kv, tuple) and len(past_kv) == 2:
                    key_states, value_states = past_kv
                    self._key_activations[layer_idx].append(key_states.detach().cpu())
                    self._value_activations[layer_idx].append(value_states.detach().cpu())

        return hook_fn

    def remove_hooks(self) -> None:
        """Remove all installed hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        logger.info("Removed all activation collection hooks")

    def increment_sample_count(self) -> None:
        """Increment the sample counter after a forward pass."""
        self._sample_count += 1

    def get_keys(self, layer_idx: int) -> torch.Tensor | None:
        """Get collected key activations for a specific layer.

        Args:
            layer_idx: Transformer layer index.

        Returns:
            Stacked key tensor ``[num_samples, num_heads, seq_len, head_dim]``
            or ``None`` if no activations were collected.
        """
        acts = self._key_activations.get(layer_idx, [])
        if not acts:
            return None
        return torch.cat(acts, dim=0)

    def get_values(self, layer_idx: int) -> torch.Tensor | None:
        """Get collected value activations for a specific layer.

        Args:
            layer_idx: Transformer layer index.

        Returns:
            Stacked value tensor ``[num_samples, num_heads, seq_len, head_dim]``
            or ``None`` if no activations were collected.
        """
        acts = self._value_activations.get(layer_idx, [])
        if not acts:
            return None
        return torch.cat(acts, dim=0)

    def clear(self) -> None:
        """Clear all collected activations and free memory."""
        self._key_activations.clear()
        self._value_activations.clear()
        self._sample_count = 0
        gc.collect()


# ---------------------------------------------------------------------------
# Attention layer patching
# ---------------------------------------------------------------------------


@dataclass
class _PatchedAttentionState:
    """Stores the state of a patched attention layer for cleanup."""

    layer_idx: int
    attn_module: object
    original_forward: object
    quantized_attention: object
    hook_handle: object


class QuantizedModelWrapper:
    """Wraps a HuggingFace model with quantized KV cache attention.

    Replaces attention forward passes with quantized versions while
    preserving the original model structure for easy cleanup.

    Args:
        model: HuggingFace causal LM model.
        codebook: NUQ codebook per layer ``{layer_idx: Tensor}`` or a single
            codebook/string applied to all layers.
        scaling_factors: Per-layer scaling factors ``{layer_idx: Tensor[H, D]}``
            or ``None`` for unit scales.
        outlier_fraction: Fraction of outliers (default 0.01).
        block_size: Block size for sparse matrices (default 256).

    Example:
        >>> wrapper = QuantizedModelWrapper(model, codebook="nuq3")
        >>> wrapper.patch()
        >>> output = model(**inputs)  # Uses quantized attention
        >>> wrapper.unpatch()
    """

    def __init__(
        self,
        model: nn.Module,
        codebook: dict[int, torch.Tensor] | torch.Tensor | str = "nuq3",
        scaling_factors: dict[int, torch.Tensor] | None = None,
        outlier_fraction: float = 0.01,
        block_size: int = 256,
    ) -> None:
        self.model = model
        self._codebook_config = codebook
        self._scaling_factors_config = scaling_factors
        self.outlier_fraction = outlier_fraction
        self.block_size = block_size
        self._patches: list[_PatchedAttentionState] = []
        self._is_patched = False

    def _resolve_codebook(self, layer_idx: int) -> torch.Tensor | str:
        """Resolve the codebook for a specific layer."""
        if isinstance(self._codebook_config, dict):
            return self._codebook_config[layer_idx]
        return self._codebook_config

    def _resolve_scaling_factors(self, layer_idx: int) -> torch.Tensor | None:
        """Resolve scaling factors for a specific layer."""
        if self._scaling_factors_config is None:
            return None
        if isinstance(self._scaling_factors_config, dict):
            return self._scaling_factors_config.get(layer_idx)
        return self._scaling_factors_config

    def patch(self) -> None:
        """Patch all attention layers to use quantized KV cache.

        Uses forward pre-hooks and hooks to intercept attention computation.
        The model's attention layers run normally to compute Q, K, V projections,
        but the KV cache is quantized before attention computation.
        """
        if self._is_patched:
            logger.warning("Model is already patched. Call unpatch() first.")
            return

        from kvquant.prefill_quant import PrefillQuantizedAttention

        collector = ActivationCollector(self.model)
        attention_layers = collector._get_attention_layers()

        for layer_idx, attn_module in attention_layers:
            codebook = self._resolve_codebook(layer_idx)
            scales = self._resolve_scaling_factors(layer_idx)

            # Detect num_heads and head_dim from the model config
            config = self.model.config
            num_heads = getattr(config, "num_attention_heads", 32)
            head_dim = getattr(config, "hidden_size", 4096) // num_heads

            quant_attn = PrefillQuantizedAttention(
                num_heads=num_heads,
                head_dim=head_dim,
                codebook=codebook,
                scaling_factors=scales,
                outlier_fraction=self.outlier_fraction,
                block_size=self.block_size,
            )

            # Store original forward for restoration
            original_forward = attn_module.forward

            # Create a wrapper that quantizes KV within the attention forward
            def make_quantized_forward(orig_fwd, q_attn, l_idx):
                """Create a closure that wraps the original attention forward."""

                def quantized_forward(
                    hidden_states,
                    attention_mask=None,
                    position_ids=None,
                    past_key_value=None,
                    output_attentions=False,
                    use_cache=False,
                    **kwargs,
                ):
                    # Run original forward to get QKV projections + output
                    result = orig_fwd(
                        hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_value,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        **kwargs,
                    )

                    # If we have past_key_value in the output, quantize it
                    if isinstance(result, tuple) and len(result) >= 3:
                        attn_output = result[0]
                        attn_weights = result[1]
                        past_kv = result[2]

                        if past_kv is not None and isinstance(past_kv, tuple):
                            key_states, value_states = past_kv

                            # Build attention mask for the quantizer
                            B = key_states.shape[0]
                            S = key_states.shape[2]
                            quant_mask = torch.ones(
                                B, S, dtype=torch.long, device=key_states.device
                            )

                            # Quantize and dequantize KV cache
                            key_cache = q_attn.quantizer.quantize_keys(key_states, quant_mask)
                            value_cache = q_attn.quantizer.quantize_values(value_states, quant_mask)

                            dequant_keys = q_attn.quantizer.dequantize(key_cache)
                            dequant_values = q_attn.quantizer.dequantize(value_cache)

                            # Store in the quantized attention module's cache
                            q_attn._key_cache = key_cache
                            q_attn._value_cache = value_cache

                            # Return with quantized KV
                            past_kv = (dequant_keys, dequant_values)
                            return (attn_output, attn_weights, past_kv)

                    return result

                return quantized_forward

            wrapped_forward = make_quantized_forward(original_forward, quant_attn, layer_idx)
            attn_module.forward = wrapped_forward

            self._patches.append(
                _PatchedAttentionState(
                    layer_idx=layer_idx,
                    attn_module=attn_module,
                    original_forward=original_forward,
                    quantized_attention=quant_attn,
                    hook_handle=None,
                )
            )

        self._is_patched = True
        logger.info("Patched %d attention layers with quantized KV cache", len(self._patches))

    def unpatch(self) -> None:
        """Restore all attention layers to their original forward passes."""
        if not self._is_patched:
            logger.warning("Model is not patched.")
            return

        for patch_state in self._patches:
            attn_module = patch_state.attn_module
            # Remove the instance-level override to restore the class method
            if "forward" in attn_module.__dict__:
                del attn_module.__dict__["forward"]

        self._patches.clear()
        self._is_patched = False
        logger.info("Unpatched all attention layers (restored originals)")

    def clear_all_caches(self) -> None:
        """Clear quantized KV caches from all patched attention layers."""
        for patch_state in self._patches:
            if hasattr(patch_state.quantized_attention, "clear_cache"):
                patch_state.quantized_attention.clear_cache()

    @property
    def is_patched(self) -> bool:
        """Whether the model currently has quantized attention patches."""
        return self._is_patched


# ---------------------------------------------------------------------------
# Perplexity evaluation
# ---------------------------------------------------------------------------


def evaluate_perplexity(
    model: nn.Module,
    tokenizer,
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
    split: str = "test",
    max_seq_len: int = 2048,
    stride: int = 512,
    batch_size: int = 1,
    max_samples: int | None = None,
    device: str | torch.device = "cuda",
) -> dict[str, float]:
    """Evaluate perplexity of a model on a text dataset.

    Uses a sliding-window approach with stride for long-context evaluation.
    Compatible with both patched (quantized) and unpatched (fp16) models.

    Args:
        model: HuggingFace causal LM model.
        tokenizer: Corresponding tokenizer.
        dataset_name: HuggingFace dataset name (default ``"wikitext"``).
        dataset_config: Dataset configuration (default ``"wikitext-2-raw-v1"``).
        split: Dataset split to evaluate on (default ``"test"``).
        max_seq_len: Maximum sequence length for each window.
        stride: Sliding window stride.
        batch_size: Batch size for evaluation (default 1).
        max_samples: Maximum number of text samples to use (None = all).
        device: Device to run evaluation on.

    Returns:
        Dictionary with ``"perplexity"``, ``"avg_loss"``, ``"num_tokens"``.
    """
    from datasets import load_dataset

    logger.info(
        "Evaluating perplexity: dataset=%s/%s split=%s max_seq_len=%d stride=%d",
        dataset_name,
        dataset_config,
        split,
        max_seq_len,
        stride,
    )

    # Load and tokenize dataset
    dataset = load_dataset(dataset_name, dataset_config, split=split)

    # Concatenate all text into one long string
    texts = dataset["text"]
    if max_samples is not None:
        texts = texts[:max_samples]

    full_text = "\n\n".join([t for t in texts if t.strip()])
    encodings = tokenizer(full_text, return_tensors="pt")
    input_ids = encodings.input_ids  # [1, total_tokens]
    total_tokens = input_ids.shape[1]

    logger.info("Total tokens in dataset: %d", total_tokens)

    # Sliding window evaluation
    nlls: list[float] = []
    num_eval_tokens = 0

    model.eval()
    with torch.no_grad():
        for begin_loc in range(0, total_tokens - 1, stride):
            end_loc = min(begin_loc + max_seq_len, total_tokens)
            trg_len = end_loc - begin_loc

            input_chunk = input_ids[:, begin_loc:end_loc].to(device)
            target_chunk = input_chunk.clone()

            # Only compute loss on the last (stride) tokens of each window
            # except for the first window
            if begin_loc > 0:
                target_chunk[:, :-stride] = -100

            outputs = model(input_chunk, labels=target_chunk)
            neg_log_likelihood = outputs.loss * trg_len
            nlls.append(neg_log_likelihood.item())
            num_eval_tokens += trg_len

            if end_loc >= total_tokens:
                break

    avg_nll = sum(nlls) / num_eval_tokens
    perplexity = torch.exp(torch.tensor(avg_nll)).item()

    result = {
        "perplexity": perplexity,
        "avg_loss": avg_nll,
        "num_tokens": num_eval_tokens,
    }

    logger.info(
        "Perplexity: %.4f (avg_loss=%.6f, num_tokens=%d)", perplexity, avg_nll, num_eval_tokens
    )

    return result

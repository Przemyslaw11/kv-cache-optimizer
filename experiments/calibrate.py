"""Calibration pipeline: compute NUQ codebooks and scaling factors from real activations.

Runs calibration on Wikitext-2 (or custom calibration data) by collecting
key/value activations from a Llama-2 (or compatible) model, then computing:
    1. Per-layer NUQ codebooks via K-means on key activations.
    2. Per-layer per-channel scaling factors via mean absolute value.

The calibrated artifacts are saved to disk and can be loaded by
``QuantizedModelWrapper`` for inference with quantized KV cache.

Expected runtime: ~30min on 1x A100-40GB (16 calibration samples, 2048 tokens).
Required GPU resources: 1x A100-40GB.
Output files:
    - results/calibration/<model_name>/codebook_layer<N>.pt
    - results/calibration/<model_name>/scales_layer<N>.pt
    - results/calibration/<model_name>/calibration_summary.json
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import time
from pathlib import Path

import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kvquant.kernels.pytorch_fused import estimate_scaling_factors
from kvquant.model_utils import ActivationCollector, load_model
from kvquant.nuq import (
    compute_nuq_codebook,
    save_per_layer_codebooks,
    save_per_layer_scaling_factors,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/calibrate.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)


def load_calibration_data(
    tokenizer,
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
    split: str = "train",
    num_samples: int = 16,
    max_seq_len: int = 2048,
    seed: int = 42,
) -> list[dict]:
    """Load and tokenize calibration samples from a dataset.

    Args:
        tokenizer: HuggingFace tokenizer.
        dataset_name: Dataset name on HuggingFace Hub.
        dataset_config: Dataset configuration.
        split: Dataset split (default ``"train"``).
        num_samples: Number of calibration samples to extract.
        max_seq_len: Maximum sequence length per sample.
        seed: Random seed for reproducibility.

    Returns:
        List of tokenized input dictionaries ready for model forward.
    """
    from datasets import load_dataset

    logger.info(
        "Loading calibration data: %s/%s split=%s num_samples=%d max_seq_len=%d",
        dataset_name,
        dataset_config,
        split,
        num_samples,
        max_seq_len,
    )

    dataset = load_dataset(dataset_name, dataset_config, split=split)

    # Concatenate text and split into chunks
    texts = [t for t in dataset["text"] if t.strip()]
    full_text = "\n\n".join(texts)

    encodings = tokenizer(full_text, return_tensors="pt")
    total_tokens = encodings.input_ids.shape[1]

    logger.info("Total calibration tokens available: %d", total_tokens)

    # Extract non-overlapping chunks
    torch.manual_seed(seed)
    samples = []
    for i in range(num_samples):
        start = i * max_seq_len
        end = start + max_seq_len
        if end > total_tokens:
            break

        chunk_ids = encodings.input_ids[:, start:end]
        chunk_mask = torch.ones_like(chunk_ids)
        samples.append(
            {
                "input_ids": chunk_ids,
                "attention_mask": chunk_mask,
            }
        )

    logger.info("Extracted %d calibration samples of length %d", len(samples), max_seq_len)
    return samples


def run_calibration(
    model_path: str,
    output_dir: str,
    nuq_datatype: str = "nuq3",
    num_samples: int = 16,
    max_seq_len: int = 2048,
    kmeans_iterations: int = 100,
    device: str = "cuda",
    seed: int = 42,
) -> dict:
    """Run the full calibration pipeline.

    Args:
        model_path: Path to HuggingFace model.
        output_dir: Directory to save calibration artifacts.
        nuq_datatype: Quantization datatype (default ``"nuq3"``).
        num_samples: Number of calibration samples.
        max_seq_len: Maximum sequence length per sample.
        kmeans_iterations: K-means iterations for codebook computation.
        device: Device for computation.
        seed: Random seed.

    Returns:
        Summary dictionary with calibration metadata.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    start_time = time.time()

    # Load model and tokenizer
    model, tokenizer = load_model(model_path, device_map="auto")

    # Load calibration data
    samples = load_calibration_data(
        tokenizer,
        num_samples=num_samples,
        max_seq_len=max_seq_len,
        seed=seed,
    )

    # Set up activation collector
    collector = ActivationCollector(model, max_samples=num_samples)
    collector.install_hooks()

    # Run forward passes to collect activations
    logger.info("Running %d forward passes for activation collection...", len(samples))
    model.eval()
    with torch.no_grad():
        for i, sample in enumerate(samples):
            inputs = {k: v.to(device) for k, v in sample.items()}

            try:
                _ = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    use_cache=True,
                    output_attentions=False,
                )
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.warning("OOM on sample %d, skipping (clearing cache)", i)
                    torch.cuda.empty_cache()
                    gc.collect()
                    continue
                raise

            collector.increment_sample_count()
            logger.info("Collected activations for sample %d/%d", i + 1, len(samples))

            # Free GPU memory
            del inputs
            torch.cuda.empty_cache()

    collector.remove_hooks()

    # Compute per-layer codebooks and scaling factors
    num_layers = collector.num_layers
    codebooks: dict[int, torch.Tensor] = {}
    scaling_factors: dict[int, torch.Tensor] = {}
    layer_stats: dict[int, dict] = {}

    logger.info("Computing codebooks and scaling factors for %d layers...", num_layers)

    for layer_idx in range(num_layers):
        keys = collector.get_keys(layer_idx)
        if keys is None:
            logger.warning("No key activations collected for layer %d, using heuristic", layer_idx)
            from kvquant.nuq import create_heuristic_codebook

            codebooks[layer_idx] = create_heuristic_codebook(nuq_datatype, device="cpu")
            continue

        logger.info(
            "Layer %d: key activations shape %s",
            layer_idx,
            tuple(keys.shape),
        )

        # Compute scaling factors: [num_heads, head_dim]
        # keys shape: [num_collected_samples, num_heads, seq_len, head_dim]
        scales = estimate_scaling_factors(keys.to(dtype=torch.float16))
        scaling_factors[layer_idx] = scales

        # Compute codebook from flattened key activations
        flat_keys = keys.flatten().to(dtype=torch.float32)
        # Subsample if too many values (>1M) for K-means efficiency
        if flat_keys.numel() > 1_000_000:
            perm = torch.randperm(flat_keys.numel())[:1_000_000]
            flat_keys = flat_keys[perm]

        codebook = compute_nuq_codebook(
            flat_keys,
            nuq_datatype,
            num_iterations=kmeans_iterations,
            device="cpu",
        )
        codebooks[layer_idx] = codebook

        layer_stats[layer_idx] = {
            "key_shape": list(keys.shape),
            "scale_mean": scales.float().mean().item(),
            "scale_std": scales.float().std().item(),
            "codebook_range": [codebook.min().item(), codebook.max().item()],
        }

        logger.info(
            "Layer %d: scales mean=%.4f, codebook range=[%.4f, %.4f]",
            layer_idx,
            scales.float().mean().item(),
            codebook.min().item(),
            codebook.max().item(),
        )

    # Save artifacts
    output_path = Path(output_dir)
    save_per_layer_codebooks(codebooks, output_path)
    save_per_layer_scaling_factors(scaling_factors, output_path)

    # Save summary
    elapsed = time.time() - start_time
    summary = {
        "model_path": model_path,
        "nuq_datatype": nuq_datatype,
        "num_samples": len(samples),
        "max_seq_len": max_seq_len,
        "kmeans_iterations": kmeans_iterations,
        "num_layers_calibrated": len(codebooks),
        "seed": seed,
        "elapsed_sec": elapsed,
        "layer_stats": {str(k): v for k, v in layer_stats.items()},
    }

    summary_path = output_path / "calibration_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(
        "Calibration complete: %d layers in %.1f sec. Artifacts saved to %s",
        len(codebooks),
        elapsed,
        output_path,
    )

    # Cleanup
    collector.clear()
    del model
    gc.collect()
    torch.cuda.empty_cache()

    return summary


def main() -> None:
    """CLI entry point for calibration."""
    parser = argparse.ArgumentParser(
        description="Run NUQ calibration on a HuggingFace model using Wikitext-2",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/Llama-2-7B-32K",
        help="Path to HuggingFace model directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for calibration artifacts "
        "(default: results/calibration/<model_name>)",
    )
    parser.add_argument(
        "--nuq_datatype",
        type=str,
        default="nuq3",
        choices=["nuq2", "nuq3", "nuq4"],
        help="NUQ quantization datatype (default: nuq3)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=16,
        help="Number of calibration samples (default: 16)",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=2048,
        help="Maximum sequence length per sample (default: 2048)",
    )
    parser.add_argument(
        "--kmeans_iterations",
        type=int,
        default=100,
        help="K-means iterations for codebook computation (default: 100)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )

    args = parser.parse_args()

    # Default output dir based on model name
    if args.output_dir is None:
        model_name = Path(args.model_path).name
        args.output_dir = f"results/calibration/{model_name}"

    # Ensure log directory exists
    Path("logs").mkdir(exist_ok=True)

    summary = run_calibration(
        model_path=args.model_path,
        output_dir=args.output_dir,
        nuq_datatype=args.nuq_datatype,
        num_samples=args.num_samples,
        max_seq_len=args.max_seq_len,
        kmeans_iterations=args.kmeans_iterations,
        seed=args.seed,
    )

    print("\n" + "=" * 60)
    print("CALIBRATION SUMMARY")
    print("=" * 60)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

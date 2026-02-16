"""Perplexity validation: compare quantized vs. fp16 KV cache on Wikitext-2.

Measures perplexity degradation introduced by NUQ quantization of the KV
cache during prefill. This is the primary accuracy validation for Phase 2.

Success criteria:
    - PPL degradation < 0.2 over fp16 baseline on Wikitext-2 test set.

Expected runtime: ~1h on 1x A100-40GB (with pre-computed calibration).
Required GPU resources: 1x A100-40GB.
Output files: results/ppl_validation_<config>.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kvquant.model_utils import (
    QuantizedModelWrapper,
    evaluate_perplexity,
    load_model,
)
from kvquant.nuq import (
    load_per_layer_codebooks,
    load_per_layer_scaling_factors,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/ppl_validation.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)


def run_ppl_validation(
    model_path: str,
    calibration_dir: str | None = None,
    nuq_datatype: str = "nuq3",
    outlier_fraction: float = 0.01,
    block_size: int = 256,
    max_seq_len: int = 2048,
    stride: int = 512,
    max_samples: int | None = None,
    seed: int = 42,
) -> dict:
    """Run perplexity validation comparing fp16 baseline vs quantized.

    Args:
        model_path: Path to HuggingFace model directory.
        calibration_dir: Directory with calibrated codebooks/scales.
            If ``None``, uses heuristic codebook (no calibration).
        nuq_datatype: NUQ datatype (default ``"nuq3"``).
        outlier_fraction: Outlier fraction for quantization.
        block_size: Block size for sparse matrices.
        max_seq_len: Maximum sequence length for perplexity evaluation.
        stride: Sliding window stride.
        max_samples: Maximum number of text samples (None = all).
        seed: Random seed.

    Returns:
        Dictionary with fp16 and quantized perplexity results.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    start_time = time.time()

    # Load model
    model, tokenizer = load_model(model_path, device_map="auto")

    # ---- Step 1: FP16 baseline perplexity ----
    logger.info("=" * 60)
    logger.info("Step 1: Evaluating FP16 baseline perplexity")
    logger.info("=" * 60)

    fp16_result = evaluate_perplexity(
        model,
        tokenizer,
        max_seq_len=max_seq_len,
        stride=stride,
        max_samples=max_samples,
    )
    fp16_ppl = fp16_result["perplexity"]
    logger.info("FP16 baseline PPL: %.4f", fp16_ppl)

    # ---- Step 2: Quantized perplexity ----
    logger.info("=" * 60)
    logger.info("Step 2: Evaluating quantized perplexity")
    logger.info("=" * 60)

    # Load calibration artifacts if available
    codebook_config: dict[int, torch.Tensor] | str
    scales_config: dict[int, torch.Tensor] | None

    if calibration_dir is not None and Path(calibration_dir).exists():
        logger.info("Loading calibrated codebooks from %s", calibration_dir)
        codebook_config = load_per_layer_codebooks(calibration_dir, device="cpu")
        try:
            scales_config = load_per_layer_scaling_factors(calibration_dir, device="cpu")
        except FileNotFoundError:
            logger.warning("No scaling factors found, using unit scales")
            scales_config = None
    else:
        logger.info("No calibration dir provided, using heuristic %s codebook", nuq_datatype)
        codebook_config = nuq_datatype
        scales_config = None

    # Patch model with quantized attention
    wrapper = QuantizedModelWrapper(
        model,
        codebook=codebook_config,
        scaling_factors=scales_config,
        outlier_fraction=outlier_fraction,
        block_size=block_size,
    )
    wrapper.patch()

    quant_result = evaluate_perplexity(
        model,
        tokenizer,
        max_seq_len=max_seq_len,
        stride=stride,
        max_samples=max_samples,
    )
    quant_ppl = quant_result["perplexity"]
    logger.info("Quantized PPL: %.4f", quant_ppl)

    # Unpatch model
    wrapper.unpatch()

    # ---- Results ----
    ppl_degradation = quant_ppl - fp16_ppl
    elapsed = time.time() - start_time

    results = {
        "model_path": model_path,
        "nuq_datatype": nuq_datatype,
        "outlier_fraction": outlier_fraction,
        "block_size": block_size,
        "max_seq_len": max_seq_len,
        "stride": stride,
        "seed": seed,
        "calibration_dir": calibration_dir,
        "fp16": fp16_result,
        "quantized": quant_result,
        "ppl_degradation": ppl_degradation,
        "passes_threshold": abs(ppl_degradation) < 0.2,
        "elapsed_sec": elapsed,
    }

    logger.info("=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)
    logger.info("FP16 PPL:          %.4f", fp16_ppl)
    logger.info("Quantized PPL:     %.4f", quant_ppl)
    logger.info("Degradation:       %.4f", ppl_degradation)
    logger.info("Passes (<0.2):     %s", results["passes_threshold"])
    logger.info("Elapsed:           %.1f sec", elapsed)

    return results


def main() -> None:
    """CLI entry point for perplexity validation."""
    parser = argparse.ArgumentParser(
        description="Compare fp16 vs quantized KV cache perplexity on Wikitext-2",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/Llama-2-7B-32K",
        help="Path to HuggingFace model directory",
    )
    parser.add_argument(
        "--calibration_dir",
        type=str,
        default=None,
        help="Directory with calibrated codebooks/scales (default: None = heuristic codebook)",
    )
    parser.add_argument(
        "--nuq_datatype",
        type=str,
        default="nuq3",
        choices=["nuq2", "nuq3", "nuq4"],
        help="NUQ datatype (default: nuq3)",
    )
    parser.add_argument(
        "--outlier_fraction",
        type=float,
        default=0.01,
        help="Outlier fraction (default: 0.01)",
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=256,
        help="Block size for sparse matrices (default: 256)",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=2048,
        help="Max sequence length for PPL evaluation (default: 2048)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=512,
        help="Sliding window stride (default: 512)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Max text samples from dataset (default: all)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path (default: results/ppl_validation_<config>.json)",
    )

    args = parser.parse_args()

    # Ensure directories exist
    Path("logs").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)

    results = run_ppl_validation(
        model_path=args.model_path,
        calibration_dir=args.calibration_dir,
        nuq_datatype=args.nuq_datatype,
        outlier_fraction=args.outlier_fraction,
        block_size=args.block_size,
        max_seq_len=args.max_seq_len,
        stride=args.stride,
        max_samples=args.max_samples,
        seed=args.seed,
    )

    # Save results
    if args.output is None:
        config_str = (
            f"{args.nuq_datatype}_outlier{args.outlier_fraction}"
            f"_block{args.block_size}_len{args.max_seq_len}"
        )
        output_path = f"results/ppl_validation_{config_str}.json"
    else:
        output_path = args.output

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("Results saved to %s", output_path)

    print("\n" + "=" * 60)
    print("PERPLEXITY VALIDATION SUMMARY")
    print("=" * 60)
    print(f"FP16 PPL:          {results['fp16']['perplexity']:.4f}")
    print(f"Quantized PPL:     {results['quantized']['perplexity']:.4f}")
    print(f"Degradation:       {results['ppl_degradation']:.4f}")
    print(f"Passes (<0.2):     {results['passes_threshold']}")


if __name__ == "__main__":
    main()

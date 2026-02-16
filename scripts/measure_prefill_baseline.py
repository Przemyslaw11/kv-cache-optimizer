"""Measure fp16 prefill latency baseline across prompt lengths and batch sizes.

This script establishes the baseline measurements that all quantized
configurations will be compared against. Results are saved to
results/baseline_prefill_results.json.

Expected runtime: ~2h on 1× A100-40GB.
Required GPU resources: 1× A100-40GB.
Output files: results/baseline_prefill_results.json
"""

from __future__ import annotations

import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def measure_prefill_latency(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt_lengths: list[int],
    batch_sizes: list[int],
    num_warmup: int = 5,
    num_measure: int = 20,
) -> dict:
    """Measure fp16 prefill latency for various configurations.

    Args:
        model: HuggingFace causal LM model on GPU.
        tokenizer: Corresponding tokenizer.
        prompt_lengths: List of prompt token counts to benchmark.
        batch_sizes: List of batch sizes to benchmark.
        num_warmup: Number of warmup iterations before measurement.
        num_measure: Number of measurement iterations.

    Returns:
        Dictionary mapping config strings to measurement results.
    """
    results = {}

    for prompt_len in prompt_lengths:
        for batch_size in batch_sizes:
            config_name = f"len{prompt_len}_batch{batch_size}"
            print(f"\n{'='*60}")
            print(f"Measuring: {config_name}")
            print(f"{'='*60}")

            # Generate dummy input
            dummy_text = "Hello world " * (prompt_len // 2)
            inputs = tokenizer(
                [dummy_text] * batch_size,
                return_tensors="pt",
                max_length=prompt_len,
                truncation=True,
                padding="max_length",
            ).to(model.device)

            actual_len = inputs["input_ids"].shape[1]
            total_tokens = actual_len * batch_size

            try:
                # Warmup
                print(f"  Warmup ({num_warmup} iters)...")
                for _ in range(num_warmup):
                    with torch.no_grad():
                        _ = model(**inputs, use_cache=True)
                    torch.cuda.synchronize()

                # Clear cache
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

                # Measure
                print(f"  Measuring ({num_measure} iters)...")
                latencies = []
                for _ in range(num_measure):
                    torch.cuda.synchronize()
                    start = time.perf_counter()

                    with torch.no_grad():
                        _ = model(**inputs, use_cache=True)

                    torch.cuda.synchronize()
                    end = time.perf_counter()
                    latencies.append(end - start)

                peak_mem_gb = torch.cuda.max_memory_allocated() / (1024**3)

                avg_latency = sum(latencies) / len(latencies)
                throughput = total_tokens / avg_latency

                results[config_name] = {
                    "prompt_length": actual_len,
                    "batch_size": batch_size,
                    "total_tokens": total_tokens,
                    "avg_latency_sec": avg_latency,
                    "min_latency_sec": min(latencies),
                    "max_latency_sec": max(latencies),
                    "throughput_tokens_per_sec": throughput,
                    "peak_memory_gb": peak_mem_gb,
                    "status": "success",
                }

                print(f"  Latency: {avg_latency:.4f}s | "
                      f"Throughput: {throughput:.0f} tok/s | "
                      f"Memory: {peak_mem_gb:.2f} GB")

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"  OOM for {config_name} — skipping")
                    torch.cuda.empty_cache()
                    results[config_name] = {
                        "prompt_length": actual_len,
                        "batch_size": batch_size,
                        "status": "OOM",
                        "error": str(e),
                    }
                else:
                    raise

    return results


def main() -> None:
    """Run baseline prefill measurements."""
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    model_path = os.environ.get(
        "MODEL_PATH", "./models/Llama-2-7B-32K"
    )

    print(f"Loading model from: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Model loaded. Device: {next(model.parameters()).device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")

    prompt_lengths = [512, 2048, 8192, 16384, 32768]
    batch_sizes = [1, 4, 16, 32]

    results = measure_prefill_latency(model, tokenizer, prompt_lengths, batch_sizes)

    # Save results
    os.makedirs("results", exist_ok=True)
    output_path = "results/baseline_prefill_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for config, data in results.items():
        if data["status"] == "success":
            print(f"  {config:25s} | {data['throughput_tokens_per_sec']:>10.0f} tok/s | "
                  f"{data['peak_memory_gb']:>6.2f} GB")
        else:
            print(f"  {config:25s} | {data['status']}")


if __name__ == "__main__":
    main()

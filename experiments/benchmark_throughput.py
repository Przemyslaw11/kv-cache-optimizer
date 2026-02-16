"""Throughput benchmark sweep: prompt_len × batch_size.

Measures: Latency (sec), throughput (tokens/sec), peak memory (GB).
Sweeps prompt lengths [512, 2048, 8192, 16384, 32768] × batch sizes [1, 4, 16, 32, 64].
Expected runtime: ~24h on 8× A100-40GB.
Required GPU resources: 8× A100-40GB (plgrid-gpu-a100).
Output files: results/throughput_<config>.json
"""

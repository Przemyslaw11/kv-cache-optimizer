"""Throughput benchmark sweep: prompt_len x batch_size.

Measures: Latency (sec), throughput (tokens/sec), peak memory (GB).
Sweeps prompt lengths [512, 2048, 8192, 16384, 32768] x batch sizes [1, 4, 16, 32, 64].
Expected runtime: ~24h on 8x A100-40GB.
Required GPU resources: 8x A100-40GB (plgrid-gpu-a100).
Output files: results/throughput_<config>.json
"""

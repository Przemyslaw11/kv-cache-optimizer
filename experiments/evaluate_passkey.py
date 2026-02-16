"""Passkey retrieval evaluation across context lengths.

Measures: Passkey retrieval accuracy at 2K, 4K, 8K, 16K, 32K context lengths.
Expected runtime: ~4h on 8x A100-40GB.
Required GPU resources: 8x A100-40GB (plgrid-gpu-a100).
Output files: results/passkey_<config>.json
"""

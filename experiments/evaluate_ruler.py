"""RULER evaluation (13 tasks) for long-context understanding.

Measures: Accuracy on needle-in-haystack, multi-key lookup, variable tracking,
          common words extraction, and multi-hop QA tasks.
Expected runtime: ~8h on 8x A100-40GB.
Required GPU resources: 8x A100-40GB (plgrid-gpu-a100).
Output files: results/ruler_<config>.json
"""

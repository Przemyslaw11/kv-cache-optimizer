"""LongBench evaluation (6 tasks) for quantized vs fp16 prefill.

Measures: Accuracy on NarrativeQA, Qasper, MultiFieldQA-en, HotpotQA,
          2WikiMultihopQA, Musique.
Expected runtime: ~12h on 8× A100-40GB.
Required GPU resources: 8× A100-40GB (plgrid-gpu-a100).
Output files: results/longbench_<config>.json
"""

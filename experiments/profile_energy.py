"""NVML energy-per-token profiling for fp16 vs quantized prefill.

Measures: Energy per token (J), average power (W), peak power (W),
          carbon emissions (gCO2e via CodeCarbon).
Expected runtime: ~12h on 8× A100-40GB.
Required GPU resources: 8× A100-40GB (plgrid-gpu-a100).
Output files: results/energy_<config>.json, results/energy_<config>.csv
"""

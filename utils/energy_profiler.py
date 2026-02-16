"""DetailedEnergyProfiler with CSV logging for experiment profiling.

Extends EnergyMonitor with structured CSV output and per-configuration
energy-per-token calculations for systematic benchmarking.

Example:
    >>> profiler = DetailedEnergyProfiler(output_dir="results/energy")
    >>> with profiler.profile("config_nuq3_len2048_batch4") as ctx:
    ...     # ... run inference ...
    ...     ctx.set_tokens(2048 * 4)
    >>> profiler.save_summary()
"""

from __future__ import annotations

import csv
import json
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event, Thread

logger = logging.getLogger(__name__)


@dataclass
class ProfileResult:
    """Result from a single profiling session.

    Attributes:
        config_name: Configuration identifier string.
        duration_sec: Wall-clock duration in seconds.
        total_energy_joules: Total energy consumed in joules.
        avg_power_watts: Average power draw in watts.
        peak_power_watts: Peak power draw in watts.
        total_tokens: Number of tokens processed.
        energy_per_token_mj: Energy per token in millijoules.
        throughput_tokens_per_sec: Tokens processed per second.
    """

    config_name: str = ""
    duration_sec: float = 0.0
    total_energy_joules: float = 0.0
    avg_power_watts: float = 0.0
    peak_power_watts: float = 0.0
    total_tokens: int = 0
    energy_per_token_mj: float = 0.0
    throughput_tokens_per_sec: float = 0.0
    power_samples: list[tuple[float, float]] = field(default_factory=list)


class _ProfileContext:
    """Context object yielded by ``DetailedEnergyProfiler.profile()``."""

    def __init__(self) -> None:
        self._tokens: int = 0

    def set_tokens(self, total_tokens: int) -> None:
        """Set the number of tokens processed during this profiling session."""
        self._tokens = total_tokens


class DetailedEnergyProfiler:
    """Energy profiler with CSV logging for systematic benchmarking.

    Samples GPU power at a configurable interval via a background thread
    and computes energy via trapezoidal integration.

    Args:
        output_dir: Directory for CSV and JSON output files.
        gpu_id: GPU device index to monitor. Defaults to 0.
        sample_interval_sec: Power sampling interval in seconds (default 0.05).

    Example:
        >>> profiler = DetailedEnergyProfiler("results/energy")
        >>> with profiler.profile("nuq3_len4096") as ctx:
        ...     run_inference(model, inputs)
        ...     ctx.set_tokens(4096)
        >>> profiler.save_summary()
    """

    def __init__(
        self,
        output_dir: str | Path = "results/energy",
        gpu_id: int = 0,
        sample_interval_sec: float = 0.05,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.gpu_id = gpu_id
        self.sample_interval_sec = sample_interval_sec
        self._results: list[ProfileResult] = []

    @contextmanager
    def profile(self, config_name: str):
        """Context manager for profiling a single configuration.

        Args:
            config_name: Descriptive name for this configuration
                (e.g. ``"nuq3_len2048_batch4"``).

        Yields:
            ``_ProfileContext`` — call ``ctx.set_tokens(n)`` to record
            the number of tokens processed.
        """
        ctx = _ProfileContext()
        samples: list[tuple[float, float]] = []
        stop_event = Event()

        # Background sampling thread
        def _sample_loop():
            try:
                import pynvml

                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_id)
                while not stop_event.is_set():
                    power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
                    power_w = power_mw / 1000.0
                    samples.append((time.perf_counter(), power_w))
                    stop_event.wait(self.sample_interval_sec)
                pynvml.nvmlShutdown()
            except Exception as e:
                logger.warning("Energy sampling failed: %s", e)

        thread = Thread(target=_sample_loop, daemon=True)

        start_time = time.perf_counter()
        thread.start()

        try:
            yield ctx
        finally:
            stop_event.set()
            thread.join(timeout=2.0)
            end_time = time.perf_counter()

        duration = end_time - start_time

        # Compute energy via trapezoidal integration
        total_energy = 0.0
        peak_power = 0.0
        for i in range(1, len(samples)):
            dt = samples[i][0] - samples[i - 1][0]
            avg_p = (samples[i][1] + samples[i - 1][1]) / 2.0
            total_energy += avg_p * dt
            peak_power = max(peak_power, samples[i][1])

        avg_power = total_energy / duration if duration > 0 else 0.0
        tokens = ctx._tokens
        energy_per_token_mj = (total_energy / tokens * 1000.0) if tokens > 0 else 0.0
        throughput = tokens / duration if duration > 0 else 0.0

        result = ProfileResult(
            config_name=config_name,
            duration_sec=duration,
            total_energy_joules=total_energy,
            avg_power_watts=avg_power,
            peak_power_watts=peak_power,
            total_tokens=tokens,
            energy_per_token_mj=energy_per_token_mj,
            throughput_tokens_per_sec=throughput,
            power_samples=samples,
        )
        self._results.append(result)

        logger.info(
            "Profile '%s': %.2f J, %.1f W avg, %.3f mJ/tok, %.0f tok/s",
            config_name,
            total_energy,
            avg_power,
            energy_per_token_mj,
            throughput,
        )

        # Write per-config CSV
        self._write_config_csv(result)

    def save_summary(self, filename: str = "energy_summary.json") -> Path:
        """Save summary of all profiled configurations to JSON.

        Args:
            filename: Output filename within ``output_dir``.

        Returns:
            Path to the saved JSON file.
        """
        output_path = self.output_dir / filename
        summary = []
        for r in self._results:
            summary.append(
                {
                    "config_name": r.config_name,
                    "duration_sec": r.duration_sec,
                    "total_energy_joules": r.total_energy_joules,
                    "avg_power_watts": r.avg_power_watts,
                    "peak_power_watts": r.peak_power_watts,
                    "total_tokens": r.total_tokens,
                    "energy_per_token_mj": r.energy_per_token_mj,
                    "throughput_tokens_per_sec": r.throughput_tokens_per_sec,
                }
            )

        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info("Saved energy summary to %s (%d configs)", output_path, len(summary))
        return output_path

    @property
    def results(self) -> list[ProfileResult]:
        """All profiling results collected so far."""
        return list(self._results)

    def _write_config_csv(self, result: ProfileResult) -> None:
        """Write power samples for a single config to CSV."""
        csv_path = self.output_dir / f"{result.config_name}_power.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp_sec", "power_watts"])
            for ts, pw in result.power_samples:
                writer.writerow([f"{ts:.6f}", f"{pw:.2f}"])

        logger.debug("Wrote power samples to %s", csv_path)

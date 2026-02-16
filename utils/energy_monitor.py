"""Lightweight NVML wrapper for GPU energy measurement.

Provides a simple interface to NVML for measuring GPU power draw
and computing energy consumption during inference.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class EnergyMeasurement:
    """Results from an energy measurement session.

    Attributes:
        duration_sec: Total measurement duration in seconds.
        total_energy_joules: Total energy consumed in joules.
        avg_power_watts: Average power draw in watts.
        peak_power_watts: Peak power draw in watts.
        power_samples: List of (timestamp, power_watts) samples.
    """

    duration_sec: float = 0.0
    total_energy_joules: float = 0.0
    avg_power_watts: float = 0.0
    peak_power_watts: float = 0.0
    power_samples: list[tuple[float, float]] = field(default_factory=list)


class EnergyMonitor:
    """Monitor GPU energy consumption using NVML.

    Args:
        gpu_id: GPU device index to monitor. Defaults to 0.
        sample_interval_ms: Sampling interval in milliseconds. Defaults to 50.

    Example:
        >>> monitor = EnergyMonitor(gpu_id=0)
        >>> monitor.start()
        >>> # ... run inference ...
        >>> results = monitor.stop()
        >>> print(f"Energy: {results.total_energy_joules:.4f} J")
    """

    def __init__(self, gpu_id: int = 0, sample_interval_ms: int = 50) -> None:
        self.gpu_id = gpu_id
        self.sample_interval_ms = sample_interval_ms
        self._handle = None
        self._start_time: float | None = None
        self._samples: list[tuple[float, float]] = []

    def start(self) -> None:
        """Begin energy measurement."""
        try:
            import pynvml

            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_id)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize NVML for GPU {self.gpu_id}: {e}") from e
        self._samples = []
        self._start_time = time.perf_counter()

    def sample(self) -> float:
        """Take a single power sample. Returns power in watts."""
        import pynvml

        if self._handle is None:
            raise RuntimeError("EnergyMonitor not started. Call start() first.")
        power_mw = pynvml.nvmlDeviceGetPowerUsage(self._handle)
        power_w = power_mw / 1000.0
        timestamp = time.perf_counter()
        self._samples.append((timestamp, power_w))
        return power_w

    def stop(self) -> EnergyMeasurement:
        """Stop measurement and return results."""
        import pynvml

        if self._start_time is None:
            raise RuntimeError("EnergyMonitor not started. Call start() first.")

        end_time = time.perf_counter()
        duration = end_time - self._start_time

        if len(self._samples) < 2:
            # Take a final sample if we don't have enough
            self.sample()

        # Compute energy via trapezoidal integration
        total_energy = 0.0
        peak_power = 0.0
        for i in range(1, len(self._samples)):
            dt = self._samples[i][0] - self._samples[i - 1][0]
            avg_p = (self._samples[i][1] + self._samples[i - 1][1]) / 2.0
            total_energy += avg_p * dt
            peak_power = max(peak_power, self._samples[i][1])

        avg_power = total_energy / duration if duration > 0 else 0.0

        pynvml.nvmlShutdown()
        self._handle = None
        self._start_time = None

        return EnergyMeasurement(
            duration_sec=duration,
            total_energy_joules=total_energy,
            avg_power_watts=avg_power,
            peak_power_watts=peak_power,
            power_samples=list(self._samples),
        )

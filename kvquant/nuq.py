"""Non-uniform quantization (NUQ) datatypes.

Implements offline-computed non-uniform quantization codebooks
following the KVQuant methodology (NeurIPS 2024).

Supported datatypes:
    - nuq2: 2-bit (4 centroids)
    - nuq3: 3-bit (8 centroids)  — default
    - nuq4: 4-bit (16 centroids)

The codebook is either pre-computed via sensitivity-weighted K-means
on calibration data, or initialized with a uniform/normal heuristic
for rapid prototyping.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Registry of supported NUQ bit-widths and their default heuristic codebooks.
# Real codebooks are computed offline from calibration data (see
# ``compute_nuq_codebook``).  The heuristic codebooks use quantiles of a
# standard normal distribution so that round-trip error is small for roughly
# Gaussian data — which is a reasonable first approximation for transformer
# key/value activations.
# ---------------------------------------------------------------------------

_NUQ_BIT_WIDTHS: dict[str, int] = {
    "nuq2": 2,
    "nuq3": 3,
    "nuq4": 4,
}


def _default_codebook(num_bits: int, device: torch.device | str = "cpu") -> torch.Tensor:
    """Generate a heuristic codebook using standard-normal quantiles.

    Args:
        num_bits: Number of quantization bits (2, 3, or 4).
        device: Target device for the codebook tensor.

    Returns:
        Sorted codebook tensor of shape ``[2**num_bits]``.
    """
    num_levels = 2**num_bits
    # Quantiles of N(0,1) — gives near-optimal centroids for Gaussian data
    quantiles = torch.linspace(0.5 / num_levels, 1.0 - 0.5 / num_levels, num_levels)
    codebook = torch.erfinv(2 * quantiles - 1) * (2**0.5)
    return codebook.to(device=device, dtype=torch.float16)


def get_nuq_bitwidth(nuq_datatype: str) -> int:
    """Return the number of bits for a given NUQ datatype name.

    Args:
        nuq_datatype: One of ``"nuq2"``, ``"nuq3"``, ``"nuq4"``.

    Returns:
        Number of quantization bits.

    Raises:
        ValueError: If *nuq_datatype* is not recognized.
    """
    if nuq_datatype not in _NUQ_BIT_WIDTHS:
        raise ValueError(
            f"Unknown NUQ datatype '{nuq_datatype}'. Supported: {list(_NUQ_BIT_WIDTHS.keys())}"
        )
    return _NUQ_BIT_WIDTHS[nuq_datatype]


def get_num_levels(nuq_datatype: str) -> int:
    """Return the number of quantization levels (2^bits).

    Args:
        nuq_datatype: One of ``"nuq2"``, ``"nuq3"``, ``"nuq4"``.

    Returns:
        Number of quantization levels.
    """
    return 2 ** get_nuq_bitwidth(nuq_datatype)


# ------------------------------------------------------------------
# Codebook creation & loading
# ------------------------------------------------------------------


def create_heuristic_codebook(
    nuq_datatype: str,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Create a heuristic (non-calibrated) codebook for rapid prototyping.

    The codebook uses quantiles of a standard normal distribution, which is
    a reasonable first approximation for transformer key/value activations.

    Args:
        nuq_datatype: One of ``"nuq2"``, ``"nuq3"``, ``"nuq4"``.
        device: Target device.

    Returns:
        Sorted codebook tensor of shape ``[2**num_bits]``, dtype ``float16``.
    """
    num_bits = get_nuq_bitwidth(nuq_datatype)
    codebook = _default_codebook(num_bits, device=device)
    logger.debug(
        "Created heuristic %s codebook (%d levels) on %s",
        nuq_datatype,
        codebook.numel(),
        device,
    )
    return codebook


def compute_nuq_codebook(
    calibration_data: torch.Tensor,
    nuq_datatype: str,
    num_iterations: int = 100,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Compute an NUQ codebook from calibration data via K-means.

    Uses Lloyd's algorithm (K-means on 1-D data) to find optimal centroids
    that minimize quantization error on the calibration distribution.

    Args:
        calibration_data: 1-D tensor of activation values (e.g. a flattened
            sample of key activations from a calibration set).
        nuq_datatype: One of ``"nuq2"``, ``"nuq3"``, ``"nuq4"``.
        num_iterations: Number of K-means iterations.
        device: Target device for computation.

    Returns:
        Sorted codebook tensor of shape ``[2**num_bits]``, dtype ``float16``.
    """
    num_bits = get_nuq_bitwidth(nuq_datatype)
    k = 2**num_bits
    data = calibration_data.flatten().to(device=device, dtype=torch.float32)

    assert data.numel() > k, f"Need at least {k} calibration samples, got {data.numel()}"

    # Initialize centroids from quantiles of the data
    quantiles = torch.linspace(0.0, 1.0, k + 2, device=device)[1:-1]
    centroids = torch.quantile(data, quantiles)

    for iteration in range(num_iterations):
        # Assignment step: each sample → nearest centroid
        # data: [N], centroids: [k]  →  distances: [N, k]
        distances = torch.abs(data.unsqueeze(1) - centroids.unsqueeze(0))
        assignments = distances.argmin(dim=1)  # [N]

        # Update step: each centroid = mean of assigned samples
        new_centroids = torch.zeros_like(centroids)
        for c in range(k):
            mask = assignments == c
            if mask.any():
                new_centroids[c] = data[mask].mean()
            else:
                # Dead centroid — reinitialize from data
                new_centroids[c] = data[torch.randint(data.numel(), (1,))]

        # Convergence check
        shift = (new_centroids - centroids).abs().max().item()
        centroids = new_centroids
        if shift < 1e-6:
            logger.debug("K-means converged at iteration %d (shift=%.2e)", iteration, shift)
            break

    centroids = centroids.sort().values.to(dtype=torch.float16)
    logger.info(
        "Computed %s codebook (%d levels) from %d samples in %d iterations",
        nuq_datatype,
        k,
        data.numel(),
        iteration + 1,
    )
    return centroids


def save_codebook(codebook: torch.Tensor, path: str | Path) -> None:
    """Save a codebook tensor to disk.

    Args:
        codebook: Codebook tensor to save.
        path: File path (will be created with ``.pt`` extension).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(codebook, path)
    logger.info("Saved codebook to %s", path)


def load_codebook(path: str | Path, device: torch.device | str = "cpu") -> torch.Tensor:
    """Load a codebook tensor from disk.

    Args:
        path: Path to the saved codebook ``.pt`` file.
        device: Device to load the codebook onto.

    Returns:
        Loaded codebook tensor.

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Codebook not found: {path}")
    codebook = torch.load(path, map_location=device, weights_only=True)
    logger.info("Loaded codebook from %s (%d levels)", path, codebook.numel())
    return codebook


# ------------------------------------------------------------------
# Quantization / dequantization primitives
# ------------------------------------------------------------------


def quantize_to_nuq(
    values: torch.Tensor,
    codebook: torch.Tensor,
) -> torch.Tensor:
    """Quantize values by mapping each to the nearest codebook centroid.

    Args:
        values: Tensor of arbitrary shape to quantize.
        codebook: 1-D codebook tensor of shape ``[num_levels]``.

    Returns:
        Integer index tensor of the same shape as *values*, with dtype
        ``torch.uint8`` (supports codebooks up to 256 levels).
    """
    # values: [*shape], codebook: [K]
    # Flatten for broadcasting, then reshape back
    flat = values.reshape(-1).to(dtype=torch.float32)
    cb = codebook.to(dtype=torch.float32)

    # distances: [N, K]
    distances = torch.abs(flat.unsqueeze(1) - cb.unsqueeze(0))
    indices = distances.argmin(dim=1).to(dtype=torch.uint8)

    return indices.reshape(values.shape)


def dequantize_from_nuq(
    indices: torch.Tensor,
    codebook: torch.Tensor,
) -> torch.Tensor:
    """Dequantize indices by looking up codebook values.

    Args:
        indices: Integer index tensor (any shape, dtype ``uint8`` or ``int``).
        codebook: 1-D codebook tensor of shape ``[num_levels]``.

    Returns:
        Dequantized tensor with the same shape as *indices*, dtype ``float16``.
    """
    return codebook[indices.long()]


@lru_cache(maxsize=16)
def get_cached_codebook(
    nuq_datatype: str,
    device: str = "cpu",
) -> torch.Tensor:
    """Return a cached heuristic codebook (useful for tests and prototyping).

    Args:
        nuq_datatype: One of ``"nuq2"``, ``"nuq3"``, ``"nuq4"``.
        device: Target device string.

    Returns:
        Cached codebook tensor.
    """
    return create_heuristic_codebook(nuq_datatype, device=device)

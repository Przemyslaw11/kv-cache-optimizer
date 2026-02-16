"""Blocked CSC/CSR sparse matrix structures with zero-copy append.

Pre-allocates memory in configurable blocks (default 256 tokens) to eliminate
reallocation overhead when building sparse outlier matrices during prefill.

Key design:
    - ``BlockedCSCMatrix``: Compressed Sparse Column — used for **Keys**
      (outlier columns correspond to token positions along seq_len).
    - ``BlockedCSRMatrix``: Compressed Sparse Row — used for **Values**
      (outlier rows correspond to token positions along seq_len).
    - O(1) append per token; O(n) only when converting to dense for
      attention computation.
    - Never use ``torch.cat`` in a loop — always use ``append_token()``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# BlockedCSCMatrix — for Key outliers
# ---------------------------------------------------------------------------


@dataclass
class _Block:
    """A single pre-allocated memory block for sparse entries.

    Attributes:
        row_indices: Pre-allocated row index buffer [max_nnz_per_block].
        values: Pre-allocated value buffer [max_nnz_per_block].
        col_offsets: Number of non-zeros stored per column (token) [block_size].
        nnz: Current number of non-zeros stored in this block.
        num_columns: Number of columns (tokens) written to this block.
    """

    row_indices: torch.Tensor
    values: torch.Tensor
    col_offsets: torch.Tensor
    nnz: int = 0
    num_columns: int = 0


class BlockedCSCMatrix:
    """Blocked Compressed Sparse Column matrix with O(1) token append.

    Pre-allocates memory in blocks of ``block_size`` tokens. When a block
    is full, a new block is allocated transparently. Converting to a dense
    tensor or standard CSC is only done when needed for attention computation.

    Args:
        num_rows: Number of rows (typically ``head_dim``).
        block_size: Number of token-columns per block. Must be a positive
            integer. Candidates for ablation: 64, 128, 256, 512, 1024.
        max_nnz_per_block: Maximum non-zeros per block. Defaults to
            ``block_size * num_rows * 0.02`` (2 % sparsity assumption).
        device: Device for pre-allocated tensors.
        dtype: Data type for values. Defaults to ``torch.float16``.

    Example:
        >>> csc = BlockedCSCMatrix(num_rows=128, block_size=256, device="cuda")
        >>> csc.append_token(
        ...     row_indices=torch.tensor([3, 17, 42], device="cuda"),
        ...     values=torch.tensor([0.5, -1.2, 0.8], device="cuda", dtype=torch.float16),
        ... )
        >>> dense = csc.to_dense()  # [num_rows, total_columns]
    """

    def __init__(
        self,
        num_rows: int,
        block_size: int = 256,
        max_nnz_per_block: int | None = None,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float16,
    ) -> None:
        assert block_size > 0, f"block_size must be positive, got {block_size}"
        assert num_rows > 0, f"num_rows must be positive, got {num_rows}"

        self.num_rows = num_rows
        self.block_size = block_size
        self.device = torch.device(device)
        self.dtype = dtype

        # Default: assume ~2% sparsity → nnz ≈ block_size * num_rows * 0.02
        if max_nnz_per_block is None:
            max_nnz_per_block = max(block_size * max(num_rows // 50, 1), 64)
        self.max_nnz_per_block = max_nnz_per_block

        self._blocks: list[_Block] = []
        self._total_columns: int = 0

        # Allocate the first block
        self._allocate_block()

        logger.debug(
            "BlockedCSCMatrix: num_rows=%d, block_size=%d, max_nnz_per_block=%d, device=%s",
            num_rows,
            block_size,
            max_nnz_per_block,
            device,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def total_columns(self) -> int:
        """Total number of columns (tokens) appended so far."""
        return self._total_columns

    @property
    def total_nnz(self) -> int:
        """Total number of non-zero entries across all blocks."""
        return sum(b.nnz for b in self._blocks)

    @property
    def num_blocks_allocated(self) -> int:
        """Number of memory blocks currently allocated."""
        return len(self._blocks)

    @property
    def shape(self) -> tuple[int, int]:
        """Logical shape ``(num_rows, total_columns)``."""
        return (self.num_rows, self._total_columns)

    def append_token(
        self,
        row_indices: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        """Append outlier entries for a single token (column) — O(1).

        Args:
            row_indices: 1-D tensor of row positions for outliers.
                Shape ``[num_outliers]``, dtype ``int32`` or ``int64``.
            values: 1-D tensor of outlier values.
                Shape ``[num_outliers]``, dtype ``self.dtype``.

        Raises:
            RuntimeError: If the current block cannot fit the new entries
                and a new block also cannot (extremely unlikely).
        """
        assert row_indices.ndim == 1, f"row_indices must be 1-D, got shape {row_indices.shape}"
        assert values.ndim == 1, f"values must be 1-D, got shape {values.shape}"
        assert row_indices.shape[0] == values.shape[0], (
            f"row_indices ({row_indices.shape[0]}) and values ({values.shape[0]}) "
            "must have the same length"
        )

        num_new = row_indices.shape[0]
        block = self._blocks[-1]

        # Check if current block is full (either by columns or by nnz capacity)
        if block.num_columns >= self.block_size or block.nnz + num_new > self.max_nnz_per_block:
            self._allocate_block()
            block = self._blocks[-1]

        # If the new entries still exceed the fresh block's capacity,
        # allocate a larger custom block for this token.
        if num_new > self.max_nnz_per_block:
            self._allocate_block(custom_nnz=num_new)
            block = self._blocks[-1]

        # Write row indices and values into the block
        start = block.nnz
        end = start + num_new
        block.row_indices[start:end] = row_indices.to(device=self.device, dtype=torch.int32)
        block.values[start:end] = values.to(device=self.device, dtype=self.dtype)
        block.col_offsets[block.num_columns] = num_new
        block.nnz += num_new
        block.num_columns += 1
        self._total_columns += 1

    def append_tokens_batch(
        self,
        row_indices_list: list[torch.Tensor],
        values_list: list[torch.Tensor],
    ) -> None:
        """Append outliers for multiple tokens at once.

        This is a convenience wrapper that calls ``append_token`` in a loop.
        The per-token overhead is already O(1), so batching mainly avoids
        Python call overhead.

        Args:
            row_indices_list: List of row-index tensors, one per token.
            values_list: List of value tensors, one per token.
        """
        assert len(row_indices_list) == len(values_list), (
            f"Mismatched lengths: {len(row_indices_list)} vs {len(values_list)}"
        )
        for row_idx, val in zip(row_indices_list, values_list, strict=False):
            self.append_token(row_idx, val)

    def to_dense(self) -> torch.Tensor:
        """Convert the blocked CSC structure to a dense 2-D tensor.

        Returns:
            Dense tensor of shape ``[num_rows, total_columns]``,
            dtype ``self.dtype``, on ``self.device``.
        """
        dense = torch.zeros(
            self.num_rows,
            self._total_columns,
            dtype=self.dtype,
            device=self.device,
        )

        col_offset = 0
        for block in self._blocks:
            nnz_cursor = 0
            for local_col in range(block.num_columns):
                num_entries = block.col_offsets[local_col].item()
                if num_entries > 0:
                    rows = block.row_indices[nnz_cursor : nnz_cursor + num_entries].long()
                    vals = block.values[nnz_cursor : nnz_cursor + num_entries]
                    global_col = col_offset + local_col
                    dense[rows, global_col] = vals
                    nnz_cursor += num_entries
            col_offset += block.num_columns

        return dense

    def to_standard_csc(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert to standard CSC format tensors.

        Returns:
            Tuple of ``(col_offsets, row_indices, values)`` where:
            - ``col_offsets``: ``[total_columns + 1]`` — cumulative nnz per column.
            - ``row_indices``: ``[total_nnz]`` — row index for each non-zero.
            - ``values``: ``[total_nnz]`` — value for each non-zero.
        """
        all_col_counts: list[torch.Tensor] = []
        all_row_indices: list[torch.Tensor] = []
        all_values: list[torch.Tensor] = []

        for block in self._blocks:
            col_counts = block.col_offsets[: block.num_columns]
            all_col_counts.append(col_counts)
            all_row_indices.append(block.row_indices[: block.nnz])
            all_values.append(block.values[: block.nnz])

        if not all_col_counts:
            empty_offsets = torch.zeros(1, dtype=torch.int32, device=self.device)
            empty_indices = torch.zeros(0, dtype=torch.int32, device=self.device)
            empty_values = torch.zeros(0, dtype=self.dtype, device=self.device)
            return empty_offsets, empty_indices, empty_values

        col_counts = torch.cat(all_col_counts)
        row_indices = torch.cat(all_row_indices)
        values = torch.cat(all_values)

        # Build cumulative col_offsets
        col_offsets = torch.zeros(self._total_columns + 1, dtype=torch.int32, device=self.device)
        col_offsets[1:] = col_counts.cumsum(dim=0)

        return col_offsets, row_indices, values

    def reset(self) -> None:
        """Clear all data and reset to a single empty block."""
        self._blocks.clear()
        self._total_columns = 0
        self._allocate_block()

    def memory_allocated_bytes(self) -> int:
        """Return total bytes of pre-allocated memory across all blocks."""
        total = 0
        for block in self._blocks:
            total += block.row_indices.nelement() * block.row_indices.element_size()
            total += block.values.nelement() * block.values.element_size()
            total += block.col_offsets.nelement() * block.col_offsets.element_size()
        return total

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _allocate_block(self, custom_nnz: int | None = None) -> None:
        """Allocate a new memory block."""
        nnz_capacity = custom_nnz if custom_nnz is not None else self.max_nnz_per_block
        block = _Block(
            row_indices=torch.zeros(nnz_capacity, dtype=torch.int32, device=self.device),
            values=torch.zeros(nnz_capacity, dtype=self.dtype, device=self.device),
            col_offsets=torch.zeros(self.block_size, dtype=torch.int32, device=self.device),
        )
        self._blocks.append(block)

    def __repr__(self) -> str:
        return (
            f"BlockedCSCMatrix(num_rows={self.num_rows}, "
            f"total_columns={self._total_columns}, "
            f"total_nnz={self.total_nnz}, "
            f"num_blocks={self.num_blocks_allocated}, "
            f"block_size={self.block_size})"
        )


# ---------------------------------------------------------------------------
# BlockedCSRMatrix — for Value outliers
# ---------------------------------------------------------------------------


class BlockedCSRMatrix:
    """Blocked Compressed Sparse Row matrix with O(1) token append.

    Mirrors ``BlockedCSCMatrix`` but stores rows instead of columns, making
    it suitable for **Value** outliers where each token corresponds to a row.

    Args:
        num_cols: Number of columns (typically ``head_dim``).
        block_size: Number of token-rows per block.
        max_nnz_per_block: Maximum non-zeros per block.
        device: Device for pre-allocated tensors.
        dtype: Data type for values. Defaults to ``torch.float16``.

    Example:
        >>> csr = BlockedCSRMatrix(num_cols=128, block_size=256, device="cuda")
        >>> csr.append_token(
        ...     col_indices=torch.tensor([5, 60], device="cuda"),
        ...     values=torch.tensor([1.1, -0.3], device="cuda", dtype=torch.float16),
        ... )
        >>> dense = csr.to_dense()  # [total_rows, num_cols]
    """

    def __init__(
        self,
        num_cols: int,
        block_size: int = 256,
        max_nnz_per_block: int | None = None,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float16,
    ) -> None:
        assert block_size > 0, f"block_size must be positive, got {block_size}"
        assert num_cols > 0, f"num_cols must be positive, got {num_cols}"

        self.num_cols = num_cols
        self.block_size = block_size
        self.device = torch.device(device)
        self.dtype = dtype

        if max_nnz_per_block is None:
            max_nnz_per_block = max(block_size * max(num_cols // 50, 1), 64)
        self.max_nnz_per_block = max_nnz_per_block

        self._blocks: list[_Block] = []
        self._total_rows: int = 0
        self._allocate_block()

        logger.debug(
            "BlockedCSRMatrix: num_cols=%d, block_size=%d, max_nnz_per_block=%d, device=%s",
            num_cols,
            block_size,
            max_nnz_per_block,
            device,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def total_rows(self) -> int:
        """Total number of rows (tokens) appended so far."""
        return self._total_rows

    @property
    def total_nnz(self) -> int:
        """Total number of non-zero entries across all blocks."""
        return sum(b.nnz for b in self._blocks)

    @property
    def num_blocks_allocated(self) -> int:
        """Number of memory blocks currently allocated."""
        return len(self._blocks)

    @property
    def shape(self) -> tuple[int, int]:
        """Logical shape ``(total_rows, num_cols)``."""
        return (self._total_rows, self.num_cols)

    def append_token(
        self,
        col_indices: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        """Append outlier entries for a single token (row) — O(1).

        Args:
            col_indices: 1-D tensor of column positions for outliers.
            values: 1-D tensor of outlier values.
        """
        assert col_indices.ndim == 1, f"col_indices must be 1-D, got shape {col_indices.shape}"
        assert values.ndim == 1, f"values must be 1-D, got shape {values.shape}"
        assert col_indices.shape[0] == values.shape[0], (
            f"col_indices ({col_indices.shape[0]}) and values ({values.shape[0]}) "
            "must have the same length"
        )

        num_new = col_indices.shape[0]
        block = self._blocks[-1]

        if block.num_columns >= self.block_size or block.nnz + num_new > self.max_nnz_per_block:
            self._allocate_block()
            block = self._blocks[-1]

        if num_new > self.max_nnz_per_block:
            self._allocate_block(custom_nnz=num_new)
            block = self._blocks[-1]

        start = block.nnz
        end = start + num_new
        # Re-use _Block's row_indices field to store col_indices (symmetric usage)
        block.row_indices[start:end] = col_indices.to(device=self.device, dtype=torch.int32)
        block.values[start:end] = values.to(device=self.device, dtype=self.dtype)
        block.col_offsets[block.num_columns] = num_new
        block.nnz += num_new
        block.num_columns += 1
        self._total_rows += 1

    def append_tokens_batch(
        self,
        col_indices_list: list[torch.Tensor],
        values_list: list[torch.Tensor],
    ) -> None:
        """Append outliers for multiple tokens (rows) at once.

        Args:
            col_indices_list: List of col-index tensors, one per token.
            values_list: List of value tensors, one per token.
        """
        assert len(col_indices_list) == len(values_list)
        for col_idx, val in zip(col_indices_list, values_list, strict=False):
            self.append_token(col_idx, val)

    def to_dense(self) -> torch.Tensor:
        """Convert to a dense 2-D tensor of shape ``[total_rows, num_cols]``."""
        dense = torch.zeros(
            self._total_rows,
            self.num_cols,
            dtype=self.dtype,
            device=self.device,
        )

        row_offset = 0
        for block in self._blocks:
            nnz_cursor = 0
            for local_row in range(block.num_columns):
                num_entries = block.col_offsets[local_row].item()
                if num_entries > 0:
                    cols = block.row_indices[nnz_cursor : nnz_cursor + num_entries].long()
                    vals = block.values[nnz_cursor : nnz_cursor + num_entries]
                    global_row = row_offset + local_row
                    dense[global_row, cols] = vals
                    nnz_cursor += num_entries
            row_offset += block.num_columns

        return dense

    def to_standard_csr(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert to standard CSR format tensors.

        Returns:
            Tuple of ``(row_offsets, col_indices, values)`` where:
            - ``row_offsets``: ``[total_rows + 1]`` — cumulative nnz per row.
            - ``col_indices``: ``[total_nnz]`` — column index for each non-zero.
            - ``values``: ``[total_nnz]`` — value for each non-zero.
        """
        all_row_counts: list[torch.Tensor] = []
        all_col_indices: list[torch.Tensor] = []
        all_values: list[torch.Tensor] = []

        for block in self._blocks:
            row_counts = block.col_offsets[: block.num_columns]
            all_row_counts.append(row_counts)
            all_col_indices.append(block.row_indices[: block.nnz])
            all_values.append(block.values[: block.nnz])

        if not all_row_counts:
            empty_offsets = torch.zeros(1, dtype=torch.int32, device=self.device)
            empty_indices = torch.zeros(0, dtype=torch.int32, device=self.device)
            empty_values = torch.zeros(0, dtype=self.dtype, device=self.device)
            return empty_offsets, empty_indices, empty_values

        row_counts = torch.cat(all_row_counts)
        col_indices = torch.cat(all_col_indices)
        values = torch.cat(all_values)

        row_offsets = torch.zeros(self._total_rows + 1, dtype=torch.int32, device=self.device)
        row_offsets[1:] = row_counts.cumsum(dim=0)

        return row_offsets, col_indices, values

    def reset(self) -> None:
        """Clear all data and reset to a single empty block."""
        self._blocks.clear()
        self._total_rows = 0
        self._allocate_block()

    def memory_allocated_bytes(self) -> int:
        """Return total bytes of pre-allocated memory across all blocks."""
        total = 0
        for block in self._blocks:
            total += block.row_indices.nelement() * block.row_indices.element_size()
            total += block.values.nelement() * block.values.element_size()
            total += block.col_offsets.nelement() * block.col_offsets.element_size()
        return total

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _allocate_block(self, custom_nnz: int | None = None) -> None:
        nnz_capacity = custom_nnz if custom_nnz is not None else self.max_nnz_per_block
        block = _Block(
            row_indices=torch.zeros(nnz_capacity, dtype=torch.int32, device=self.device),
            values=torch.zeros(nnz_capacity, dtype=self.dtype, device=self.device),
            col_offsets=torch.zeros(self.block_size, dtype=torch.int32, device=self.device),
        )
        self._blocks.append(block)

    def __repr__(self) -> str:
        return (
            f"BlockedCSRMatrix(num_cols={self.num_cols}, "
            f"total_rows={self._total_rows}, "
            f"total_nnz={self.total_nnz}, "
            f"num_blocks={self.num_blocks_allocated}, "
            f"block_size={self.block_size})"
        )

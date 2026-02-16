"""Tests for BlockedCSCMatrix / BlockedCSRMatrix — zero-copy append correctness.

Validates:
    - Single-token and multi-token append correctness.
    - ``to_dense()`` matches naive ``torch.cat`` reference.
    - ``to_standard_csc()`` / ``to_standard_csr()`` format correctness.
    - Block boundary handling (auto-allocation of new blocks).
    - Empty matrix, empty token, and edge-case handling.
    - Memory accounting (``memory_allocated_bytes``).
    - Reset clears all state.
"""

import pytest
import torch

from kvquant.blocked_sparse import BlockedCSCMatrix, BlockedCSRMatrix

# =====================================================================
# BlockedCSCMatrix tests
# =====================================================================


class TestBlockedCSCMatrix:
    """Test suite for BlockedCSCMatrix (Key outliers, column = token)."""

    def test_empty_matrix_shape(self) -> None:
        """Empty matrix should have zero columns and correct num_rows."""
        csc = BlockedCSCMatrix(num_rows=128, block_size=64)
        assert csc.shape == (128, 0)
        assert csc.total_columns == 0
        assert csc.total_nnz == 0

    def test_empty_matrix_to_dense(self) -> None:
        """to_dense() on empty matrix should return zero-size tensor."""
        csc = BlockedCSCMatrix(num_rows=64, block_size=32)
        dense = csc.to_dense()
        assert dense.shape == (64, 0)

    def test_single_token_append(self) -> None:
        """Append a single token with known outlier positions."""
        csc = BlockedCSCMatrix(num_rows=8, block_size=4)
        rows = torch.tensor([1, 5, 7], dtype=torch.int32)
        vals = torch.tensor([0.5, -1.0, 2.0], dtype=torch.float16)

        csc.append_token(rows, vals)

        assert csc.total_columns == 1
        assert csc.total_nnz == 3

        dense = csc.to_dense()
        assert dense.shape == (8, 1)
        assert dense[1, 0].item() == pytest.approx(0.5, abs=0.01)
        assert dense[5, 0].item() == pytest.approx(-1.0, abs=0.01)
        assert dense[7, 0].item() == pytest.approx(2.0, abs=0.01)
        # Other positions should be zero
        assert dense[0, 0].item() == 0.0
        assert dense[3, 0].item() == 0.0

    def test_multi_token_append(self) -> None:
        """Append multiple tokens and verify dense reconstruction."""
        csc = BlockedCSCMatrix(num_rows=4, block_size=8)

        # Token 0: outliers at rows 0, 2
        csc.append_token(
            torch.tensor([0, 2], dtype=torch.int32),
            torch.tensor([1.0, 2.0], dtype=torch.float16),
        )
        # Token 1: outlier at row 3
        csc.append_token(
            torch.tensor([3], dtype=torch.int32),
            torch.tensor([-0.5], dtype=torch.float16),
        )
        # Token 2: no outliers
        csc.append_token(
            torch.tensor([], dtype=torch.int32),
            torch.tensor([], dtype=torch.float16),
        )

        assert csc.total_columns == 3
        assert csc.total_nnz == 3

        dense = csc.to_dense()
        assert dense.shape == (4, 3)
        assert dense[0, 0].item() == pytest.approx(1.0, abs=0.01)
        assert dense[2, 0].item() == pytest.approx(2.0, abs=0.01)
        assert dense[3, 1].item() == pytest.approx(-0.5, abs=0.01)
        assert dense[0, 2].item() == 0.0  # Empty token column

    def test_block_boundary_auto_allocation(self) -> None:
        """Exceeding block_size should auto-allocate a new block."""
        block_size = 4
        csc = BlockedCSCMatrix(num_rows=8, block_size=block_size)

        for i in range(10):  # Exceeds block_size=4
            csc.append_token(
                torch.tensor([i % 8], dtype=torch.int32),
                torch.tensor([float(i)], dtype=torch.float16),
            )

        assert csc.total_columns == 10
        assert csc.num_blocks_allocated >= 3  # At least ceil(10/4) blocks

        dense = csc.to_dense()
        assert dense.shape == (8, 10)
        # Verify each token
        for i in range(10):
            assert dense[i % 8, i].item() == pytest.approx(float(i), abs=0.1)

    def test_to_standard_csc_format(self) -> None:
        """Verify CSC format conversion is correct."""
        csc = BlockedCSCMatrix(num_rows=4, block_size=8)

        # Token 0: 2 outliers
        csc.append_token(
            torch.tensor([0, 3], dtype=torch.int32),
            torch.tensor([1.0, 2.0], dtype=torch.float16),
        )
        # Token 1: 1 outlier
        csc.append_token(
            torch.tensor([1], dtype=torch.int32),
            torch.tensor([3.0], dtype=torch.float16),
        )

        col_offsets, row_indices, values = csc.to_standard_csc()

        # col_offsets: [0, 2, 3]  (col 0 has 2 nnz, col 1 has 1 nnz)
        assert col_offsets.shape == (3,)  # total_columns + 1
        assert col_offsets[0].item() == 0
        assert col_offsets[1].item() == 2
        assert col_offsets[2].item() == 3

        assert row_indices.shape == (3,)
        assert values.shape == (3,)

    def test_to_dense_matches_naive_cat_reference(self) -> None:
        """to_dense() output must match a naive torch.cat reference."""
        num_rows = 16
        num_tokens = 50
        csc = BlockedCSCMatrix(num_rows=num_rows, block_size=8)

        # Build reference with torch.cat
        reference_cols = []
        torch.manual_seed(42)

        for _t in range(num_tokens):
            num_outliers = torch.randint(0, 4, (1,)).item()
            row_idx = torch.randperm(num_rows)[:num_outliers].to(dtype=torch.int32)
            vals = torch.randn(num_outliers, dtype=torch.float16)

            csc.append_token(row_idx, vals)

            col = torch.zeros(num_rows, dtype=torch.float16)
            if num_outliers > 0:
                col[row_idx.long()] = vals
            reference_cols.append(col.unsqueeze(1))

        dense = csc.to_dense()
        reference = torch.cat(reference_cols, dim=1) if reference_cols else torch.zeros(num_rows, 0)

        assert dense.shape == reference.shape
        torch.testing.assert_close(dense, reference, atol=0.01, rtol=0.01)

    def test_reset(self) -> None:
        """Reset should clear all data."""
        csc = BlockedCSCMatrix(num_rows=4, block_size=4)
        csc.append_token(
            torch.tensor([0], dtype=torch.int32), torch.tensor([1.0], dtype=torch.float16)
        )
        assert csc.total_columns == 1

        csc.reset()
        assert csc.total_columns == 0
        assert csc.total_nnz == 0
        assert csc.num_blocks_allocated == 1

    def test_memory_allocated_bytes(self) -> None:
        """memory_allocated_bytes should return a positive value."""
        csc = BlockedCSCMatrix(num_rows=64, block_size=128)
        mem = csc.memory_allocated_bytes()
        assert mem > 0

    def test_batch_append(self) -> None:
        """append_tokens_batch should produce same result as individual appends."""
        csc1 = BlockedCSCMatrix(num_rows=8, block_size=4)
        csc2 = BlockedCSCMatrix(num_rows=8, block_size=4)

        rows_list = [
            torch.tensor([1, 3], dtype=torch.int32),
            torch.tensor([0], dtype=torch.int32),
            torch.tensor([5, 7], dtype=torch.int32),
        ]
        vals_list = [
            torch.tensor([1.0, 2.0], dtype=torch.float16),
            torch.tensor([3.0], dtype=torch.float16),
            torch.tensor([-1.0, 0.5], dtype=torch.float16),
        ]

        # Individual appends
        for r, v in zip(rows_list, vals_list, strict=False):
            csc1.append_token(r, v)

        # Batch append
        csc2.append_tokens_batch(rows_list, vals_list)

        torch.testing.assert_close(csc1.to_dense(), csc2.to_dense())

    def test_repr(self) -> None:
        """__repr__ should include key info."""
        csc = BlockedCSCMatrix(num_rows=64, block_size=128)
        r = repr(csc)
        assert "BlockedCSCMatrix" in r
        assert "64" in r


# =====================================================================
# BlockedCSRMatrix tests
# =====================================================================


class TestBlockedCSRMatrix:
    """Test suite for BlockedCSRMatrix (Value outliers, row = token)."""

    def test_empty_matrix_shape(self) -> None:
        """Empty matrix should have zero rows and correct num_cols."""
        csr = BlockedCSRMatrix(num_cols=128, block_size=64)
        assert csr.shape == (0, 128)
        assert csr.total_rows == 0
        assert csr.total_nnz == 0

    def test_single_token_append(self) -> None:
        """Append a single token (row) with known outlier positions."""
        csr = BlockedCSRMatrix(num_cols=8, block_size=4)
        cols = torch.tensor([2, 5], dtype=torch.int32)
        vals = torch.tensor([1.5, -0.3], dtype=torch.float16)

        csr.append_token(cols, vals)

        assert csr.total_rows == 1
        assert csr.total_nnz == 2

        dense = csr.to_dense()
        assert dense.shape == (1, 8)
        assert dense[0, 2].item() == pytest.approx(1.5, abs=0.01)
        assert dense[0, 5].item() == pytest.approx(-0.3, abs=0.01)

    def test_multi_token_append(self) -> None:
        """Append multiple tokens and verify dense reconstruction."""
        csr = BlockedCSRMatrix(num_cols=4, block_size=8)

        csr.append_token(
            torch.tensor([0, 3], dtype=torch.int32),
            torch.tensor([1.0, 2.0], dtype=torch.float16),
        )
        csr.append_token(
            torch.tensor([1], dtype=torch.int32),
            torch.tensor([-0.5], dtype=torch.float16),
        )

        assert csr.total_rows == 2
        dense = csr.to_dense()
        assert dense.shape == (2, 4)
        assert dense[0, 0].item() == pytest.approx(1.0, abs=0.01)
        assert dense[0, 3].item() == pytest.approx(2.0, abs=0.01)
        assert dense[1, 1].item() == pytest.approx(-0.5, abs=0.01)

    def test_block_boundary(self) -> None:
        """Exceeding block_size should auto-allocate a new block."""
        csr = BlockedCSRMatrix(num_cols=4, block_size=3)

        for i in range(7):
            csr.append_token(
                torch.tensor([i % 4], dtype=torch.int32),
                torch.tensor([float(i)], dtype=torch.float16),
            )

        assert csr.total_rows == 7
        assert csr.num_blocks_allocated >= 3

        dense = csr.to_dense()
        assert dense.shape == (7, 4)

    def test_to_dense_matches_reference(self) -> None:
        """to_dense() must match a naive reference build."""
        num_cols = 16
        num_tokens = 40
        csr = BlockedCSRMatrix(num_cols=num_cols, block_size=8)

        reference_rows = []
        torch.manual_seed(123)

        for _t in range(num_tokens):
            num_outliers = torch.randint(0, 5, (1,)).item()
            col_idx = torch.randperm(num_cols)[:num_outliers].to(dtype=torch.int32)
            vals = torch.randn(num_outliers, dtype=torch.float16)

            csr.append_token(col_idx, vals)

            row = torch.zeros(num_cols, dtype=torch.float16)
            if num_outliers > 0:
                row[col_idx.long()] = vals
            reference_rows.append(row.unsqueeze(0))

        dense = csr.to_dense()
        reference = torch.cat(reference_rows, dim=0)

        assert dense.shape == reference.shape
        torch.testing.assert_close(dense, reference, atol=0.01, rtol=0.01)

    def test_to_standard_csr_format(self) -> None:
        """Verify CSR format conversion."""
        csr = BlockedCSRMatrix(num_cols=4, block_size=8)

        csr.append_token(
            torch.tensor([0, 2], dtype=torch.int32),
            torch.tensor([1.0, 2.0], dtype=torch.float16),
        )
        csr.append_token(
            torch.tensor([3], dtype=torch.int32),
            torch.tensor([3.0], dtype=torch.float16),
        )

        row_offsets, _col_indices, _values = csr.to_standard_csr()

        assert row_offsets.shape == (3,)  # total_rows + 1
        assert row_offsets[0].item() == 0
        assert row_offsets[1].item() == 2
        assert row_offsets[2].item() == 3

    def test_reset(self) -> None:
        """Reset should clear all data."""
        csr = BlockedCSRMatrix(num_cols=4, block_size=4)
        csr.append_token(
            torch.tensor([0], dtype=torch.int32), torch.tensor([1.0], dtype=torch.float16)
        )
        assert csr.total_rows == 1

        csr.reset()
        assert csr.total_rows == 0
        assert csr.total_nnz == 0
        assert csr.num_blocks_allocated == 1

    def test_repr(self) -> None:
        """__repr__ should include key info."""
        csr = BlockedCSRMatrix(num_cols=64, block_size=128)
        r = repr(csr)
        assert "BlockedCSRMatrix" in r
        assert "64" in r

"""Blocked CSC/CSR sparse matrix structures with zero-copy append.

Pre-allocates memory in configurable blocks (default 256 tokens) to eliminate
reallocation overhead when building sparse outlier matrices during prefill.
"""

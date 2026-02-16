"""Drop-in quantized attention replacement for the prefill phase.

Composes BatchedKVQuantizer and BlockedCSCMatrix into a complete
attention module that quantizes KV cache during prefill.
"""

"""Shared bounded generation/serialization limits (not execution guarantees)."""
MAX_NODES = 64
MAX_TENSORS = MAX_NODES * 4
MAX_SHAPES = 8
MAX_CASE_BYTES = 512 * 1024
MAX_CONTEXT_BYTES = 768 * 1024
MAX_SYNC_BYTES = 2 * 1024 * 1024

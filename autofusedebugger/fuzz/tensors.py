"""Canonical storage decoding shared by generators and execution adapters."""
import struct

DTYPES = {"float16": ("e", 2), "float32": ("f", 4)}


def storage_values(bundle, tensor):
    fmt, width = DTYPES[tensor.dtype]
    raw = bundle.storages[tensor.alias_group]
    return struct.unpack("<" + fmt * (len(raw) // width), raw)


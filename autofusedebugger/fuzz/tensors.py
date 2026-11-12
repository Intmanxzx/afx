"""Canonical storage encoding/decoding shared by generators and adapters."""
import struct

# (struct format, byte width). Complex values are encoded as real/imag pairs.
DTYPES = {
    "bool": ("?", 1), "uint8": ("B", 1), "uint16": ("H", 2),
    "uint32": ("I", 4), "int8": ("b", 1), "int16": ("h", 2),
    "int32": ("i", 4), "int64": ("q", 8),
    "float16": ("e", 2), "bfloat16": ("e", 2),
    "float32": ("f", 4), "float64": ("d", 8),
    "complex64": ("ff", 8), "complex128": ("dd", 16),
}


def _bf16_to_float(value):
    # Decode a bfloat16 payload without requiring NumPy/torch at import time.
    bits = int.from_bytes(value, "little") << 16
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def decode_storage(raw, dtype):
    """Decode one canonical storage buffer into Python scalar values."""
    if dtype == "bfloat16":
        return tuple(_bf16_to_float(raw[i:i + 2]) for i in range(0, len(raw), 2))
    fmt, width = DTYPES[dtype]
    values = struct.unpack("<" + fmt * (len(raw) // width), raw)
    if dtype in {"complex64", "complex128"}:
        return tuple(complex(values[i], values[i + 1])
                     for i in range(0, len(values), 2))
    return values


def storage_values(bundle, tensor):
    return decode_storage(bundle.storages[tensor.alias_group], tensor.dtype)

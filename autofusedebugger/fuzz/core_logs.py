"""Drain native/Python worker output into a bounded in-memory tail, never a file."""
from contextlib import contextmanager
import os
import sys
import threading
from .failure_details import redact


@contextmanager
def capture_tail(limit=16384):
    chunks = bytearray()
    important = bytearray()
    read_fd, write_fd = os.pipe()
    saved = [os.dup(fd) for fd in (1, 2)]

    def drain():
        try:
            while True:
                chunk = os.read(read_fd, 4096)
                if not chunk:
                    break
                chunks.extend(chunk)
                del chunks[:-limit]
                for line in chunk.splitlines():
                    if any(word in line.lower() for word in (b'error:', b'af_mlir_strict', b'unsupportedoperation', b'fatal')):
                        important.extend(line + b'\n')
                        del important[:-1600]
        finally:
            os.close(read_fd)

    reader = threading.Thread(target=drain, daemon=True)
    reader.start()
    sys.stdout.flush(); sys.stderr.flush()
    try:
        os.dup2(write_fd, 1); os.dup2(write_fd, 2)
        os.close(write_fd)
        yield lambda: redact(bytes(important or chunks).decode('utf-8', errors='replace'))[-1600:]
    finally:
        sys.stdout.flush(); sys.stderr.flush()
        for fd, original in zip((1, 2), saved):
            os.dup2(original, fd); os.close(original)
        reader.join(timeout=2)

"""Drain native/Python worker output into a bounded in-memory tail, never a file."""
from contextlib import contextmanager
import os
import sys
import threading
from .failure_details import redact


@contextmanager
def capture_tail(limit=16384, *, forward=False):
    chunks = bytearray()
    important = bytearray()
    read_fd, write_fd = os.pipe()
    saved = [os.dup(fd) for fd in (1, 2)]

    def drain():
        pending = b''
        context_lines = 0
        def collect(line):
            nonlocal context_lines
            if any(word in line.lower() for word in (b'error:', b'af_mlir_strict', b'unsupportedoperation', b'fatal')):
                context_lines = 3
            if context_lines:
                important.extend(line + b'\n')
                del important[:-1600]
                context_lines -= 1
        try:
            while True:
                chunk = os.read(read_fd, 4096)
                if not chunk:
                    if pending:
                        collect(pending)
                    break
                if forward:
                    os.write(saved[1], chunk)
                chunks.extend(chunk)
                del chunks[:-limit]
                lines = (pending + chunk).split(b'\n')
                pending = lines.pop()[-limit:]
                for line in lines:
                    collect(line)
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
            os.dup2(original, fd)
        reader.join(timeout=2)
        for original in saved:
            os.close(original)


class ProcessTail:
    """Parent-side capture survives worker segfaults and timeout termination."""
    def __init__(self, stream, limit=16384):
        self.data = bytearray()
        self.stream = stream
        self.limit = limit
        self.thread = threading.Thread(target=self._drain, daemon=True)
        self.thread.start()

    def _drain(self):
        try:
            while True:
                chunk = self.stream.read(4096)
                if not chunk:
                    return
                self.data.extend(chunk)
                del self.data[:-self.limit]
        finally:
            self.stream.close()

    def finish(self):
        self.thread.join(timeout=2)
        return redact(bytes(self.data).decode('utf-8', errors='replace'))[-1600:]

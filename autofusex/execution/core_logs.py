"""Bounded native logs; pipe draining must not depend on the producer's GIL."""
from contextlib import contextmanager
import os
import subprocess
import sys
import threading
from autofusex.execution.failure_details import redact


class _Tail:
    def __init__(self, limit=16384):
        self.limit = limit
        self.chunks = bytearray()
        self.important = bytearray()
        self.verdicts = bytearray()
        self.pending = b''
        self.context_lines = 0

    def _line(self, line):
        if b'mlir codegen result' in b' '.join(line.lower().split()):
            self.verdicts.extend(line + b'\n')
            del self.verdicts[:-600]
        if any(word in line.lower() for word in (b'error:', b'af_mlir_strict', b'unsupportedoperation', b'fatal')):
            self.context_lines = 3
        if self.context_lines:
            self.important.extend(line + b'\n')
            del self.important[:-1600]
            self.context_lines -= 1

    def feed(self, chunk):
        self.chunks.extend(chunk)
        del self.chunks[:-self.limit]
        lines = (self.pending + chunk).split(b'\n')
        self.pending = lines.pop()[-self.limit:]
        for line in lines:
            self._line(line)

    def finish(self):
        if self.pending:
            self._line(self.pending)
            self.pending = b''
        return self.text()

    def text(self):
        verdict = redact(bytes(self.verdicts).decode('utf-8', errors='replace'))[-600:]
        rest = redact(bytes(self.important or self.chunks).decode('utf-8', errors='replace'))
        return verdict + rest[-(1600-len(verdict)):] if verdict else rest[-1600:]


def _drain(limit, forward):
    """Separate interpreter: native writes holding the worker GIL cannot starve us."""
    tail = _Tail(limit)
    while True:
        chunk = os.read(0, 4096)
        if not chunk:
            break
        tail.feed(chunk)
        if forward:
            view = memoryview(chunk)
            while view:
                try:
                    written = os.write(2, view)
                except BrokenPipeError:
                    forward = False
                    break
                view = view[written:]
    sys.stdout.write(tail.finish())


@contextmanager
def capture_tail(limit=16384, *, forward=False):
    """Capture stdout/stderr without a reader in this process; tail available on exit.

    The helper inherits the worker's process group so cancellation also reaps it.
    Only the bounded final diagnostic crosses its stdout pipe; forwarded logs go
    directly to the original stderr (normally drained by the supervisor).
    """
    sys.stdout.flush(); sys.stderr.flush()
    saved = [os.dup(fd) for fd in (1, 2)]
    helper = None
    text = ''
    try:
        helper = subprocess.Popen(
            [sys.executable, '-m', __name__, str(limit), str(int(forward))],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=saved[1],
            close_fds=True)
        os.dup2(helper.stdin.fileno(), 1)
        os.dup2(helper.stdin.fileno(), 2)
        helper.stdin.close()
        helper.stdin = None
        yield lambda: text
    finally:
        try:
            sys.stdout.flush(); sys.stderr.flush()
        finally:
            for fd, original in zip((1, 2), saved):
                os.dup2(original, fd)
                os.close(original)
            if helper is not None:
                try:
                    output, _ = helper.communicate(timeout=5)
                except subprocess.TimeoutExpired:
                    # A leaked writer must not hang shutdown indefinitely.
                    helper.kill()
                    output, _ = helper.communicate()
                text = redact(output.decode('utf-8', errors='replace'))[-1600:]


class ProcessTail:
    """Parent-side capture survives worker segfaults and timeout termination."""
    def __init__(self, stream, limit=16384):
        self.stream = stream
        self.tail = _Tail(limit)
        self.thread = threading.Thread(target=self._drain, daemon=True)
        self.thread.start()

    def _drain(self):
        try:
            while True:
                # read1 returns available bytes, including short pre-crash logs.
                chunk = self.stream.read1(4096)
                if not chunk:
                    self.tail.finish()
                    return
                self.tail.feed(chunk)
        finally:
            self.stream.close()

    def finish(self):
        self.thread.join(timeout=2)
        return self.tail.text()


if __name__ == '__main__':
    _drain(int(sys.argv[1]), bool(int(sys.argv[2])))

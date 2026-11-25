"""Bounded process-local read caches with one computation per concurrent key."""
from collections import OrderedDict
from concurrent.futures import Future
import threading


class ReadCache:
    def __init__(self, *, max_entries, max_bytes, size):
        self.max_entries, self.max_bytes, self.size = max_entries, max_bytes, size
        self._entries = OrderedDict()
        self._pending = {}
        self._bytes = 0
        self._lock = threading.Lock()

    def get(self, key, load):
        with self._lock:
            if key in self._entries:
                value, _ = self._entries[key]
                self._entries.move_to_end(key)
                return value
            pending = self._pending.get(key)
            leader = pending is None
            if leader:
                pending = self._pending[key] = Future()
        if not leader:
            return pending.result()
        try:
            value = load()
            size = self.size(value)
            with self._lock:
                if 0 < size <= self.max_bytes:
                    self._entries[key] = (value, size)
                    self._bytes += size
                    while len(self._entries) > self.max_entries or self._bytes > self.max_bytes:
                        _, (_, removed) = self._entries.popitem(last=False)
                        self._bytes -= removed
                del self._pending[key]
            pending.set_result(value)
            return value
        except BaseException as exc:
            with self._lock:
                self._pending.pop(key, None)
            pending.set_exception(exc)
            raise

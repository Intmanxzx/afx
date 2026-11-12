"""Single-producer public responses. HTTP readers never run cache builders."""
from dataclasses import dataclass
import hashlib
import json
import logging
import threading
import time

from ..storage.queries import ResultQueries
from ..storage.repository import _connect

INTERVAL = 30
MAX_RETRY_INTERVAL = 300
PAGE_SIZES = (10, 20, 25, 50, 100, 200)
MAX_BODY = 2 * 1024 * 1024


@dataclass(frozen=True)
class Response:
    body: bytes
    etag: str
    generated_at: float
    status: int = 200


def response(value, generated_at):
    body = json.dumps(value, ensure_ascii=True, allow_nan=False).encode()
    status = 200
    if len(body) > MAX_BODY:
        body = b'{"error":"response too large"}'
        status = 413
    return Response(body, '"'+hashlib.sha256(body).hexdigest()+'"', generated_at, status)


class PublicCache:
    def __init__(self, database, *, interval=INTERVAL):
        self.database = database
        self._statistics = None
        self.interval = interval
        self._lock = threading.Lock()
        self._entries = {}
        self._failed = False
        self._stop = threading.Event()
        self.ready = threading.Event()
        self._thread = threading.Thread(target=self._run, name='public-cache', daemon=True)

    def start(self):
        self._thread.start()

    def close(self):
        self._stop.set()
        self._thread.join()

    def get(self, key):
        with self._lock:
            return self._entries.get(key), self._failed

    def build(self):
        from ..reporting.incremental import IncrementalStatistics
        queries = ResultQueries(self.database)
        # One bounded list read; derive page sizes using the original SQL cursors.
        page = queries.case_page(limit=200, _include_cursors=True)
        cursors = page.pop('_cursors')
        if self._statistics is None:
            self._statistics = IncrementalStatistics(self.database, stopped=self._stop.is_set)
        charts = self._statistics.refresh()
        counts = {row['status']:row['count'] for row in charts['overall']}
        mlir_failed=counts.pop('mlir_failed',0)
        if mlir_failed:counts['error']=counts.get('error',0)+mlir_failed
        values = {'summary': {'total':sum(counts.values()), 'by_status':counts, 'mlir_failed':mlir_failed, 'unit':'shape_case'},
                  'statistics': charts}
        for size in PAGE_SIZES:
            more = len(page['items']) > size
            values['cases:'+str(size)] = {
                'items': page['items'][:size],
                'next_cursor': cursors[size-1] if more else page['next_cursor'],
            }
        now = time.time()
        return {key: response(value, now) for key, value in values.items()}

    def _run(self):
        conn = None
        version = None
        file_id = None
        retry_delay = self.interval
        try:
            while not self._stop.is_set():
                delay = self.interval
                try:
                    stat = self.database.stat()
                    identity = (stat.st_dev, stat.st_ino)
                    if conn is None or identity != file_id:
                        if conn is not None:
                            conn.close()
                        from ..reporting.incremental import IncrementalStatistics
                        self._statistics = IncrementalStatistics(self.database, stopped=self._stop.is_set)
                        self._statistics.prepare()
                        conn = _connect(self.database)
                        file_id = identity
                        version = None
                    # Persistent connection: data_version detects commits by ANY
                    # writer, including facts uploads and offline maintenance.
                    current = conn.execute('PRAGMA data_version').fetchone()[0]
                    if current != version:
                        entries = self.build()
                        with self._lock:
                            self._entries = entries
                            self._failed = False
                        # Capture BEFORE the build: writes during it trigger the
                        # next refresh rather than disappearing into the cache.
                        version = current
                        self.ready.set()
                    else:
                        with self._lock:
                            self._failed = False
                    retry_delay = self.interval
                except Exception:
                    with self._lock:
                        self._failed = True
                    delay = retry_delay
                    retry_delay = min(max(self.interval, MAX_RETRY_INTERVAL), retry_delay * 2)
                    logging.getLogger(__name__).exception(
                        'Public cache refresh failed; keeping last snapshot; retry in %.1fs', delay)
                # Rest AFTER work, including slow/failed builds. Never catch up
                # by immediately rescanning a busy database. Shutdown wakes this wait.
                self._stop.wait(delay)
        finally:
            if conn is not None:
                conn.close()

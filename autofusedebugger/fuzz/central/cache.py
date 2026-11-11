"""Single-producer public responses. HTTP readers never run cache builders."""
from dataclasses import dataclass
import hashlib
import json
import logging
import threading
import time

from ..storage.queries import ResultQueries
from ..storage.repository import _connect

INTERVAL = 15
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
        from ..reporting.statistics import statistics
        queries = ResultQueries(self.database)
        # One bounded list read; derive page sizes using the original SQL cursors.
        page = queries.case_page(limit=200, _include_cursors=True)
        cursors = page.pop('_cursors')
        charts = statistics(self.database, _timeout=30)
        counts = {row['status']:row['count'] for row in charts['overall']}
        values = {'summary': {'total':sum(counts.values()), 'by_status':counts, 'unit':'shape_case'},
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
        try:
            while not self._stop.is_set():
                start = time.monotonic()
                try:
                    stat = self.database.stat()
                    identity = (stat.st_dev, stat.st_ino)
                    if conn is None or identity != file_id:
                        if conn is not None:
                            conn.close()
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
                except Exception:
                    with self._lock:
                        self._failed = True
                    logging.getLogger(__name__).exception('Public cache refresh failed; keeping last snapshot')
                # No overlapping builds or catch-up bursts after a slow refresh.
                self._stop.wait(max(1, self.interval - (time.monotonic()-start)))
        finally:
            if conn is not None:
                conn.close()

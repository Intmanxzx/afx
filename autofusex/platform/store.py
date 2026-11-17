"""Connections, the single writer lock, settings and content blobs."""
from __future__ import annotations

import hashlib
import json
import os
import secrets
import sqlite3
import threading
import time
import zlib
from contextlib import contextmanager
from pathlib import Path

from . import schema

DEFAULT_SETTINGS = {
    # Kernel ratio x/y inside [1/(1+t), 1+t] counts as a tie (design 6.2).
    "perf.tie_threshold": 0.05,
    # Regression selection budget and layer shares (design 2.1); used from P1.
    "regression.budget": 100,
    "regression.max_budget": 300,
    "regression.shares": {"neighbor": 50, "sentinel": 20, "history": 20, "random": 10},
}

BUSY_TIMEOUT_MS = 10_000


def now_ms() -> int:
    return int(time.time() * 1000)


class Store:
    """Owns the data directory. One writer at a time; readers use their own connections."""

    def __init__(self, data_dir: str | os.PathLike):
        self.dir = Path(data_dir).resolve()
        self.dir.mkdir(parents=True, exist_ok=True)
        self.core_path = self.dir / "core.db"
        self.content_path = self.dir / "content.db"
        self.stats_path = self.dir / "stats.db"
        self.write_lock = threading.Lock()
        self._local = threading.local()
        self.secret = self._load_secret()
        with self.connect(readonly=False) as conn:
            schema.apply(conn, "main", schema.CORE)
            schema.apply(conn, "content", schema.CONTENT)
            schema.apply(conn, "stats", schema.STATS)
            self._seed_settings(conn)

    # -- connections -------------------------------------------------------
    def _open(self, readonly: bool) -> sqlite3.Connection:
        if readonly:
            uri = f"file:{self.core_path}?mode=ro"
            conn = sqlite3.connect(uri, uri=True, isolation_level=None, check_same_thread=False)
            conn.execute(f"ATTACH DATABASE 'file:{self.content_path}?mode=ro' AS content")
            conn.execute(f"ATTACH DATABASE 'file:{self.stats_path}?mode=ro' AS stats")
            conn.execute("PRAGMA query_only=ON")
        else:
            conn = sqlite3.connect(self.core_path, isolation_level=None, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("ATTACH DATABASE ? AS content", (str(self.content_path),))
            conn.execute("ATTACH DATABASE ? AS stats", (str(self.stats_path),))
            for name in ("content", "stats"):
                conn.execute(f"PRAGMA {name}.journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
        conn.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=-65536")
        conn.row_factory = sqlite3.Row
        return conn

    @contextmanager
    def connect(self, readonly: bool = True):
        conn = self._open(readonly)
        try:
            yield conn
        finally:
            conn.close()

    def reader(self) -> sqlite3.Connection:
        """Per-thread cached read-only connection."""
        conn = getattr(self._local, "reader", None)
        if conn is None:
            conn = self._open(readonly=True)
            self._local.reader = conn
        return conn

    @contextmanager
    def write(self):
        """Serialize writers in-process and hold one IMMEDIATE transaction."""
        with self.write_lock:
            conn = getattr(self._local, "writer", None)
            if conn is None:
                conn = self._open(readonly=False)
                self._local.writer = conn
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
            except BaseException:
                conn.execute("ROLLBACK")
                raise
            else:
                conn.execute("COMMIT")

    # -- secret ------------------------------------------------------------
    def _load_secret(self) -> bytes:
        path = self.dir / "secret.key"
        if not path.exists():
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "wb") as handle:
                handle.write(secrets.token_bytes(32))
        return path.read_bytes()

    # -- settings ----------------------------------------------------------
    def _seed_settings(self, conn: sqlite3.Connection) -> None:
        for key, value in DEFAULT_SETTINGS.items():
            conn.execute("INSERT OR IGNORE INTO platform_settings(key, value, updated_at) VALUES(?,?,?)",
                         (key, json.dumps(value), now_ms()))

    def settings(self, conn: sqlite3.Connection | None = None) -> dict:
        conn = conn or self.reader()
        return {row["key"]: {"value": json.loads(row["value"]), "revision": row["revision"]}
                for row in conn.execute("SELECT key, value, revision FROM platform_settings")}

    def setting(self, conn: sqlite3.Connection, key: str):
        row = conn.execute("SELECT value FROM platform_settings WHERE key=?", (key,)).fetchone()
        return json.loads(row["value"]) if row else DEFAULT_SETTINGS[key]


# -- content blobs -----------------------------------------------------------
def put_content(conn: sqlite3.Connection, data: bytes) -> int:
    """Store bytes once by SHA-256 of the uncompressed content; verify on hit."""
    digest = hashlib.sha256(data).digest()
    row = conn.execute("SELECT content_id, codec, data FROM content.contents WHERE sha256=?",
                       (digest,)).fetchone()
    if row is not None:
        if _decode(row["codec"], row["data"]) != data:
            raise ValueError("content hash collision or corruption")
        return row["content_id"]
    packed = zlib.compress(data, 6)
    codec, stored = ("zlib", packed) if len(packed) < len(data) else ("raw", data)
    cur = conn.execute("INSERT INTO content.contents(sha256, codec, raw_size, data, created_at) "
                       "VALUES(?,?,?,?,?)", (digest, codec, len(data), stored, now_ms()))
    return cur.lastrowid


def get_content(conn: sqlite3.Connection, content_id: int) -> bytes | None:
    row = conn.execute("SELECT sha256, codec, data FROM content.contents WHERE content_id=?",
                       (content_id,)).fetchone()
    if row is None:
        return None
    data = _decode(row["codec"], row["data"])
    if hashlib.sha256(data).digest() != row["sha256"]:
        raise ValueError(f"content {content_id} failed integrity check")
    return data


def _decode(codec: str, data: bytes) -> bytes:
    if codec == "zlib":
        return zlib.decompress(data)
    if codec == "raw":
        return bytes(data)
    raise ValueError(f"unknown codec {codec}")


def canonical_json(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()

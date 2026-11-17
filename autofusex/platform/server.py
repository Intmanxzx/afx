"""HTTP API for the platform (design 5). Loopback upstream behind a TLS proxy.

Every route requires a bearer token. Responses use the envelope
{"schema_version":1,"ok":true,"data":...} or {"ok":false,"error":{"code","message"}}.
"""
from __future__ import annotations

import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

from . import queries
from .auth import AuthError, Principal, authenticate
from .ingest import Ingestor, UploadError, check_stats, update_settings
from .store import Store

MAX_UPLOAD_BODY = 2 * 1024 * 1024
MAX_BODY = 64 * 1024
MAX_CONCURRENT = 16
MAX_AGGREGATE_CONCURRENT = 4
_PARAM = re.compile(r"^[A-Za-z0-9_.:/+<>=-]{1,256}$")


class RateLimiter:
    """Token bucket per (principal, class)."""

    def __init__(self, per_minute: int):
        self.rate = per_minute / 60.0
        self.capacity = float(per_minute)
        self.buckets: dict = {}
        self.lock = threading.Lock()

    def allow(self, key) -> bool:
        now = time.monotonic()
        with self.lock:
            tokens, last = self.buckets.get(key, (self.capacity, now))
            tokens = min(self.capacity, tokens + (now - last) * self.rate)
            if tokens < 1:
                self.buckets[key] = (tokens, now)
                return False
            self.buckets[key] = (tokens - 1, now)
            if len(self.buckets) > 10_000:
                self.buckets.clear()
            return True


class HTTPError(Exception):
    def __init__(self, status: int, code: str, message: str, headers=None):
        super().__init__(message)
        self.status, self.code, self.headers = status, code, headers or {}


class PlatformServer(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 128

    def __init__(self, store: Store, address):
        self.store = store
        self.ingestor = Ingestor(store)
        self.slots = threading.BoundedSemaphore(MAX_CONCURRENT)
        self.aggregate_slots = threading.BoundedSemaphore(MAX_AGGREGATE_CONCURRENT)
        self.read_limit = RateLimiter(600)
        self.upload_limit = RateLimiter(1200)
        super().__init__(address, Handler)


def _param(params, name, *, required=False, integer=False):
    values = params.get(name)
    if not values:
        if required:
            raise HTTPError(400, "invalid", f"{name} is required")
        return None
    value = values[0]
    if not _PARAM.match(value):
        raise HTTPError(400, "invalid", f"{name} has invalid characters or length")
    if integer:
        try:
            return int(value)
        except ValueError:
            raise HTTPError(400, "invalid", f"{name} must be an integer") from None
    return value


class Handler(BaseHTTPRequestHandler):
    server: PlatformServer
    protocol_version = "HTTP/1.1"
    timeout = 30

    def log_message(self, fmt, *args):  # keep stdout quiet; proxies log access
        pass

    # -- plumbing ---------------------------------------------------------------
    def _send(self, status: int, payload: dict, headers=None) -> None:
        body = queries.as_json(payload)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-store")
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def _body(self, limit: int) -> bytes:
        length = self.headers.get("Content-Length")
        if length is None or not length.isdigit():
            raise HTTPError(411, "length_required", "Content-Length is required")
        if int(length) > limit:
            raise HTTPError(413, "too_large", f"body exceeds {limit} bytes")
        if (self.headers.get("Content-Type") or "").split(";")[0].strip() != "application/json":
            raise HTTPError(415, "unsupported_media_type", "Content-Type must be application/json")
        return self.rfile.read(int(length))

    def _json(self, raw: bytes):
        try:
            return json.loads(raw)
        except (ValueError, UnicodeDecodeError):
            raise HTTPError(400, "invalid_json", "body is not valid JSON") from None

    def _dispatch(self, method: str) -> None:
        if not self.server.slots.acquire(blocking=False):
            self._send(503, _error("busy", "server busy"), {"Retry-After": "2"})
            return
        try:
            url = urlsplit(self.path)
            params = parse_qs(url.query, max_num_fields=16)
            try:
                principal = authenticate(self.server.store, self.headers.get("Authorization"))
            except AuthError as exc:
                raise HTTPError(401, "unauthorized", str(exc)) from None
            data, status = self._route(method, url.path, params, principal)
            self._send(status, {"schema_version": 1, "ok": True, "data": data})
        except HTTPError as exc:
            self._send(exc.status, _error(exc.code, str(exc)), exc.headers)
        except AuthError as exc:  # authenticated but missing a scope
            self._send(403, _error("forbidden", str(exc)))
        except UploadError as exc:
            self._send(exc.status, _error(exc.code, str(exc)))
        except queries.QueryError as exc:
            self._send(400, _error("invalid", str(exc)))
        except Exception:  # never leak internals
            self._send(500, _error("internal", "internal error"))
        finally:
            self.server.slots.release()

    def do_GET(self):
        self._dispatch("GET")

    def do_POST(self):
        self._dispatch("POST")

    def do_PUT(self):
        self._dispatch("PUT")

    # -- routes -----------------------------------------------------------------
    def _route(self, method: str, path: str, params, principal: Principal):
        store = self.server.store
        if method == "POST" and path == "/api/client/executions":
            if not self.server.upload_limit.allow(principal.token_id):
                raise HTTPError(429, "rate_limited", "too many uploads", {"Retry-After": "5"})
            raw = self._body(MAX_UPLOAD_BODY)
            return self.server.ingestor.upload(principal, self._json(raw), raw), 200
        if not self.server.read_limit.allow(principal.token_id):
            raise HTTPError(429, "rate_limited", "too many requests", {"Retry-After": "5"})
        if method == "GET":
            if path == "/api/me":
                return {"user_id": principal.user_id, "role": principal.role, "kind": principal.kind,
                        "scopes": sorted(principal.scopes)}, 200
            if path.startswith("/api/overview/"):
                return self._aggregate(path, params), 200
            if path == "/api/cases":
                return queries.case_page(
                    store, after=_param(params, "after", integer=True),
                    limit=_param(params, "limit", integer=True) or 50, backend=_param(params, "backend"),
                    status=_param(params, "status"), arch=_param(params, "arch"),
                    case_key=_param(params, "case_key")), 200
            if path.startswith("/api/executions/"):
                exec_uuid = path[len("/api/executions/"):]
                if not _PARAM.match(exec_uuid):
                    raise HTTPError(400, "invalid", "invalid execution id")
                detail = queries.execution_detail(store, exec_uuid)
                if detail is None:
                    raise HTTPError(404, "not_found", "execution not found")
                return detail, 200
            if path == "/api/clients":
                return queries.clients(store), 200
            if path == "/api/settings":
                return queries.settings(store), 200
        if method == "PUT" and path == "/api/admin/settings":
            principal.require("admin")
            body = self._json(self._body(MAX_BODY))
            if not isinstance(body, dict) or not isinstance(body.get("changes"), dict) \
                    or not isinstance(body.get("base_revisions"), dict):
                raise HTTPError(400, "invalid", "changes and base_revisions objects are required")
            return update_settings(store, principal.user_id, body["changes"], body["base_revisions"]), 200
        raise HTTPError(404, "not_found", "no such route")

    def _aggregate(self, path, params):
        store = self.server.store
        if not self.server.aggregate_slots.acquire(timeout=5):
            raise HTTPError(503, "busy", "aggregate queries busy", {"Retry-After": "2"})
        try:
            arch = _param(params, "arch")
            dim = _param(params, "dim") or "all"
            if path == "/api/overview/pipeline":
                return queries.pipeline(store, arch)
            if path == "/api/overview/backend":
                return queries.backend_stats(store, _param(params, "backend", required=True), dim, arch)
            if path == "/api/overview/perf":
                return queries.perf_pair(store, _param(params, "x", required=True),
                                         _param(params, "y", required=True), dim, arch)
            if path == "/api/overview/dims":
                return queries.dims(store)
            raise HTTPError(404, "not_found", "no such route")
        finally:
            self.server.aggregate_slots.release()


def _error(code: str, message: str) -> dict:
    return {"schema_version": 1, "ok": False, "error": {"code": code, "message": message}}


def create_server(data_dir, host: str = "127.0.0.1", port: int = 8100) -> PlatformServer:
    store = Store(data_dir)
    check_stats(store)
    return PlatformServer(store, (host, port))

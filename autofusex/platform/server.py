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
from urllib.parse import parse_qs, unquote, urlsplit

from importlib import resources

from . import issues, knowledge, ops, profile, queries, regression, retests
from .issues import IssueError
from .search import SearchError
from .auth import ANONYMOUS, AuthError, Principal, authenticate
from .ingest import Ingestor, UploadError, check_stats, update_settings
from .store import Store

MAX_UPLOAD_BODY = 2 * 1024 * 1024
MAX_BODY = 64 * 1024
MAX_HEARTBEAT_BODY = 1024 * 1024
MAX_CONCURRENT = 16
MAX_AGGREGATE_CONCURRENT = 4
_PARAM = re.compile(r"^[A-Za-z0-9_.:/+<>=-]{1,256}$")
_TEXT_PARAM = re.compile(r"^[^\x00-\x1f\x7f]{1,128}$")
_ISSUE_PATH = re.compile(r"^/api/issues/(ISS-[FAP]-\d{6})(?:/(\w[\w-]*))?$")
_ADMIN_ISSUE_PATH = re.compile(r"^/api/admin/issues/(ISS-[FAP]-\d{6})/(close|merge)$")
_ROUND_PATH = re.compile(r"^/api/issues/(ISS-[FAP]-\d{6})/rounds/(\d{1,6})/tasks$")
_TASK_PATH = re.compile(r"^/api/client/retests/(\d{1,12})/complete$")
_KNOWLEDGE_PATH = re.compile(r"^/api/knowledge/(KB-\d{6})$")
_ADMIN_KNOWLEDGE_PATH = re.compile(r"^/api/admin/knowledge/(KB-\d{6})/status$")
_PIN_PATH = re.compile(r"^/api/regression/pins/(\d{1,12})$")
_CASE_PATH = re.compile(r"^/api/cases/([^/]{1,600})/(history|spec)$")
_CASE_KEY = re.compile(r"^(?:[EF][0-9A-Z]{26}|[BM]:[^\x00-\x1f]{1,256})$")


STATIC_TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
                ".css": "text/css; charset=utf-8", ".svg": "image/svg+xml"}
CSP = ("default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; "
       "font-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'; object-src 'none'")
_STATIC_NAME = re.compile(r"^[a-z0-9_-]{1,40}\.(html|js|css|svg)$")


def _static(name: str):
    if not _STATIC_NAME.match(name):
        return None
    try:
        data = resources.files("autofusex.platform").joinpath("static", name).read_bytes()
    except (FileNotFoundError, OSError):
        return None
    return data, STATIC_TYPES["." + name.rsplit(".", 1)[1]]


_INDEX_CACHE: dict = {}


def _versioned_index(html: bytes) -> bytes:
    """Reference assets by content hash so CDN/browser caches can never serve a stale script."""
    import hashlib
    key = hashlib.sha256(html).hexdigest()
    if key not in _INDEX_CACHE:
        out = html
        for asset in ("app.js", "app.css", "favicon.svg"):
            found = _static(asset)
            if found:
                digest = hashlib.sha256(found[0]).hexdigest()[:12]
                out = out.replace(f'"{asset}"'.encode(), f'"{asset}?v={digest}"'.encode())
        _INDEX_CACHE.clear()
        _INDEX_CACHE[key] = out
    return _INDEX_CACHE[key]


class RawBody:
    def __init__(self, data: bytes, content_type: str):
        self.data, self.content_type = data, content_type


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


def _text(params, name):
    values = params.get(name)
    if not values:
        return None
    if not _TEXT_PARAM.match(values[0]):
        raise HTTPError(400, "invalid", f"{name} must be 1..128 printable chars")
    return values[0]


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

    def _drain(self) -> None:
        """Consume the request body up front so no response can leave bytes on a reused connection.

        A reverse proxy reuses upstream connections: an unread body would be parsed as the
        next request. Bodies that are missing a length or exceed the upload cap close the
        connection instead of being read.
        """
        self._raw = b""
        if self.headers.get("Transfer-Encoding"):
            self.close_connection = True
            raise HTTPError(411, "length_required", "chunked bodies are not accepted; send Content-Length")
        length = self.headers.get("Content-Length")
        if length is None:
            return
        if not length.isdigit():
            self.close_connection = True
            raise HTTPError(400, "invalid", "invalid Content-Length")
        if int(length) > MAX_UPLOAD_BODY:
            self.close_connection = True
            raise HTTPError(413, "too_large", f"body exceeds {MAX_UPLOAD_BODY} bytes")
        self._raw = self.rfile.read(int(length))

    def _body(self, limit: int) -> bytes:
        if self.headers.get("Content-Length") is None:
            raise HTTPError(411, "length_required", "Content-Length is required")
        if len(self._raw) > limit:
            raise HTTPError(413, "too_large", f"body exceeds {limit} bytes")
        if (self.headers.get("Content-Type") or "").split(";")[0].strip() != "application/json":
            raise HTTPError(415, "unsupported_media_type", "Content-Type must be application/json")
        return self._raw

    def _json(self, raw: bytes):
        try:
            return json.loads(raw)
        except (ValueError, UnicodeDecodeError):
            raise HTTPError(400, "invalid_json", "body is not valid JSON") from None

    def _dispatch(self, method: str) -> None:
        self._raw = b""
        if not self.server.slots.acquire(blocking=False):
            self.close_connection = True
            self._send(503, _error("busy", "server busy"), {"Retry-After": "2"})
            return
        try:
            self._drain()
            url = urlsplit(self.path)
            if method == "GET" and not url.path.startswith("/api/"):
                self._serve_static(url.path)
                return
            params = parse_qs(url.query, max_num_fields=16)
            header = self.headers.get("Authorization")
            if header is None and method == "GET" and url.path != "/api/me":
                principal = ANONYMOUS  # browsing is public; changes and commands need a token
            else:
                try:
                    principal = authenticate(self.server.store, header)
                except AuthError as exc:
                    raise HTTPError(401, "unauthorized", str(exc)) from None
            try:
                data, status = self._route(method, url.path, params, principal)
            except AuthError as exc:
                if principal.anonymous:
                    raise HTTPError(401, "unauthorized", str(exc)) from None
                raise
            if isinstance(data, RawBody):
                self._send_raw(status, data)
            else:
                self._send(status, {"schema_version": 1, "ok": True, "data": data})
        except HTTPError as exc:
            self._send(exc.status, _error(exc.code, str(exc)), exc.headers)
        except AuthError as exc:  # authenticated but missing a scope
            self._send(403, _error("forbidden", str(exc)))
        except (UploadError, IssueError, SearchError) as exc:
            self._send(exc.status, _error(exc.code, str(exc)))
        except queries.QueryError as exc:
            self._send(400, _error("invalid", str(exc)))
        except Exception:  # never leak internals
            self._send(500, _error("internal", "internal error"))
        finally:
            self.server.slots.release()

    def _rate_key(self, principal: Principal):
        if not principal.anonymous:
            return principal.token_id
        peer = self.client_address[0]
        forwarded = self.headers.get("X-Forwarded-For")
        # Only trust the proxy header when the request comes from the local reverse proxy.
        if forwarded and peer in ("127.0.0.1", "::1"):
            peer = forwarded.split(",")[0].strip()[:64]
        return "ip:" + peer

    def do_GET(self):
        self._dispatch("GET")

    def do_POST(self):
        self._dispatch("POST")

    def do_PUT(self):
        self._dispatch("PUT")

    def do_DELETE(self):
        self._dispatch("DELETE")

    def _serve_static(self, path: str) -> None:
        if path == "/healthz":
            self._send(200, {"schema_version": 1, "ok": True, "data": {"status": "up"}})
            return
        name = "index.html" if path in ("/", "/index.html") else path.rsplit("/", 1)[-1]
        found = _static(name) if path.count("/") <= 2 else None
        if found is None:
            self._send(404, _error("not_found", "no such page"))
            return
        data, content_type = found
        if name == "index.html":
            data = _versioned_index(data)
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Content-Security-Policy", CSP)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        # HTML must always be revalidated; hashed assets may be cached forever.
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate" if name == "index.html"
                         else "public, max-age=31536000, immutable")
        self.end_headers()
        self.wfile.write(data)

    def _send_raw(self, status: int, raw: RawBody) -> None:
        self.send_response(status)
        self.send_header("Content-Type", raw.content_type)
        self.send_header("Content-Length", str(len(raw.data)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw.data)

    # -- routes -----------------------------------------------------------------
    def _route(self, method: str, path: str, params, principal: Principal):
        store = self.server.store
        if method == "POST" and path == "/api/client/executions":
            if not self.server.upload_limit.allow(principal.token_id):
                raise HTTPError(429, "rate_limited", "too many uploads", {"Retry-After": "5"})
            raw = self._body(MAX_UPLOAD_BODY)
            return self.server.ingestor.upload(principal, self._json(raw), raw), 200
        if not self.server.read_limit.allow(self._rate_key(principal)):
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
                    case_key=_param(params, "case_key"), case_id=_param(params, "case_id", integer=True),
                    q=_text(params, "q")), 200
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
            routed = self._get_lifecycle(path, params, principal)
            if routed is not None:
                return routed, 200
        if method == "POST" and path.startswith("/api/") and path != "/api/client/executions":
            routed = self._post_lifecycle(path, principal)
            if routed is not None:
                return routed, 200
        if method == "DELETE":
            match = _PIN_PATH.match(path)
            if match:
                return regression.revoke_pin(store, principal, int(match.group(1)), self._json_body()), 200
        if method == "PUT" and path == "/api/admin/settings":
            principal.require("admin")
            body = self._json(self._body(MAX_BODY))
            if not isinstance(body, dict) or not isinstance(body.get("changes"), dict) \
                    or not isinstance(body.get("base_revisions"), dict):
                raise HTTPError(400, "invalid", "changes and base_revisions objects are required")
            return update_settings(store, principal.user_id, body["changes"], body["base_revisions"]), 200
        raise HTTPError(404, "not_found", "no such route")

    def _json_body(self) -> dict:
        body = self._json(self._body(MAX_BODY))
        if not isinstance(body, dict):
            raise HTTPError(400, "invalid", "body must be a JSON object")
        return body

    def _get_lifecycle(self, path, params, principal):
        store = self.server.store
        if path == "/api/issues":
            return issues.list_issues(store, dimension=_param(params, "dimension"), backend=_param(params, "backend"),
                                      status=_param(params, "status"), after=_param(params, "after", integer=True),
                                      limit=_param(params, "limit", integer=True) or 50)
        match = _ROUND_PATH.match(path)
        if match:
            return retests.round_tasks(store, match.group(1), int(match.group(2)))
        match = _ISSUE_PATH.match(path)
        if match:
            issue_id, sub = match.groups()
            if sub is None:
                return issues.issue_detail(store, issue_id)
            if sub == "cases":
                return issues.issue_cases(store, issue_id, _param(params, "role") or "affected",
                                          _param(params, "after", integer=True))
            return None
        if path == "/api/ops/summary":
            return ops.summary(store)
        if path == "/api/ops/efficiency":
            return ops.efficiency(store, _param(params, "client_id", integer=True))
        match = _CASE_PATH.match(path)
        if match:
            key = unquote(match.group(1))
            if not _CASE_KEY.match(key):
                raise HTTPError(400, "invalid", "invalid case key")
            if match.group(2) == "history":
                return queries.case_history(store, key, after=_param(params, "after", integer=True),
                                            limit=_param(params, "limit", integer=True) or 50)
            return queries.case_spec(store, key)
        if path == "/api/retests":
            return retests.list_manual(store, principal)
        if path == "/api/profile/axes":
            return profile.axes(store)
        if path in ("/api/profile/points", "/api/profile/patterns"):
            return self._aggregate(path, params)
        if path.startswith("/api/models/jobs"):
            ops.not_implemented("model analysis")
        if path == "/api/regression/preview":
            return regression.preview(store, _param(params, "issue_id", required=True))
        if path == "/api/regression/pins":
            return regression.list_pins(store, _param(params, "scope"))
        if path == "/api/knowledge/search":
            return knowledge.search(store.reader(), q=_text(params, "q"), kind=_param(params, "kind"),
                                    op=_param(params, "op"), dtype=_param(params, "dtype"),
                                    backend=_param(params, "backend"), confidence=_param(params, "confidence"),
                                    include_unpublished=principal.role == "admin",
                                    after=_param(params, "after", integer=True),
                                    limit=_param(params, "limit", integer=True) or 20)
        if path == "/api/knowledge/export":
            data = "".join(knowledge.export_jsonl(store.reader())).encode()
            return RawBody(data, "application/x-ndjson; charset=utf-8")
        match = _KNOWLEDGE_PATH.match(path)
        if match:
            return knowledge.get(store.reader(), match.group(1), include_unpublished=principal.role == "admin")
        return None

    def _post_lifecycle(self, path, principal):
        store = self.server.store
        if path == "/api/issues/claim":
            return issues.claim(store, principal, self._json_body())
        if path == "/api/client/heartbeat":
            # carries run snapshots and finished efficiency windows, larger than ordinary writes
            body = self._json(self._body(MAX_HEARTBEAT_BODY))
            if not isinstance(body, dict):
                raise HTTPError(400, "invalid", "body must be a JSON object")
            return ops.heartbeat(store, principal, body)
        if path == "/api/admin/clients/update":
            return ops.request_update(store, principal, self._json_body())
        if path == "/api/retests":
            return retests.create_manual(store, principal, self._json_body())
        if path == "/api/admin/profile/drafts":
            return profile.create_drafts(store, principal, self._json_body())
        if path == "/api/import/baize":
            ops.not_implemented("Baize case import")
        if path == "/api/models/jobs":
            ops.not_implemented("model analysis")
        match = _ISSUE_PATH.match(path)
        if match and match.group(2) is not None:
            handler = {"renew": issues.renew, "release": issues.release, "verify": issues.verify,
                       "regression-cases": issues.pin_cases, "knowledge": issues.submit_knowledge}.get(match.group(2))
            if handler is not None:
                return handler(store, principal, match.group(1), self._json_body())
            return None
        match = _ADMIN_ISSUE_PATH.match(path)
        if match:
            handler = issues.close if match.group(2) == "close" else issues.merge
            return handler(store, principal, match.group(1), self._json_body())
        if path == "/api/client/retests/claim":
            return retests.claim(store, principal, self._json_body())
        match = _TASK_PATH.match(path)
        if match:
            return retests.complete(store, principal, int(match.group(1)), self._json_body())
        if path == "/api/regression/pins":
            return regression.add_pins(store, principal, self._json_body())
        if path == "/api/knowledge":
            return knowledge.propose(store, principal, self._json_body())
        match = _ADMIN_KNOWLEDGE_PATH.match(path)
        if match:
            return knowledge.set_status(store, principal, match.group(1), self._json_body())
        return None

    def _aggregate(self, path, params):
        store = self.server.store
        if not self.server.aggregate_slots.acquire(timeout=5):
            raise HTTPError(503, "busy", "aggregate queries busy", {"Retry-After": "2"})
        try:
            arch = _param(params, "arch")
            dim = _param(params, "dim") or "all"
            if path == "/api/overview/pipeline":
                return queries.pipeline(store, arch, _text(params, "q"))
            if path == "/api/overview/backend":
                return queries.backend_stats(store, _param(params, "backend", required=True), dim, arch,
                                             _text(params, "q"))
            if path == "/api/profile/points":
                return profile.points(store, _param(params, "x", required=True), _param(params, "y", required=True),
                                      _param(params, "fx", required=True), _param(params, "fy", required=True), arch)
            if path == "/api/profile/patterns":
                return profile.patterns(store, _param(params, "x", required=True),
                                        _param(params, "y", required=True), arch)
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

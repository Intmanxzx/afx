"""FastAPI HTTP transport for the AutofuseX platform."""
from __future__ import annotations

import re
import json
import inspect
import socket
import threading
import time
from importlib import resources
from typing import Optional
from typing_extensions import Annotated

import uvicorn
from fastapi import FastAPI, Request, Depends, Query, Path as PathParam
from fastapi.exceptions import RequestValidationError
from fastapi.security import HTTPBearer
from fastapi.responses import Response, JSONResponse
from starlette.exceptions import HTTPException
from starlette.concurrency import run_in_threadpool

from autofusex.server import accounts
from autofusex.server import issues
from autofusex.server import knowledge
from autofusex.server import ops
from autofusex.server import profile
from autofusex.server import queries
from autofusex.server import regression
from autofusex.server import retests
from autofusex.server import search
from autofusex.server import api_models as m
from autofusex.server.auth import ANONYMOUS, AuthError, authenticate
from autofusex.server.ingest import Ingestor, UploadError, check_stats, update_settings
from autofusex.server.store import Store

MAX_UPLOAD_BODY = 2 * 1024 * 1024
MAX_BODY = 64 * 1024
MAX_HEARTBEAT_BODY = 1024 * 1024
MAX_CONCURRENT = 16
MAX_AGGREGATE_CONCURRENT = 4

QueryValue = Annotated[Optional[str], Query(pattern=r"^[A-Za-z0-9_.:/+<>=-]{1,256}$")]
QueryText = Annotated[Optional[str], Query(min_length=1, max_length=128, pattern=r"^[^\x00-\x1f\x7f]+$")]
RequiredValue = Annotated[str, Query(pattern=r"^[A-Za-z0-9_.:/+<>=-]{1,256}$")]


STATIC_TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
                ".css": "text/css; charset=utf-8", ".svg": "image/svg+xml"}
CSP = ("default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; "
       "font-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'; object-src 'none'")
_STATIC_NAME = re.compile(r"^[a-z0-9_-]{1,40}\.(html|js|css|svg)$")


def _static(name: str):
    if not _STATIC_NAME.match(name):
        return None
    try:
        data = resources.files("autofusex.server").joinpath("static", name).read_bytes()
    except (FileNotFoundError, OSError):
        return None
    return data, STATIC_TYPES["." + name.rsplit(".", 1)[1]]


_INDEX_CACHE: dict = {}
# The only static files index.html references by a literal name that _versioned_index rewrites
# to a content-hashed query string; only these may be cached as immutable (see _serve_static).
# app.js imports its submodules (router.js, page-*.js, ...) by plain relative path from inside
# its own source text, which this function never touches, so those files keep a stable URL and
# must stay revalidate-on-every-request or a browser could run stale code after a deploy.
_HASHED_ASSETS = ("bootstrap.js", "app.css", "favicon.svg")


def _versioned_index(html: bytes) -> bytes:
    """Reference assets by content hash so CDN/browser caches can never serve a stale script."""
    import hashlib
    key = hashlib.sha256(html).hexdigest()
    if key not in _INDEX_CACHE:
        out = html
        for asset in _HASHED_ASSETS:
            found = _static(asset)
            if found:
                digest = hashlib.sha256(found[0]).hexdigest()[:12]
                out = out.replace(f'"{asset}"'.encode(), f'"{asset}?v={digest}"'.encode())
        _INDEX_CACHE.clear()
        _INDEX_CACHE[key] = out
    return _INDEX_CACHE[key]


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



def _error(code, message):
    return {'schema_version': 1, 'ok': False, 'error': {'code': code, 'message': message}}


def _success(data):
    return {'schema_version': 1, 'ok': True, 'data': data}


def _rate_key(request, principal):
    if not principal.anonymous:
        return principal.token_id
    peer = request.client.host if request.client else ''
    forwarded = request.headers.get('X-Forwarded-For')
    if forwarded and peer in ('127.0.0.1', '::1'):
        peer = forwarded.split(',')[0].strip()[:64]
    return 'ip:' + peer


bearer = HTTPBearer(auto_error=False)


def identity(request: Request, credentials=Depends(bearer)):
    header = request.headers.get('Authorization')
    if header is None:
        return ANONYMOUS
    try:
        return authenticate(request.app.state.store, header)
    except AuthError as exc:
        raise HTTPError(401, 'unauthorized', str(exc)) from None


def writer(principal=Depends(identity)):
    if principal.anonymous:
        raise HTTPError(401, 'unauthorized', 'Bearer token required')
    principal.require_write()
    return principal


def administrator(principal=Depends(writer)):
    principal.require_admin()
    return principal


def data(body):
    # Preserve omission vs explicit null and unknown extension fields for idempotency.
    return body.model_dump(exclude_unset=True)


def create_app(store: Store) -> FastAPI:
    app = FastAPI(title='AutofuseX Platform API', version='1', docs_url=None, redoc_url=None,
                  redirect_slashes=False)
    app.state.store = store
    ingestor = Ingestor(store)
    slots = threading.BoundedSemaphore(MAX_CONCURRENT)
    aggregate_slots = threading.BoundedSemaphore(MAX_AGGREGATE_CONCURRENT)
    read_limit, upload_limit = RateLimiter(600), RateLimiter(1200)
    application_limit, application_global_limit, receipt_limit = RateLimiter(3), RateLimiter(30), RateLimiter(60)

    @app.middleware('http')
    async def transport(request: Request, call_next):
        if not slots.acquire(blocking=False):
            return JSONResponse(_error('busy', 'server busy'), 503, headers={'Retry-After': '2'})
        try:
            length = request.headers.get('Content-Length')
            if request.headers.get('Transfer-Encoding'):
                raise HTTPError(411, 'length_required', 'chunked bodies are not accepted; send Content-Length')
            if length is not None and not length.isdigit():
                raise HTTPError(400, 'invalid', 'invalid Content-Length')
            path = request.url.path
            limit = (MAX_UPLOAD_BODY if path in {'/api/client/executions','/api/source-cases','/api/source-results'} or path.startswith('/api/source-jobs/') else
                     MAX_HEARTBEAT_BODY if path == '/api/client/heartbeat' else MAX_BODY)
            if length is not None and int(length) > limit:
                raise HTTPError(413, 'too_large', f'body exceeds {limit} bytes')
            raw = bytearray()
            async for chunk in request.stream():
                raw.extend(chunk)
                if len(raw) > limit:
                    raise HTTPError(413, 'too_large', f'body exceeds {limit} bytes')
            request._body = bytes(raw)
            request.state.raw = bytes(raw)
            if path.startswith('/api/'):
                public_application = request.method == 'POST' and path in (
                    '/api/registration/apply', '/api/registration/status', '/api/registration/claim')
                if public_application:
                    principal = ANONYMOUS
                else:
                    principal = await run_in_threadpool(identity, request, None)
                    if principal.anonymous and (request.method != 'GET' or path == '/api/me'):
                        raise HTTPError(401, 'unauthorized', 'Bearer token required')
                limiter = upload_limit if path == '/api/client/executions' else read_limit
                if not limiter.allow(_rate_key(request, principal)):
                    raise HTTPError(429, 'rate_limited', 'too many requests', {'Retry-After': '5'})
                needs_body = request.method in ('POST', 'PUT') or (
                    request.method == 'DELETE' and path.startswith('/api/regression/pins/'))
                if needs_body:
                    if length is None:
                        raise HTTPError(411, 'length_required', 'Content-Length is required')
                    if request.headers.get('Content-Type', '').split(';')[0].strip() != 'application/json':
                        raise HTTPError(415, 'unsupported_media_type', 'Content-Type must be application/json')
            from autofusex.server.client_access import allowed, protected, read as access_policy
            if protected(path, request.method):
                try:
                    policy = access_policy(store.dir)
                    if not policy['enabled']:
                        admitted = True
                    else:
                        body = json.loads(raw) if raw else {}
                        admitted = allowed(store, principal, path, request.method, body if isinstance(body, dict) else {}, policy)
                except (ValueError, OSError, TypeError, AttributeError):
                    admitted = False
                if not admitted:
                    raise HTTPError(503, 'client_paused', 'Client admission is temporarily restricted', {'Retry-After': '60'})
            response = await call_next(request)
        except HTTPError as exc:
            response = JSONResponse(_error(exc.code, str(exc)), exc.status, headers=exc.headers)
        except Exception:
            response = JSONResponse(_error('internal', 'internal error'), 500)
        finally:
            slots.release()
        response.headers.setdefault('X-Content-Type-Options', 'nosniff')
        response.headers.setdefault('Cache-Control', 'no-store')
        return response

    @app.exception_handler(HTTPError)
    async def http_error(request, exc):
        return JSONResponse(_error(exc.code, str(exc)), exc.status, headers=exc.headers)

    @app.exception_handler(AuthError)
    async def auth_error(request, exc):
        return JSONResponse(_error('forbidden', str(exc)), 403)

    @app.exception_handler(accounts.AccountError)
    async def account_error(request, exc):
        return JSONResponse(_error('account_error', str(exc)), exc.status)

    async def business_error(request, exc):
        return JSONResponse(_error(exc.code, str(exc)), exc.status)
    for error_type in (UploadError, issues.IssueError, search.SearchError):
        app.add_exception_handler(error_type, business_error)

    @app.exception_handler(ValueError)
    async def value_error(request, exc):
        return JSONResponse(_error('invalid', str(exc)), 400)

    @app.exception_handler(PermissionError)
    async def permission_error(request, exc):
        return JSONResponse(_error('forbidden', str(exc)), 403)

    @app.exception_handler(LookupError)
    async def lookup_error(request, exc):
        return JSONResponse(_error('not_found', str(exc)), 404)

    @app.exception_handler(queries.QueryError)
    async def query_error(request, exc):
        return JSONResponse(_error('invalid', str(exc)), 400)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        errors = exc.errors()
        code = 'invalid_json' if any(e['type'] == 'json_invalid' for e in errors) else 'invalid'
        message = '; '.join('.'.join(map(str, e['loc'])) + ': ' + e['msg'] for e in errors)
        return JSONResponse(_error(code, message), 400)

    @app.exception_handler(HTTPException)
    async def framework_error(request, exc):
        # Preserve the old transport's 404 for unsupported methods.
        status = 404 if exc.status_code == 405 else exc.status_code
        return JSONResponse(_error('not_found' if status == 404 else 'http_error', str(exc.detail)), status)

    def route(path, method='GET', **kwargs):
        def register(endpoint):
            dependencies = [getattr(p.default, 'dependency', None)
                            for p in inspect.signature(endpoint).parameters.values()]
            extra = {}
            if identity in dependencies:
                extra['security'] = [{}, {'HTTPBearer': []}]
            if administrator in dependencies:
                extra['x-required-role'] = 'admin'
            elif writer in dependencies:
                extra['x-required-role'] = 'developer'
            return app.api_route(path, methods=[method], response_model=m.Success,
                                 responses={400: {'model': m.Error}, 401: {'model': m.Error},
                                            403: {'model': m.Error}, 404: {'model': m.Error},
                                            409: {'model': m.Error}, 429: {'model': m.Error},
                                            500: {'model': m.Error}, 503: {'model': m.Error}},
                                 openapi_extra=extra, **kwargs)(endpoint)
        return register

    def aggregate():
        if not aggregate_slots.acquire(timeout=5):
            raise HTTPError(503, 'busy', 'aggregate queries busy', {'Retry-After': '2'})
        try:
            yield
        finally:
            aggregate_slots.release()

    @route('/api/me')
    def me(principal=Depends(writer)):
        return _success({'user_id': principal.user_id, 'role': principal.role})

    @route('/api/cases')
    def cases(after: Optional[int] = None, limit: int = 50, backend: QueryValue = None,
              status: QueryValue = None, arch: QueryValue = None, case_key: QueryValue = None,
              case_id: Optional[int] = None, q: QueryText = None, principal=Depends(identity)):
        return _success(queries.case_page(store, after=after, limit=limit, backend=backend, status=status,
                                          arch=arch, case_key=case_key, case_id=case_id, q=q))

    @route('/api/cases/{case_key:path}/history')
    def case_history(case_key: str, after: Optional[int] = None, limit: int = 50, principal=Depends(identity)):
        return _success(queries.case_history(store, case_key, after=after, limit=limit))

    @route('/api/cases/{case_key:path}/spec')
    def case_spec(case_key: str, principal=Depends(identity)):
        return _success(queries.case_spec(store, case_key))

    @route('/api/executions/{exec_uuid:path}')
    def execution(exec_uuid: str, principal=Depends(identity)):
        result = queries.execution_detail(store, exec_uuid)
        if result is None:
            raise HTTPError(404, 'not_found', 'execution not found')
        return _success(result)

    @route('/api/clients')
    def clients(principal=Depends(identity)):
        return _success(queries.clients(store))

    @route('/api/settings')
    def settings(principal=Depends(identity)):
        return _success(queries.settings(store))

    @route('/api/search/fields')
    def search_fields(principal=Depends(identity)):
        return _success(search.field_values(store))

    @route('/api/issues')
    def issue_list(dimension: QueryValue = None, backend: QueryValue = None, status: QueryValue = None,
                   q: QueryText = None, after: QueryValue = None, limit: int = 50,
                   principal=Depends(identity)):
        return _success(issues.list_issues(store, dimension=dimension, backend=backend, status=status, q=q, after=after, limit=limit))

    @route('/api/issues/claim', 'POST')
    def claim(body: m.Claim, principal=Depends(writer)):
        return _success(issues.claim(store, principal, data(body)))

    @route('/api/issues/{issue_id}')
    def issue_detail(issue_id: str = PathParam(pattern='^[FAP][0-9]{1,9}$'), principal=Depends(identity)):
        return _success(issues.issue_detail(store, issue_id))

    @route('/api/issues/{issue_id}/cases')
    def issue_cases(issue_id: str, role: RequiredValue = 'affected', after: Optional[int] = None, principal=Depends(identity)):
        return _success(issues.issue_cases(store, issue_id, role, after))

    @route('/api/issues/{issue_id}/rounds/{round_no}/tasks')
    def round_tasks(issue_id: str, round_no: int, principal=Depends(identity)):
        return _success(retests.round_tasks(store, issue_id, round_no))

    @route('/api/issues/{issue_id}/renew', 'POST')
    def renew(issue_id: str, body: m.Lease, principal=Depends(writer)):
        return _success(issues.renew(store, principal, issue_id, data(body)))

    @route('/api/issues/{issue_id}/release', 'POST')
    def release(issue_id: str, body: m.Lease, principal=Depends(writer)):
        return _success(issues.release(store, principal, issue_id, data(body)))

    @route('/api/issues/{issue_id}/verify', 'POST')
    def verify(issue_id: str, body: m.Verify, principal=Depends(writer)):
        return _success(issues.verify(store, principal, issue_id, data(body)))

    @route('/api/issues/{issue_id}/regression-cases', 'POST')
    def issue_pins(issue_id: str, body: m.IssuePins, principal=Depends(writer)):
        return _success(issues.pin_cases(store, principal, issue_id, data(body)))

    @route('/api/issues/{issue_id}/knowledge', 'POST')
    def submit_knowledge(issue_id: str, body: m.SubmitKnowledge, principal=Depends(writer)):
        return _success(issues.submit_knowledge(store, principal, issue_id, data(body)))

    @route('/api/admin/issues/{issue_id}/close', 'POST')
    def close_issue(issue_id: str, body: m.CloseIssue, principal=Depends(administrator)):
        return _success(issues.close(store, principal, issue_id, data(body)))

    @route('/api/admin/issues/{issue_id}/merge', 'POST')
    def merge_issue(issue_id: str, body: m.MergeIssue, principal=Depends(administrator)):
        return _success(issues.merge(store, principal, issue_id, data(body)))

    @route('/api/ops/summary')
    def summary(scope: RequiredValue = 'mine', principal=Depends(identity)):
        return _success(ops.summary(store, principal, scope))

    @route('/api/ops/efficiency')
    def efficiency(client_id: Optional[int] = None, principal=Depends(identity)):
        return _success(ops.efficiency(store, client_id))

    @route('/api/source-cases', 'POST')
    def source_case(body: m.SourceBundle, principal=Depends(writer)):
        from autofusex.server import source_jobs
        return _success(source_jobs.register(store, principal, data(body)))

    @route('/api/source-jobs', 'POST')
    def source_submit(body: m.SourceJob, principal=Depends(writer)):
        from autofusex.server import source_jobs
        return _success(source_jobs.submit(store, principal, data(body)))

    @route('/api/source-jobs/claim', 'POST')
    def source_claim(body: m.SourceClaim, principal=Depends(writer)):
        from autofusex.server import source_jobs
        return _success(source_jobs.claim(store, principal, body.machine_id))

    @route('/api/source-jobs')
    def source_list(principal=Depends(writer)):
        from autofusex.server import source_jobs
        return _success(source_jobs.list_jobs(store, principal))

    @route('/api/source-results', 'POST')
    def source_result(body: m.SourceResult, principal=Depends(writer)):
        from autofusex.server import source_jobs
        return _success(source_jobs.record_result(store, principal, data(body)))

    @route('/api/source-jobs/{job_id}')
    def source_get(job_id: str, principal=Depends(writer)):
        from autofusex.server import source_jobs
        return _success(source_jobs.get(store, principal, job_id))

    @route('/api/source-jobs/{job_id}/artifacts', 'POST')
    def source_artifact_put(job_id: str, body: m.SourceArtifact, principal=Depends(writer)):
        from autofusex.server import source_jobs
        return _success(source_jobs.artifact(store, principal, job_id, data(body)))

    @app.get('/api/source-jobs/{job_id}/artifact')
    def source_artifact_get(job_id: str, path: str, principal=Depends(writer)):
        from autofusex.server import source_jobs
        raw=source_jobs.artifact(store, principal, job_id, path=path)
        return Response(raw, media_type='application/octet-stream', headers={
            'Content-Disposition':'attachment; filename="artifact"',
            'Content-Security-Policy':"default-src 'none'; sandbox"})

    @route('/api/source-jobs/{job_id}/complete', 'POST')
    def source_complete(job_id: str, body: m.SourceReceipt, principal=Depends(writer)):
        from autofusex.server import source_jobs
        return _success(source_jobs.complete(store, principal, job_id, data(body)))

    @route('/api/client/heartbeat', 'POST')
    def heartbeat(body: m.Heartbeat, principal=Depends(writer)):
        return _success(ops.heartbeat(store, principal, data(body)))

    @route('/api/admin/clients/update', 'POST')
    def client_update(body: m.ClientUpdate, principal=Depends(administrator)):
        return _success(ops.request_update(store, principal, data(body)))

    @route('/api/clients/daemon', 'POST')
    def client_daemon(body: m.ClientControl, principal=Depends(writer)):
        return _success(ops.request_daemon_state(store, principal, data(body)))

    @route('/api/admin/clients/{client_id}', 'DELETE')
    def delete_client(client_id: int, principal=Depends(administrator)):
        return _success(ops.delete_client(store, principal, client_id))

    @route('/api/client/executions', 'POST')
    def upload(request: Request, body: m.Upload, principal=Depends(writer)):
        return _success(ingestor.upload(principal, data(body), request.state.raw))

    @route('/api/retests')
    def retest_list(principal=Depends(identity)):
        return _success(retests.list_manual(store, principal))

    @route('/api/retests', 'POST')
    def manual_retest(body: m.ManualRetest, principal=Depends(writer)):
        return _success(retests.create_manual(store, principal, data(body)))

    @route('/api/client/retests/claim', 'POST')
    def claim_retests(body: m.ClaimRetests, principal=Depends(writer)):
        return _success(retests.claim(store, principal, data(body)))

    @route('/api/client/retests/{task_id}/complete', 'POST')
    def complete_retest(task_id: int, body: m.CompleteRetest, principal=Depends(writer)):
        return _success(retests.complete(store, principal, task_id, data(body)))

    @route('/api/regression/preview')
    def preview(issue_id: str, principal=Depends(identity)):
        return _success(regression.preview(store, issue_id))

    @route('/api/regression/pins')
    def pins(scope: QueryValue = None, principal=Depends(identity)):
        return _success(regression.list_pins(store, scope))

    @route('/api/regression/pins', 'POST')
    def add_pins(body: m.Pins, principal=Depends(administrator)):
        return _success(regression.add_pins(store, principal, data(body)))

    @route('/api/regression/pins/{pin_id}', 'DELETE')
    def revoke_pin(pin_id: int, body: m.RequestKey, principal=Depends(administrator)):
        return _success(regression.revoke_pin(store, principal, pin_id, data(body)))

    @route('/api/knowledge/search')
    def knowledge_search(q: QueryText = None, kind: QueryValue = None,
                         op: QueryValue = None, dtype: QueryValue = None, backend: QueryValue = None,
                         confidence: QueryValue = None, after: Optional[int] = None, limit: int = 20,
                         principal=Depends(identity)):
        return _success(knowledge.search(store.reader(), q=q, kind=kind, op=op, dtype=dtype, backend=backend,
                                         confidence=confidence, after=after, limit=limit, include_unpublished=principal.role == 'admin'))

    @app.get('/api/knowledge/export', response_class=Response,
             responses={200: {'content': {'application/x-ndjson': {'schema': {'type': 'string'}}}}})
    def knowledge_export(principal=Depends(identity)):
        return Response(''.join(knowledge.export_jsonl(store.reader())), media_type='application/x-ndjson')

    @route('/api/knowledge/{knowledge_id}')
    def knowledge_item(knowledge_id: str, principal=Depends(identity)):
        return _success(knowledge.get(store.reader(), knowledge_id, include_unpublished=principal.role == 'admin'))

    @route('/api/knowledge', 'POST')
    def propose_knowledge(body: m.ProposeKnowledge, principal=Depends(writer)):
        return _success(knowledge.propose(store, principal, data(body)))

    @route('/api/admin/knowledge/{knowledge_id}/status', 'POST')
    def knowledge_status(knowledge_id: str, body: m.KnowledgeStatus, principal=Depends(administrator)):
        return _success(knowledge.set_status(store, principal, knowledge_id, data(body)))

    @route('/api/admin/settings', 'PUT')
    def change_settings(body: m.Settings, principal=Depends(administrator)):
        return _success(update_settings(store, principal.user_id, body.changes, body.base_revisions))

    @route('/api/config/operators')
    def operators(principal=Depends(writer)):
        from autofusex.workloads.fuzz.nnsmith.aten_catalog import RULES
        return _success({'items': [{'id': r.target, 'name': r.display_name, 'category': r.category} for r in RULES]})

    @route('/api/config/me')
    def config_me(principal=Depends(writer)):
        from autofusex.server import task_config
        return _success(task_config.get(store, principal))

    @route('/api/config/default')
    def config_default(principal=Depends(administrator)):
        from autofusex.server import task_config
        return _success(task_config.get(store, principal, administrator=True))

    @route('/api/config/me', 'POST')
    def put_config_me(body: m.Configuration, principal=Depends(writer)):
        from autofusex.server import task_config
        return _success(task_config.put(store, principal, data(body)))

    @route('/api/config/default', 'POST')
    def put_config_default(body: m.Configuration, principal=Depends(administrator)):
        from autofusex.server import task_config
        return _success(task_config.put(store, principal, data(body), administrator=True))

    @route('/api/registration/apply', 'POST')
    def apply(request: Request, body: m.Application):
        if not application_global_limit.allow('all') or not application_limit.allow(_rate_key(request, ANONYMOUS)):
            raise HTTPError(429, 'rate_limit', '申请过于频繁，请稍后再试')
        return _success(accounts.apply(store, data(body)))

    def receipt_guard(request: Request):
        if not receipt_limit.allow(_rate_key(request, ANONYMOUS)):
            raise HTTPError(429, 'rate_limit', '请求过于频繁，请稍后再试')

    @route('/api/registration/status', 'POST', dependencies=[Depends(receipt_guard)])
    def application_status(body: m.Receipt):
        return _success(accounts.status(store, data(body)))

    @route('/api/registration/claim', 'POST', dependencies=[Depends(receipt_guard)])
    def application_claim(body: m.Receipt):
        return _success(accounts.claim(store, data(body)))

    @route('/api/admin/users')
    def users(principal=Depends(administrator)):
        return _success(accounts.listing(store, principal))

    @route('/api/admin/users/review', 'POST')
    def review(body: m.Review, principal=Depends(administrator)):
        return _success(accounts.review(store, principal, data(body)))

    @route('/api/admin/users/manage', 'POST')
    def manage(body: m.Manage, principal=Depends(administrator)):
        return _success(accounts.manage(store, principal, data(body)))

    @route('/api/overview/pipeline', dependencies=[Depends(aggregate)])
    def pipeline(arch: QueryValue = None, q: QueryText = None, principal=Depends(identity)):
        return _success(queries.pipeline(store, arch, q))

    @route('/api/overview/backend', dependencies=[Depends(aggregate)])
    def backend(backend: RequiredValue, dim: RequiredValue = 'all', arch: QueryValue = None, q: QueryText = None, principal=Depends(identity)):
        return _success(queries.backend_stats(store, backend, dim, arch, q))

    @route('/api/overview/perf', dependencies=[Depends(aggregate)])
    def perf(x: RequiredValue, y: RequiredValue, dim: RequiredValue = 'all', arch: QueryValue = None, principal=Depends(identity)):
        return _success(queries.perf_pair(store, x, y, dim, arch))

    @route('/api/overview/dims', dependencies=[Depends(aggregate)])
    def dims(principal=Depends(identity)):
        return _success(queries.dims(store))

    @route('/api/overview/matrix', dependencies=[Depends(aggregate)])
    def matrix(backends: str, arch: QueryValue = None, q: QueryText = None, principal=Depends(identity)):
        names = [b for b in backends.split(',') if b][:8]
        if not names or not all(re.fullmatch('[A-Za-z0-9_.:/+<>=-]{1,256}', b) for b in names):
            raise HTTPError(400, 'invalid', 'backends must be a comma separated list of backend names')
        dimensions = queries.dims(store)
        return _success({'dims': dimensions, 'backends': names,
                         'cells': {b: {d: queries.backend_stats(store, b, d, arch, q) for d in dimensions} for b in names}})

    @route('/api/profile/axes')
    def axes(principal=Depends(identity)):
        return _success(profile.axes(store))

    @route('/api/profile/points', dependencies=[Depends(aggregate)])
    def points(x: RequiredValue, y: RequiredValue, fx: RequiredValue, fy: RequiredValue, arch: QueryValue = None, principal=Depends(identity)):
        return _success(profile.points(store, x, y, fx, fy, arch))

    @route('/api/profile/patterns', dependencies=[Depends(aggregate)])
    def patterns(x: RequiredValue, y: RequiredValue, arch: QueryValue = None, principal=Depends(identity)):
        return _success(profile.patterns(store, x, y, arch))

    @route('/api/admin/profile/drafts', 'POST')
    def drafts(body: m.Drafts, principal=Depends(administrator)):
        return _success(profile.create_drafts(store, principal, data(body)))

    @app.get('/api/capabilities', include_in_schema=False)
    def capabilities():
        return {'mode': 'server', 'read_only': False}

    @route('/healthz', include_in_schema=False)
    def health():
        return _success({'status': 'up'})

    @app.get('/', include_in_schema=False)
    @app.get('/{path}', include_in_schema=False)
    @app.get('/static/{path}', include_in_schema=False)
    def static(path: str = ''):
        full = '/' + path
        name = 'index.html' if full in ('/', '/index.html') else full.rsplit('/', 1)[-1]
        found = _static(name) if full.count('/') <= 2 and not full.startswith('/api/') else None
        if found is None:
            raise HTTPError(404, 'not_found', 'no such page')
        content, content_type = found
        if name == 'index.html':
            content = _versioned_index(content)
        return Response(content, headers={
            'Content-Type': content_type, 'Content-Security-Policy': CSP,
            'X-Frame-Options': 'DENY', 'Referrer-Policy': 'no-referrer',
            'Cache-Control': 'public, max-age=31536000, immutable' if name in _HASHED_ASSETS
                             else 'no-cache, no-store, must-revalidate',
        })

    generate_openapi = app.openapi

    def openapi():
        schema = generate_openapi()
        for methods in schema['paths'].values():
            for operation in methods.values():
                if isinstance(operation, dict) and 'responses' in operation:
                    if '422' in operation['responses']:
                        operation['responses'].pop('422')
                        operation['responses']['400'] = {'description': 'Invalid request',
                            'content': {'application/json': {'schema': {'$ref': '#/components/schemas/Error'}}}}
        return schema

    app.openapi = openapi
    return app


class PlatformServer:
    """Uvicorn lifecycle adapter preserving the CLI and embedded-server API."""
    def __init__(self, store: Store, address):
        self.store = store
        self.app = create_app(store)
        self.socket = socket.socket(socket.AF_INET6 if ':' in address[0] else socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind(address)
        self.socket.listen(128)
        self.server_address = self.socket.getsockname()
        self.server_port = self.server_address[1]
        self._server = uvicorn.Server(uvicorn.Config(self.app, log_level='error', access_log=False,
                                                    proxy_headers=False, timeout_keep_alive=30))
        self._finished = threading.Event()
        self._running = False

    def serve_forever(self):
        self._running = True
        try:
            self._server.run(sockets=[self.socket])
        finally:
            self._finished.set()

    def shutdown(self):
        self._server.should_exit = True
        if self._running:
            self._finished.wait(timeout=30)

    def server_close(self):
        self.shutdown()
        self.socket.close()


def create_server(data_dir, host='127.0.0.1', port=8100):
    store = Store(data_dir)
    check_stats(store)
    return PlatformServer(store, (host, port))

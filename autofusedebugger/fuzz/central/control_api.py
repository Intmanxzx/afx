"""Same-origin browser authentication and token-authenticated retest endpoints."""
from collections import OrderedDict
from contextlib import closing
from http.cookies import SimpleCookie, CookieError
import json
import sqlite3
import threading
import time
from urllib.parse import urlsplit, parse_qs

from . import sessions, retests, issue_groups
from .maintenance import WriteAllowlist, MaintenanceBlocked, reject
from .repository import authenticate
from ..storage.repository import _connect


class ControlAPI:
    def __init__(self, database, on_publish=None, write_allowlist=None):
        self.write_allowlist = write_allowlist or WriteAllowlist()
        self.database = database
        self.on_publish = on_publish
        self.login_lock = threading.Lock()
        self.login_attempts = OrderedDict()

    def _login_limit(self, address):
        now = time.monotonic()
        with self.login_lock:
            samples = [t for t in self.login_attempts.pop(address, []) if now-t < 60]
            allowed = len(samples) < 10
            if allowed:
                samples.append(now)
            self.login_attempts[address] = samples
            while len(self.login_attempts) > 1024:
                self.login_attempts.popitem(last=False)
            return allowed

    @staticmethod
    def _secret(handler):
        cookies = SimpleCookie()
        try:
            cookies.load(handler.headers.get('Cookie',''))
        except CookieError:
            raise PermissionError('invalid cookie') from None
        return cookies[sessions.COOKIE].value if sessions.COOKIE in cookies else ''

    @staticmethod
    def _origin(handler):
        # This service binds loopback behind a trusted TLS proxy. The proxy must
        # preserve Host. Never trust a caller-supplied forwarded host/protocol.
        origin = handler.headers.get('Origin','')
        if origin != 'https://' + handler.headers.get('Host',''):
            raise PermissionError('same-origin HTTPS required')
        if handler.headers.get('Sec-Fetch-Site','same-origin') not in {'same-origin','none'}:
            raise PermissionError('cross-site request')

    def handle(self, handler, method):
        if len(handler.path) > 2048:
            handler.send(414, {'error':'URL too long'})
            return True
        url = urlsplit(handler.path)
        route = url.path
        if route != '/api/session' and not (route.startswith('/api/retest/') or route.startswith('/api/issues')):
            return False
        try:
            body = None
            if method == 'POST':
                if handler.headers.get('Transfer-Encoding') or handler.headers.get('Content-Type','').split(';')[0] != 'application/json':
                    raise ValueError('JSON required')
                length = int(handler.headers.get('Content-Length','0'))
                if not 1 <= length <= 65536:
                    handler.send(413, {'error':'bounded body required'})
                    return True
            if route == '/api/session' and method == 'POST':
                if url.query:
                    raise ValueError('unexpected query')
                self._origin(handler)
                # Reverse-proxy clients share this budget deliberately: no
                # spoofable forwarding headers are used as rate-limit keys.
                if not self._login_limit(handler.client_address[0]):
                    handler.send(429, {'error':'登录尝试过于频繁，请稍后重试'}, headers={'Retry-After':'60'})
                    return True
                raw = handler.rfile.read(length)
                if len(raw) != length:
                    raise ValueError('incomplete body')
                body = json.loads(raw)
                if not isinstance(body,dict) or set(body) != {'token'}:
                    raise ValueError('invalid login')
                secret = sessions.issue(self.database, body['token'])
                handler.send(200, sessions.identify(self.database, secret), headers={'Set-Cookie':sessions.cookie(secret)})
                return True
            if route.startswith('/api/retest/history/') and method == 'GET':
                params = parse_qs(url.query, keep_blank_values=True, max_num_fields=1)
                if set(params)-{'after'} or any(len(v)!=1 for v in params.values()):
                    raise ValueError('invalid history query')
                handler.send(200, retests.history(self.database,route.rsplit('/',1)[1],
                    int(params.get('after',[str(2**63-1)])[0])))
                return True
            agent_routes = {'/api/retest/register','/api/retest/claim','/api/retest/renew','/api/retest/complete'}
            if route in agent_routes:
                if method != 'POST' or url.query:
                    handler.send(405, {'error':'POST required'})
                    return True
                auth = handler.headers.get('Authorization','')
                if not auth.startswith('Bearer ') or len(auth) > 256:
                    raise PermissionError('credential required')
                with closing(_connect(self.database)) as conn:
                    user = authenticate(conn, auth[7:])
            else:
                if method != 'GET':
                    self._origin(handler)
                    csrf = handler.headers.get('X-CSRF-Token','')
                    if not csrf:
                        raise PermissionError('CSRF token required')
                else:
                    csrf = None
                profile = sessions.identify(self.database, self._secret(handler), csrf)
                user = profile['user_id']
            if method == 'POST':
                raw = handler.rfile.read(length)
                if len(raw) != length:
                    raise ValueError('incomplete body')
                body = json.loads(raw)
            if route in agent_routes:
                self.write_allowlist.client(user, body.get('client_id') if isinstance(body,dict) else None)
            elif method != 'GET' and route != '/api/session':
                self.write_allowlist.browser(user)
                # Preview/create must target the allowed client too; cancellation
                # remains available to the permitted user and checks ownership.
                if route in {'/api/retest/preview','/api/retest/batches','/api/issues/retest'}:
                    self.write_allowlist.client(user, body.get('client_id') if isinstance(body,dict) else None)
            params = parse_qs(url.query, keep_blank_values=True, max_num_fields=2)
            if any(len(v) != 1 for v in params.values()):
                raise ValueError('duplicate query')
            parts = route.strip('/').split('/')
            batch_detail = len(parts) == 4 and parts[:3] == ['api','retest','batches']
            batch_cancel = len(parts) == 5 and parts[:3] == ['api','retest','batches'] and parts[4] == 'cancel'
            allowed_query = {'before'} if route == '/api/retest/batches' else {'after'} if batch_detail else {'dimension'} if route == '/api/issues' else set()
            if params and (method != 'GET' or set(params) != allowed_query):
                raise ValueError('unexpected query')
            if route == '/api/issues' and method == 'GET':
                result = issue_groups.list_groups(self.database, params.get('dimension', [None])[0])
            elif route == '/api/issues/claim' and method == 'POST':
                if not isinstance(body, dict) or set(body) != {'issue_key'}:
                    raise ValueError('invalid issue claim')
                result = issue_groups.claim(self.database, user, body['issue_key'])
            elif route == '/api/issues/retest' and method == 'POST':
                if not isinstance(body, dict) or set(body) - {'issue_key','client_id','repetitions','request_key'} or 'issue_key' not in body or 'client_id' not in body:
                    raise ValueError('invalid issue retest')
                result = issue_groups.retest(self.database, user, body['issue_key'], body['client_id'], body.get('repetitions', 1), body.get('request_key'))
            elif route == '/api/issues/release' and method == 'POST':
                if not isinstance(body, dict) or set(body) != {'issue_key'}:
                    raise ValueError('invalid issue release')
                result = issue_groups.release(self.database, user, body['issue_key'])
            elif route == '/api/session' and method == 'GET':
                result = profile
            elif route == '/api/session' and method == 'DELETE':
                sessions.revoke(self.database, self._secret(handler))
                handler.send(200, {'signed_out':True}, headers={'Set-Cookie':sessions.cookie('',clear=True)})
                return True
            elif route == '/api/retest/clients' and method == 'GET':
                result = retests.clients(self.database,user)
            elif route == '/api/retest/batches' and method == 'GET':
                result = retests.batches(self.database,user,int(params.get('before',[str(2**63-1)])[0]))
            elif route == '/api/retest/preview' and method == 'POST':
                result = retests.create(self.database,user,body,preview=True)
            elif route == '/api/retest/batches' and method == 'POST':
                result = retests.create(self.database,user,body)
            elif batch_detail and method == 'GET':
                result = retests.detail(self.database,user,parts[3],int(params.get('after',['0'])[0]))
            elif batch_cancel and method == 'POST':
                if body != {}:
                    raise ValueError('invalid cancel')
                result = retests.cancel(self.database,user,parts[3])
            elif route in agent_routes:
                result = {'/api/retest/register':retests.register,
                          '/api/retest/claim':retests.claim,
                          '/api/retest/renew':retests.renew,
                          '/api/retest/complete':retests.complete}[route](self.database,user,body)
            else:
                handler.send(404, {'error':'not found'})
                return True
            if route == '/api/retest/complete' and result.get('accepted') and self.on_publish:
                self.on_publish()
            handler.send(200,result)
        except MaintenanceBlocked:
            reject(handler)
        except PermissionError:
            handler.send(401, {'error':'鉴权失败或无操作权限'})
        except LookupError:
            handler.send(404, {'error':'record not found'})
        except (ValueError, TypeError, KeyError, OverflowError):
            handler.send(400, {'error':'invalid control request'})
        except (sqlite3.Error, OSError):
            handler.send(503, {'error':'control database unavailable'})
        return True

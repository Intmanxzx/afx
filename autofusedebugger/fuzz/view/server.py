"""Loopback-only read-only viewer. URLs are local capabilities, not public links."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
import json
from pathlib import Path
import secrets
import sqlite3
from urllib.parse import parse_qs, urlsplit
import webbrowser

from ..storage.queries import ResultQueries

ASSETS = {'favicon.svg': ('favicon.svg', 'image/svg+xml'), 'runtime.js': ('runtime.js', 'text/javascript; charset=utf-8'), 'report': ('index.html', 'text/html; charset=utf-8'), 'charts.js': ('charts.js', 'text/javascript; charset=utf-8'), '': ('index.html', 'text/html; charset=utf-8'),
          'app.js': ('app.js', 'text/javascript; charset=utf-8'),
          'style.css': ('style.css', 'text/css; charset=utf-8'), 'issues.js': ('issues.js', 'text/javascript; charset=utf-8'), 'issues.css': ('issues.css', 'text/css; charset=utf-8')}


class ViewerServer(ThreadingHTTPServer):
    daemon_threads = True
    def get_request(self):
        connection, address = super().get_request()
        connection.settimeout(5)
        return connection, address


def create_server(database, *, port=0):
    if type(port) is not int or not 0 <= port <= 65535:
        raise ValueError('port must be between 0 and 65535')
    queries = ResultQueries(Path(database).expanduser().resolve())  # Validate before listening.
    token = secrets.token_urlsafe(24)
    prefix = '/' + token + '/'

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass  # Do not log the local capability or database contents.

        def respond(self, status, data, content_type='application/json; charset=utf-8', policy=None):
            if not isinstance(data, bytes):
                data = json.dumps(data, ensure_ascii=True).encode('utf-8')
            if len(data) > (16 if content_type.startswith('text/html') else 2) * 1024 * 1024:
                status, data = 413, b'{"error":"response too large"}'
            self.send_response(status)
            for key, value in {
                'Content-Type': content_type, 'Content-Length': str(len(data)),
                'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff',
                'Referrer-Policy': 'no-referrer',
                'Content-Security-Policy': policy or "default-src 'none'; script-src 'self'; style-src 'self'; style-src-attr 'unsafe-inline'; connect-src 'self'; img-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'",
            }.items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            authority = f'127.0.0.1:{self.server.server_port}'
            if self.headers.get('Host') != authority or self.headers.get('Origin', 'http://' + authority) != 'http://' + authority:
                self.respond(403, {'error': 'local origin required'}); return
            if len(self.path) > 2048:
                self.respond(414, {'error': 'URL too long'}); return
            url = urlsplit(self.path)
            if not url.path.startswith(prefix):
                self.respond(404, {'error': 'not found'}); return
            route = url.path[len(prefix):]
            try:
                if route in ASSETS:
                    name, mime = ASSETS[route]
                    data = resources.files('autofusedebugger.fuzz.view').joinpath('static').joinpath(name).read_bytes()
                    self.respond(200, data, mime); return
                raw = parse_qs(url.query, keep_blank_values=True, max_num_fields=8)
                if any(len(v) != 1 for v in raw.values()):
                    raise ValueError('duplicate query field')
                params = {k: v[0] for k,v in raw.items()}
                if route == 'api/issues' and not set(params)-{'dimension'}:
                    from ..central.issue_groups import list_groups
                    self.respond(200, list_groups(database, params.get('dimension')))
                    return
                if route == 'api/statistics' and not set(params)-{'seed','status','user_id','q'}:
                    from ..reporting.statistics import statistics
                    self.respond(200, statistics(database, **params))
                    return
                if route == 'api/case-history':
                    if set(params)-{'seed','shape','limit','after','snapshot'} or not {'seed','shape'} <= set(params):
                        raise ValueError('invalid history query')
                    for key in ('shape','limit','after','snapshot'):
                        if key in params: params[key]=int(params[key])
                    self.respond(200, queries.case_page(_history=True, **params))
                    return
                if route == 'api/cases':
                    if set(params)-{'seed','status','user_id','q','limit','after'}:
                        raise ValueError('unknown field')
                    for key in ('limit','after'):
                        if key in params: params[key]=int(params[key])
                    self.respond(200, queries.case_page(**params))
                    return
                if route in {'api/results','api/summary'}:
                    allowed = {'seed','status','user_id','q'} | ({'limit','after'} if route == 'api/results' else set())
                    if set(params) - allowed:
                        raise ValueError('unknown query field')
                    for key in ('limit','after'):
                        if key in params:
                            params[key] = int(params[key])
                    value = queries.page(**params) if route == 'api/results' else queries.case_summary(**params)
                elif route == 'api/compare' and set(params) == {'left', 'right'}:
                    from ..storage.comparison import compare
                    value = compare(database, params['left'], params['right'])
                elif route.startswith('api/attempts/') and not set(params)-{'shape'}:
                    value = queries.detail(route[len('api/attempts/'):], shape_index=int(params['shape']) if 'shape' in params else None)
                else:
                    self.respond(404, {'error': 'not found'}); return
                self.respond(200, value)
            except LookupError:
                self.respond(404, {'error': 'attempt not found'})
            except (ValueError, TypeError):
                self.respond(400, {'error': 'invalid query or result data'})
            except (sqlite3.Error, OSError):
                self.respond(503, {'error': 'database unavailable or query limit exceeded'})

        def do_POST(self):
            self.respond(405, {'error': 'read-only viewer'})
        do_PUT = do_DELETE = do_PATCH = do_POST

    server = ViewerServer(('127.0.0.1', port), Handler)
    server.viewer_url = f'http://127.0.0.1:{server.server_port}{prefix}'
    return server


def serve(database, *, port=0, open_browser=True):
    with create_server(database, port=port) as server:
        print(f'Offline read-only viewer: {server.viewer_url}', flush=True)
        print('Press Ctrl+C to stop. No background service is installed.', flush=True)
        if open_browser:
            try:
                webbrowser.open(server.viewer_url)
            except webbrowser.Error:
                print('Browser unavailable; open the URL above on this computer.', flush=True)
        try:
            server.serve_forever(poll_interval=0.2)
        except KeyboardInterrupt:
            pass
    return 0

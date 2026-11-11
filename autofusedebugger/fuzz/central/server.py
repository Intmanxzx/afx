"""Loopback upstream for a TLS reverse proxy, bounded workers and request bodies."""
from concurrent.futures import ThreadPoolExecutor
from http.server import HTTPServer, BaseHTTPRequestHandler
from importlib import resources
import json
from pathlib import Path
import sqlite3
import threading
import time
from urllib.parse import urlsplit, parse_qs

from .repository import initialize, ingest
from ..storage.queries import ResultQueries


class Server(HTTPServer):
    def __init__(self,*args,**kwargs):
        self.pool=ThreadPoolExecutor(max_workers=4)
        self.slots=threading.BoundedSemaphore(12)
        super().__init__(*args,**kwargs)
    def get_request(self):
        conn,addr=super().get_request();conn.settimeout(15);return conn,addr
    def process_request(self,request,address):
        if not self.slots.acquire(blocking=False):
            self.shutdown_request(request);return
        self.pool.submit(self.work,request,address)
    def work(self,request,address):
        try:
            self.finish_request(request,address)
        except Exception:
            pass
        finally:
            self.shutdown_request(request);self.slots.release()
    def server_close(self):
        super().server_close();self.pool.shutdown(wait=True)
        if hasattr(self, 'public_cache'):self.public_cache.close()


def create_server(database,port=8090, *, cache_interval=15):
    database=Path(database).resolve();initialize(database);queries=ResultQueries(database)
    from contextlib import closing
    from ..storage.repository import _connect
    from .runtime import initialize as initialize_runtime, connect as runtime_connect
    with closing(runtime_connect(database)) as conn,conn:initialize_runtime(conn)
    from .cache import PublicCache, PAGE_SIZES
    public_cache = PublicCache(database, interval=cache_interval)
    statistics_lock = threading.Lock()
    statistics_cache = {}

    def cached_statistics(params):
        from ..reporting.statistics import statistics
        key=tuple(sorted(params.items()))
        # One chart calculation per key/15s instead of four simultaneous full
        # rollups competing with uploads on a small server. Bounded cache.
        with statistics_lock:
            now=time.monotonic()
            entry=statistics_cache.get(key)
            if entry and now-entry[0]<15:return entry[1]
            value=statistics(database, **params)
            if len(statistics_cache)>=16:statistics_cache.pop(next(iter(statistics_cache)))
            statistics_cache[key]=(time.monotonic(),value)
            return value

    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):
            pass
        def send(self,status,value,mime='application/json; charset=utf-8',policy=None,headers=None):
            body=value if isinstance(value,bytes) else json.dumps(value,ensure_ascii=True,allow_nan=False).encode()
            if len(body)>(16 if mime.startswith('text/html') else 2)*1024*1024:
                status,body=413,b'{"error":"response too large"}'
            self.send_response(status)
            for key,value in {'Content-Type':mime,'Content-Length':str(len(body)),'Cache-Control':'no-store',
                'X-Content-Type-Options':'nosniff','Referrer-Policy':'no-referrer',
                'Content-Security-Policy':policy or "default-src 'none'; script-src 'self'; style-src 'self'; style-src-attr 'unsafe-inline'; connect-src 'self'; img-src 'self'; frame-ancestors 'none'; base-uri 'none'"}.items():
                if status != 304 or key != 'Content-Length':self.send_header(key,value)
            for key,value in (headers or {}).items():self.send_header(key,value)
            self.end_headers()
            if status != 304:self.wfile.write(body)
        def cached(self, key):
            entry, failed = public_cache.get(key)
            if entry is None:
                return self.send(503, {'error':'公共数据准备中，请稍后重试'},
                                 headers={'Retry-After':'15', 'X-Cache-State':'warming'})
            headers = {'ETag':entry.etag,
                       'X-Data-Generated-At':str(entry.generated_at),
                       'X-Cache-State':'stale' if failed else 'ready'}
            tags = [tag.strip().removeprefix('W/') for tag in self.headers.get('If-None-Match','').split(',')]
            if entry.status == 200 and (entry.etag in tags or '*' in tags):
                return self.send(304, b'', headers=headers)
            return self.send(entry.status, entry.body, headers=headers)

        def do_GET(self):
            try:
                if len(self.path)>2048:
                    return self.send(414,{'error':'URL too long'})
                url=urlsplit(self.path);route=url.path
                if route=='/api/identity':
                    from contextlib import closing
                    from ..storage.repository import _connect
                    from .repository import authenticate
                    auth = self.headers.get('Authorization', '')
                    if not auth.startswith('Bearer ') or len(auth) > 256:
                        return self.send(401, {'error': 'credential required'})
                    with closing(_connect(database)) as conn:
                        user = authenticate(conn, auth[7:])
                        name = conn.execute('SELECT display_name FROM users WHERE id=?', (user,)).fetchone()[0]
                    return self.send(200, {'user_id': user, 'display_name': name})
                if route=='/healthz':
                    return self.send(200,{'status':'ok','protocol':1})
                if route in {'/','/report','/app.js','/charts.js','/style.css','/runtime.js'}:
                    name={'/':'index.html','/report':'index.html','/app.js':'app.js','/charts.js':'charts.js','/style.css':'style.css','/runtime.js':'runtime.js'}[route]
                    data=resources.files('autofusedebugger.fuzz.view').joinpath('static').joinpath(name).read_bytes()
                    if route=='/':
                        data=data.replace('离线结果'.encode(),'团队结果'.encode()).replace('只读 · 本机查看'.encode(),'公开结果 · 自动同步'.encode()).replace('查看不会运行模型或上传数据'.encode(),'公共缓存每15秒检查更新；页面每15秒刷新；上传周期由测试端配置'.encode())
                    return self.send(200,data,{'/':'text/html; charset=utf-8','/report':'text/html; charset=utf-8','/app.js':'text/javascript; charset=utf-8','/charts.js':'text/javascript; charset=utf-8','/style.css':'text/css; charset=utf-8','/runtime.js':'text/javascript; charset=utf-8'}[route])
                raw=parse_qs(url.query,keep_blank_values=True,max_num_fields=8)
                if any(len(v)!=1 for v in raw.values()):
                    raise ValueError('duplicate field')
                params={k:v[0] for k,v in raw.items()}
                if route == '/api/runtime':
                    from .runtime import snapshot
                    if set(params)-{'user_id','machine_id','model','backend','version','minutes'}:raise ValueError('unknown field')
                    return self.send(200,snapshot(database,**params))
                if route == '/api/statistics' and not set(params)-{'seed','status','user_id','q'}:
                    from ..reporting.statistics import statistics
                    if set(params) <= {'q'} and not params.get('q'):
                        return self.cached('statistics')
                    self.send(200, cached_statistics(params))
                    return
                if route == '/api/cases':
                    if set(params)-{'seed','status','user_id','q','limit','after'}:
                        raise ValueError('unknown field')
                    for key in ('limit','after'):
                        if key in params: params[key]=int(params[key])
                    if (set(params) <= {'limit','after','q'} and not params.get('q')
                            and params.get('after',0) == 0 and params.get('limit',50) in PAGE_SIZES):
                        return self.cached('cases:'+str(params.get('limit',50)))
                    self.send(200, queries.case_page(**params))
                    return
                if route=='/api/compare' and set(params)=={'left','right'}:
                    from ..storage.comparison import compare
                    return self.send(200,compare(database,params['left'],params['right']))
                if route in {'/api/results','/api/summary'}:
                    allowed={'seed','status','user_id','q'}|({'limit','after'} if route=='/api/results' else set())
                    if set(params)-allowed:
                        raise ValueError('unknown field')
                    for key in ('limit','after'):
                        if key in params:params[key]=int(params[key])
                    if route == '/api/summary' and set(params) <= {'q'} and not params.get('q'):
                        return self.cached('summary')
                    result=queries.page(**params,newest=True) if route=='/api/results' else queries.case_summary(**params)
                    if route=='/api/results':
                        from contextlib import closing
                        from ..storage.repository import _connect
                        with closing(_connect(database)) as conn:
                            for item in result['items']:
                                owner=conn.execute('SELECT u.display_name FROM owners o JOIN users u ON u.id=o.user_id WHERE o.attempt_id=?',(item['id'],)).fetchone()
                                item['display_name']=owner[0] if owner else 'unattributed'
                    return self.send(200,result)
                if route.startswith('/api/attempts/') and not set(params)-{'shape'}:
                    return self.send(200,queries.detail(route[len('/api/attempts/'):], shape_index=int(params['shape']) if 'shape' in params else None))
                return self.send(404,{'error':'not found'})
            except PermissionError:
                self.send(401, {'error': 'invalid credential'})
            except LookupError:
                self.send(404,{'error':'not found'})
            except (ValueError,TypeError):
                self.send(400,{'error':'invalid query'})
            except (OSError,sqlite3.Error):
                self.send(503,{'error':'database unavailable'})
        def do_POST(self):
            if self.path not in {'/api/upload','/api/facts','/api/heartbeat'}:return self.send(404,{'error':'not found'})
            try:
                if self.headers.get('Transfer-Encoding') or self.headers.get('Content-Type','').split(';')[0]!='application/json':
                    return self.send(400,{'error':'JSON Content-Length required'})
                length=int(self.headers.get('Content-Length','0'))
                if not 1<=length<=2*1024*1024:return self.send(413,{'error':'bounded body required'})
                auth=self.headers.get('Authorization','')
                if not auth.startswith('Bearer ') or len(auth)>256:
                    return self.send(401,{'error':'credential required'})
                # Authenticate before reading/validating any submitted records.
                from contextlib import closing
                from ..storage.repository import _connect
                from .repository import authenticate
                with closing(_connect(database)) as conn:authenticate(conn,auth[7:])
                body=self.rfile.read(length)
                if len(body)!=length:raise ValueError('incomplete body')
                if self.path=='/api/heartbeat':
                    from .runtime import ingest as heartbeat
                    result=heartbeat(database,auth[7:],json.loads(body))
                elif self.path=='/api/facts':
                    from ..sync.facts import ingest as ingest_facts
                    result=ingest_facts(database,auth[7:],json.loads(body))
                else:
                    result=ingest(database,auth[7:],json.loads(body))
                self.send(200,result)
            except PermissionError:
                self.send(401,{'error':'invalid credential or ownership'})
            except (ValueError,KeyError,TypeError,OverflowError):
                self.send(400,{'error':'invalid batch or revision conflict'})
            except (OSError,sqlite3.Error) as exc:
                import logging
                logging.getLogger(__name__).warning('upload unavailable: %s %s',type(exc).__name__,getattr(exc,'sqlite_errorname',''))
                self.send(503,{'error':'upload temporarily unavailable'})
        def do_DELETE(self):self.send(405,{'error':'method not allowed'})
        do_PUT=do_PATCH=do_DELETE
    server = Server(('127.0.0.1',port),Handler)
    server.public_cache = public_cache
    public_cache.start()
    return server


def serve(database,port=8090):
    with create_server(database,port) as server:
        server.serve_forever()

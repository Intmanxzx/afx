import http.client
import json
import sqlite3
import threading
import time

from autofusex.fuzz.central.cache import PublicCache, PAGE_SIZES
from autofusex.fuzz.central.repository import initialize
from autofusex.fuzz.central.server import create_server
from autofusex.fuzz.storage import ResultRepository
from autofusex.fuzz.storage.queries import ResultQueries
from test_fuzz_case_summary import save


def wait_until(predicate, timeout=5):
    deadline = time.monotonic()+timeout
    while time.monotonic()<deadline:
        if predicate():return
        time.sleep(.02)
    assert predicate()


def test_cached_page_sizes_preserve_cursors(tmp_path):
    db=tmp_path/'db';initialize(db);repo=ResultRepository(db)
    for i in range(205):save(repo,str(i),'af-mlir','ok')
    entries=PublicCache(db).build();queries=ResultQueries(db)
    for size in PAGE_SIZES:
        assert json.loads(entries['cases:'+str(size)].body)==queries.case_page(limit=size)
    assert json.loads(entries['summary'].body)==queries.case_summary()


def test_producer_detects_external_writes_and_keeps_stale_snapshot(tmp_path, monkeypatch):
    db=tmp_path/'db';initialize(db)
    cache=PublicCache(db,interval=.1)
    original=cache.build;calls=[];fail=[False]
    def build():
        calls.append(threading.current_thread().name)
        if fail[0]:raise RuntimeError('test failure')
        return original()
    monkeypatch.setattr(cache,'build',build)
    cache.start()
    try:
        assert cache.ready.wait(3)
        initial=cache.get('summary')[0]
        time.sleep(1.2)
        assert len(calls)==1 # no writes: no statistics SQL
        fail[0]=True
        with sqlite3.connect(db) as conn:conn.execute('CREATE TABLE external_fact(value)')
        wait_until(lambda:cache.get('summary')[1])
        assert cache.get('summary')[0] is initial
        fail[0]=False
        wait_until(lambda:cache.get('summary')[0] is not initial)
        assert not cache.get('summary')[1]
        assert set(calls)=={'public-cache'}
    finally:cache.close()
    assert not cache._thread.is_alive()


def test_http_warming_etag_and_no_request_builders(tmp_path, monkeypatch):
    gate=threading.Event();original=PublicCache.build;calls=[]
    def build(self):
        calls.append(threading.current_thread().name)
        gate.wait(3)
        return original(self)
    monkeypatch.setattr(PublicCache,'build',build)
    server=create_server(tmp_path/'db',0)
    thread=threading.Thread(target=server.serve_forever);thread.start()
    def request(path,headers=None):
        conn=http.client.HTTPConnection('127.0.0.1',server.server_port)
        try:
            conn.request('GET',path,headers=headers or {})
            r=conn.getresponse();return r.status,dict(r.getheaders()),r.read()
        finally:conn.close()
    try:
        status,headers,_=request('/api/summary')
        assert status==503 and headers['X-Cache-State']=='warming'
        gate.set();assert server.public_cache.ready.wait(3)
        for path in ['/api/summary','/api/statistics','/api/cases?limit=10&after=0','/api/cases?q=&limit=25']:
            status,headers,body=request(path)
            assert status==200 and json.loads(body) is not None
            for _ in range(5):
                code,h,b=request(path,{'If-None-Match':headers['ETag']})
                assert code==304 and b==b'' and h['ETag']==headers['ETag']
        assert calls==['public-cache']
        assert request('/api/cases?limit=0')[0]==400
        assert request('/api/summary?limit=1')[0]==400
        assert request('/api/cases?after=1&limit=10')[0]==200
        assert request('/api/summary?q=failed')[0]==200
    finally:
        gate.set();server.shutdown();server.server_close();thread.join()


def test_default_refresh_interval_and_retry_header(tmp_path, monkeypatch):
    monkeypatch.setattr(PublicCache, 'start', lambda self: None)
    monkeypatch.setattr(PublicCache, 'close', lambda self: None)
    server = create_server(tmp_path/'db', 0)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        assert server.public_cache.interval == 30
        conn = http.client.HTTPConnection('127.0.0.1', server.server_port)
        try:
            conn.request('GET', '/api/summary')
            reply = conn.getresponse()
            assert reply.status == 503
            assert reply.getheader('Retry-After') == '30'
            reply.read()
        finally:
            conn.close()
    finally:
        server.shutdown(); server.server_close(); thread.join()


def test_refresh_rest_backoff_cap_and_recovery(tmp_path, monkeypatch):
    db = tmp_path/'db'; initialize(db)
    cache = PublicCache(db)
    waits = []
    builds = []
    initial = object()
    cache._entries = {'summary': initial}

    class Stop:
        def is_set(self):
            return len(waits) >= 9

        def wait(self, delay):
            waits.append(delay)
            # Failed refreshes must keep the old snapshot available.
            if len(waits) <= 6:
                assert cache.get('summary') == (initial, True)
            if len(waits) == 7:
                # A write after successful recovery triggers another build.
                with sqlite3.connect(db) as conn:
                    conn.execute('CREATE TABLE changed(value)')

    class Wake:
        def wait(self, delay):cache._stop.wait(delay)
        def clear(self):pass

    monkeypatch.setattr(cache, '_wake', Wake())

    def build():
        builds.append(1)
        if len(builds) <= 6 or len(builds) == 8:
            raise RuntimeError('busy database')
        return {'summary': initial}

    monkeypatch.setattr(cache, '_stop', Stop())
    monkeypatch.setattr(cache, 'build', build)
    # Simulate an arbitrarily slow build: elapsed time must not shorten rest.
    from itertools import count
    clock = count(0, 100)
    monkeypatch.setattr('autofusex.fuzz.central.cache.time.monotonic', lambda: next(clock))
    cache._run()
    assert waits == [30, 60, 120, 240, 300, 300, 30, 30, 30]
    assert len(builds) == 9
    assert cache.ready.is_set()
    assert cache.get('summary') == (initial, False)


def test_close_interrupts_long_refresh_wait(tmp_path):
    db = tmp_path/'db'; initialize(db)
    cache = PublicCache(db, interval=300)
    cache.start()
    try:
        assert cache.ready.wait(3)
    finally:
        closer = threading.Thread(target=cache.close, daemon=True)
        closer.start()
        closer.join(2)
    assert not closer.is_alive()
    assert not cache._thread.is_alive()


def test_incremental_bootstrap_falls_back_to_exact_snapshot(tmp_path,monkeypatch):
    from autofusex.fuzz.reporting.incremental import IncrementalStatistics
    from autofusex.fuzz.reporting.statistics import statistics
    db=tmp_path/'db';initialize(db);save(ResultRepository(db),'one','af-mlir','ok')
    def backlog(self):raise TimeoutError('checkpoint saved')
    monkeypatch.setattr(IncrementalStatistics,'refresh',backlog)
    result=PublicCache(db).build()
    assert json.loads(result['summary'].body)==ResultQueries(db).case_summary()
    assert json.loads(result['statistics'].body)==statistics(db)


def test_busy_rollup_does_not_block_health_or_cached_reads(tmp_path):
    server = create_server(tmp_path/'db', 0)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        assert server.public_cache.ready.wait(5)
        with server.public_cache.query_lock:
            for path, expected in [('/api/cases?after=1',503),
                                   ('/api/summary?q=failed',503),
                                   ('/api/statistics?q=failed',503),
                                   ('/healthz',200),('/api/cases?limit=10',200)]:
                conn = http.client.HTTPConnection('127.0.0.1',server.server_port,timeout=2)
                try:
                    conn.request('GET',path)
                    reply=conn.getresponse()
                    assert reply.status==expected
                    if expected==503:assert reply.getheader('Retry-After')=='5'
                    reply.read()
                finally:conn.close()
    finally:
        server.shutdown();server.server_close();thread.join()

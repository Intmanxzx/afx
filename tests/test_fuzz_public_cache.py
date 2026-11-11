import http.client
import json
import sqlite3
import threading
import time

from autofusedebugger.fuzz.central.cache import PublicCache, PAGE_SIZES
from autofusedebugger.fuzz.central.repository import initialize
from autofusedebugger.fuzz.central.server import create_server
from autofusedebugger.fuzz.storage import ResultRepository
from autofusedebugger.fuzz.storage.queries import ResultQueries
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

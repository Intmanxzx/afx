import copy
import json
from pathlib import Path
import sqlite3
import threading
from urllib.request import build_opener, ProxyHandler

import pytest

from autofusedebugger.fuzz.storage import ResultRepository
from autofusedebugger.fuzz.sync.outbox import pending, acknowledge
from autofusedebugger.fuzz.central.repository import create_user, ingest
from autofusedebugger.fuzz.central.server import create_server
from autofusedebugger.fuzz.sync.client import once


def result(seed='42'):
    return {'seed':seed,'workload_id':'a'*64,'oracle_id':'ref/1','backend':'cpu-eager','status':'ok',
            'execution':{'backend':'cpu-eager','device':'cpu'},'shape_results':[{'shape_index':0,'correctness':'pass','performance':'not_requested'}]}


def test_outbox_atomic_with_save_and_ack(tmp_path):
    repo=ResultRepository(tmp_path/'local');identity=repo.save(result())
    batch=pending(repo.path)
    assert batch['items'][0]['attempt_id']==identity
    assert pending(repo.path)==batch
    acknowledge(repo.path,batch['source_id'],batch['items'][-1]['revision'])
    assert not pending(repo.path)['items']
    with pytest.raises(ValueError):acknowledge(repo.path,batch['source_id'],99999)
    snap=repo.snapshot(tmp_path/'snapshot')
    with sqlite3.connect(snap) as conn:
        assert conn.execute("SELECT name FROM sqlite_master WHERE name LIKE 'sync_%'").fetchall()==[]


def test_ingest_idempotence_conflict_identity_and_atomicity(tmp_path):
    server=tmp_path/'server';key=tmp_path/'key'
    create_user(server,'Tester',key)
    repo=ResultRepository(tmp_path/'local');repo.save(result());repo.save(result('43'))
    batch=pending(repo.path);secret=key.read_text().strip()
    answer=ingest(server,secret,batch)
    assert ingest(server,secret,batch)==answer
    assert len(ResultRepository(server).list()['items'])==2
    changed=copy.deepcopy(batch);changed['items'][0]['record']['attempt']['status']='fail'
    with pytest.raises(ValueError):ingest(server,secret,changed)
    other=tmp_path/'other-key';create_user(server,'Other',other)
    with pytest.raises(PermissionError):ingest(server,other.read_text().strip(),batch)
    with pytest.raises(PermissionError):ingest(server,'wrong',batch)
    repo.save(result('44'));repo.save(result('45')); acknowledge(repo.path,batch['source_id'],answer['ack'])
    later=pending(repo.path);later['items'][-1]['record']['attempt']['status']='not-valid'
    with pytest.raises(ValueError):ingest(server,secret,later)
    assert len(ResultRepository(server).list()['items'])==2


def test_http_sync_lost_ack_retry_and_public_query(tmp_path):
    serverdb=tmp_path/'server';key=tmp_path/'key';create_user(serverdb,'Tester',key)
    home=tmp_path/'home';home.mkdir();repo=ResultRepository(home/'results');repo.save(result())
    server=create_server(serverdb,0)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        endpoint=f'http://127.0.0.1:{server.server_port}'
        # HTTP allowed only in test injected config/opener; configure requires TLS.
        (home/'sync.json').write_text(json.dumps({'endpoint':endpoint,'key_file':str(key),'interval':30}))
        opener=build_opener(ProxyHandler({})).open
        assert once(home,repo.path,opener=opener)['uploaded']==1
        assert once(home,repo.path,opener=opener)['pending']==0
        with opener(endpoint+'/api/summary') as response:assert json.load(response)['total']==1
        with opener(endpoint+'/') as response:assert '团队结果' in response.read().decode()
        repo.save(result('43'))
        def lost(request,**kw):
            response=opener(request,**kw);response.read();response.close()
            raise OSError('ACK lost')
        with pytest.raises(OSError):once(home,repo.path,opener=lost)
        assert once(home,repo.path,opener=opener)['uploaded']==1
        assert len(ResultRepository(serverdb).list()['items'])==2
    finally:
        server.shutdown();server.server_close();thread.join()


def test_private_backup_preserves_receipts_auth_and_public_snapshot_strips_them(tmp_path):
    from autofusedebugger.fuzz.central.backup import backup
    server=tmp_path/'server';key=tmp_path/'key';create_user(server,'Tester',key)
    repo=ResultRepository(tmp_path/'local');repo.save(result())
    batch=pending(repo.path);secret=key.read_text().strip()
    ack=ingest(server,secret,batch)
    private=tmp_path/'private';backup(server,private)
    assert ingest(private,secret,batch)==ack
    assert len(ResultRepository(private).list()['items'])==1
    with pytest.raises(ValueError):backup(server,private)
    public=ResultRepository(server).snapshot(tmp_path/'public')
    with sqlite3.connect(public) as conn:
        assert not conn.execute("SELECT name FROM sqlite_master WHERE name IN ('users','receipts','sources','owners')").fetchall()


def test_sync_failure_does_not_ack_or_block_new_results(tmp_path):
    repo=ResultRepository(tmp_path/'local');repo.save(result())
    key=tmp_path/'key';key.write_text('dummy-test-key-1234567890');key.chmod(0o600)
    from autofusedebugger.fuzz.sync.client import configure
    configure(tmp_path,'https://example.invalid',key,30)
    before=pending(repo.path)
    def offline(*args,**kwargs):raise OSError('offline')
    with pytest.raises(OSError):once(tmp_path,repo.path,opener=offline)
    assert pending(repo.path)==before
    repo.save(result('43'))
    assert len(pending(repo.path)['items'])==2


def test_sync_configuration_requires_https_private_key_and_valid_interval(tmp_path):
    from autofusedebugger.fuzz.sync.client import configure
    key=tmp_path/'key';key.write_text('dummy-test-key-1234567890');key.chmod(0o600)
    for endpoint,interval in [('http://example.com',30),('https://user:secret@example.com',30),('https://example.com',0)]:
        with pytest.raises(ValueError):configure(tmp_path,endpoint,key,interval)
    assert configure(tmp_path,'https://example.com',key,5)['interval']==5

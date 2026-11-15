import copy
import json
from pathlib import Path
import sqlite3
import threading
from urllib.request import build_opener, ProxyHandler

import pytest

from autofusex.fuzz.storage import ResultRepository
from autofusex.fuzz.sync.outbox import pending, acknowledge
from autofusex.fuzz.central.repository import create_user, ingest
from autofusex.fuzz.central.server import create_server
from autofusex.fuzz.sync.client import once


def result(seed='680564733841876926926749214863536422954'):
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
    repo=ResultRepository(tmp_path/'local');repo.save(result());repo.save(result('680564733841876926926749214863536422955'))
    batch=pending(repo.path);secret=key.read_text().strip()
    answer=ingest(server,secret,batch)
    assert ingest(server,secret,batch)==answer
    assert len(ResultRepository(server).list()['items'])==2
    changed=copy.deepcopy(batch);changed['items'][0]['record']['attempt']['status']='fail'
    with pytest.raises(ValueError):ingest(server,secret,changed)
    other=tmp_path/'other-key';create_user(server,'Other',other)
    with pytest.raises(PermissionError):ingest(server,other.read_text().strip(),batch)
    with pytest.raises(PermissionError):ingest(server,'wrong',batch)
    repo.save(result('680564733841876926926749214863536422956'));repo.save(result('680564733841876926926749214863536422957')); acknowledge(repo.path,batch['source_id'],answer['ack'])
    later=pending(repo.path);later['items'][-1]['record']['attempt']['status']='not-valid'
    with pytest.raises(ValueError):ingest(server,secret,later)
    assert len(ResultRepository(server).list()['items'])==2


def test_http_sync_lost_ack_retry_and_public_query(tmp_path):
    serverdb=tmp_path/'server';key=tmp_path/'key';create_user(serverdb,'Tester',key)
    home=tmp_path/'home';home.mkdir();repo=ResultRepository(home/'results')
    # Public cases exclude CPU-only references. Exercise lost ACK recovery with
    # an actual tested backend rather than expecting a hidden reference row.
    tested=result();tested['backend']='af-mlir';tested['execution']={'backend':'af-mlir','device':'npu:0'}
    repo.save(tested)
    server=create_server(serverdb,0,cache_interval=.1)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        endpoint=f'http://127.0.0.1:{server.server_port}'
        # HTTP allowed only in test injected config/opener; configure requires TLS.
        (home/'sync.json').write_text(json.dumps({'endpoint':endpoint,'key_file':str(key),'interval':30}))
        opener=build_opener(ProxyHandler({})).open
        assert once(home,repo.path,opener=opener)['uploaded']==1
        assert once(home,repo.path,opener=opener)['pending']==0
        from test_fuzz_public_cache import wait_until
        wait_until(lambda: server.public_cache.get('summary')[0] is not None and
                   json.loads(server.public_cache.get('summary')[0].body)['total']==1)
        with opener(endpoint+'/api/summary') as response:assert json.load(response)['total']==1
        with opener(endpoint+'/') as response:assert '主导航' in response.read().decode()
        repo.save(result('680564733841876926926749214863536422955'))
        def lost(request,**kw):
            response=opener(request,**kw);response.read();response.close()
            raise OSError('ACK lost')
        with pytest.raises(OSError):once(home,repo.path,opener=lost)
        assert once(home,repo.path,opener=opener)['uploaded']==1
        assert len(ResultRepository(serverdb).list()['items'])==2
    finally:
        server.shutdown();server.server_close();thread.join()


def test_private_backup_preserves_receipts_auth_and_public_snapshot_strips_them(tmp_path):
    from autofusex.fuzz.central.backup import backup
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
    from autofusex.fuzz.sync.client import configure
    configure(tmp_path,'https://example.invalid',key,30)
    before=pending(repo.path)
    def offline(*args,**kwargs):raise OSError('offline')
    with pytest.raises(OSError):once(tmp_path,repo.path,opener=offline)
    assert pending(repo.path)==before
    repo.save(result('680564733841876926926749214863536422955'))
    assert len(pending(repo.path)['items'])==2


def test_sync_configuration_requires_https_private_key_and_valid_interval(tmp_path):
    from autofusex.fuzz.sync.client import configure
    key=tmp_path/'key';key.write_text('dummy-test-key-1234567890');key.chmod(0o600)
    for endpoint,interval in [('http://example.com',30),('https://user:secret@example.com',30),('https://example.com',0)]:
        with pytest.raises(ValueError):configure(tmp_path,endpoint,key,interval)
    assert configure(tmp_path,'https://example.com',key,5)['interval']==5


def test_central_newest_pagination_and_stable_cursor(tmp_path):
    from autofusex.fuzz.storage.queries import ResultQueries
    repo=ResultRepository(tmp_path/'local')
    for seed in ('680564733841876926926749214863536422913','680564733841876926926749214863536422914','680564733841876926926749214863536422915'):repo.save(result(seed))
    query=ResultQueries(repo.path)
    first=query.page(limit=2,newest=True)
    assert [r['seed'] for r in first['items']]==['680564733841876926926749214863536422915','680564733841876926926749214863536422914']
    repo.save(result('680564733841876926926749214863536422916'))
    assert [r['seed'] for r in query.page(limit=2,after=first['next_cursor'],newest=True)['items']]==['680564733841876926926749214863536422913']
    assert query.page(limit=1,newest=True)['items'][0]['seed']=='680564733841876926926749214863536422916'


def test_configuration_does_not_echo_admin_key(tmp_path):
    from autofusex.fuzz.sync.client import configure
    key=tmp_path/'key';secret='test-secret-long-value-0123456789'
    key.write_text(secret);key.chmod(0o600)
    answer=configure(tmp_path,'https://example.com',key,30)
    assert secret not in json.dumps(answer)
    assert secret not in (tmp_path/'sync.json').read_text()


def test_setup_configures_and_starts_sync_once_with_admin_key(tmp_path,monkeypatch):
    from autofusex.fuzz.cli import main
    from autofusex.fuzz.sync import client
    key=tmp_path/'key';key.write_text('admin-private-key-0123456789');key.chmod(0o600)
    calls=[]
    monkeypatch.setattr(client,'start',lambda *args:calls.append(args) or {'state':'started'})
    import io,uuid
    from autofusex.fuzz import onboarding
    actual=onboarding.initialize_user
    monkeypatch.setattr(onboarding,'initialize_user',lambda home,token,interval: actual(home,token,interval,
        opener=lambda *a,**k:io.BytesIO(json.dumps({'user_id':str(uuid.uuid4()),'display_name':'Tester'}).encode())))
    assert main(['setup','--data-dir',str(tmp_path),'--token',key.read_text(),'--sync-interval','30'])==0
    assert len(calls)==1
    assert json.loads((tmp_path/'sync.json').read_text())['interval']==30


def test_old_server_backup_recovers_with_explicit_epoch_without_duplicates(tmp_path):
    from autofusex.fuzz.sync.outbox import reset_epoch
    from autofusex.fuzz.sync.client import status
    server = tmp_path/'server'; key = tmp_path/'key'
    create_user(server, 'Recovery', key)
    repo = ResultRepository(tmp_path/'local')
    repo.save(result())
    first = pending(repo.path)
    ingest(server, key.read_text().strip(), first)
    acknowledge(repo.path, first['source_id'], first['items'][-1]['revision'])
    repo.save(result('680564733841876926926749214863536422955'))
    second = pending(repo.path)
    # Client has ACK for a record lost by a stale server restore.
    acknowledge(repo.path, second['source_id'], second['items'][-1]['revision'])
    repo.save(result('680564733841876926926749214863536422956'))
    with pytest.raises(ValueError, match='ahead'):
        ingest(server, key.read_text().strip(), pending(repo.path))
    reset_epoch(repo.path)
    replay = pending(repo.path)
    ack = ingest(server, key.read_text().strip(), replay)
    acknowledge(repo.path, ack['source_id'], ack['ack'])
    with sqlite3.connect(server) as conn:
        assert conn.execute('SELECT count(*) FROM attempts').fetchone()[0] == 3
    assert status(tmp_path, repo.path)['pending'] == 0


def test_offline_rotation_preserves_owner_and_rejects_old_key(tmp_path):
    from autofusex.fuzz.central.repository import rotate_key
    server = tmp_path/'server'; key = tmp_path/'key'; new = tmp_path/'new-key'
    user = create_user(server, 'Rotation', key)
    repo = ResultRepository(tmp_path/'local'); repo.save(result())
    batch = pending(repo.path)
    ingest(server, key.read_text().strip(), batch)
    rotate_key(server, user['user_id'], new)
    with pytest.raises(PermissionError):
        ingest(server, key.read_text().strip(), batch)
    assert ingest(server, new.read_text().strip(), batch)['ack'] == batch['items'][-1]['revision']
    with sqlite3.connect(server) as conn:
        assert conn.execute('SELECT count(*) FROM attempts').fetchone()[0] == 1
    assert new.stat().st_mode & 0o077 == 0


def test_identity_is_authenticated_and_does_not_expose_key(tmp_path):
    from urllib.request import Request, urlopen
    from urllib.error import HTTPError
    server_db = tmp_path/'server'; key = tmp_path/'key'
    user = create_user(server_db, 'Identity', key)
    with create_server(server_db, port=0) as server:
        thread = threading.Thread(target=server.serve_forever); thread.start()
        try:
            url = f'http://127.0.0.1:{server.server_port}/api/identity'
            with pytest.raises(HTTPError) as failure:
                urlopen(url)
            assert failure.value.code == 401
            with urlopen(Request(url, headers={'Authorization': 'Bearer '+key.read_text().strip()})) as response:
                assert json.load(response) == {'user_id': user['user_id'], 'display_name': 'Identity'}
        finally:
            server.shutdown(); thread.join()


def test_token_only_setup_resolves_cloud_name(tmp_path, monkeypatch):
    from autofusex.fuzz import onboarding
    from autofusex.fuzz.central.server import create_server
    from autofusex.fuzz.central.repository import create_user
    import threading
    server_db=tmp_path/'central.sqlite'
    identity=create_user(server_db,'niuyuhu')
    with create_server(server_db,port=0) as server:
        thread=threading.Thread(target=server.serve_forever);thread.start()
        try:
            monkeypatch.setattr(onboarding,'ENDPOINT',f'http://127.0.0.1:{server.server_port}')
            # Production configure enforces HTTPS; this local fixture only tests
            # real identity authentication and config identity assignment.
            from autofusex.fuzz.sync import client
            monkeypatch.setattr(client,'configure',lambda *args:None)
            home=tmp_path/'client'
            result=onboarding.initialize_user(home,identity['token'],start_sync=False)
            assert result['user']=='niuyuhu'
            assert json.loads((home/'config.json').read_text())['identity']['user_id']==identity['user_id']
            assert identity['token'] not in json.dumps(result)
        finally:
            server.shutdown();thread.join()

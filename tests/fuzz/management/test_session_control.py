from types import SimpleNamespace
from autofusex.client.store import Store
from autofusex.client.artifacts import atomic
from autofusex.client.session_control import reconcile
from autofusex.client import local_session, tasks
from autofusex.client.paths import debug_directory, sync_directory


def test_debug_directory_migrates_legacy_dumps(tmp_path):
    legacy = tmp_path / 'autofuse_compile_debug'
    legacy.mkdir()
    (legacy / 'ascgen_dump_pid_1').mkdir()
    (legacy / 'ascgen_dump_pid_1' / 'graph.txt').write_text('dump')
    target = debug_directory(tmp_path)
    assert target == tmp_path / 'debug' / 'autofuse_compile'
    assert (target / 'ascgen_dump_pid_1' / 'graph.txt').read_text() == 'dump'
    assert not legacy.exists()


def test_sync_directory_migrates_legacy_state(tmp_path):
    legacy = tmp_path / 'platform-sync.sqlite'
    legacy.write_bytes(b'state')
    target = sync_directory(tmp_path)
    assert (target / 'platform-sync.sqlite').read_bytes() == b'state'
    assert not legacy.exists()


def test_local_manifest_preserves_execution_environment(monkeypatch):
    monkeypatch.delenv('ASCEND_TOOLKIT_HOME', raising=False)
    monkeypatch.delenv('ASCEND_OPP_PATH', raising=False)
    monkeypatch.setenv('ASCEND_HOME_PATH', '/opt/cann')
    monkeypatch.setenv('PYTHONPATH', '/opt/cann/python')
    monkeypatch.setenv('LD_LIBRARY_PATH', '/opt/cann/lib64')
    monkeypatch.setenv('PATH', '/opt/cann/bin')
    manifest = local_session._local_manifest({'env': {'CUSTOM': 'kept'}})
    assert manifest['env'] == {
        'CUSTOM': 'kept',
        'ASCEND_HOME_PATH': '/opt/cann',
        'PYTHONPATH': '/opt/cann/python',
        'LD_LIBRARY_PATH': '/opt/cann/lib64',
        'PATH': '/opt/cann/bin',
    }


def test_start_stop_start_and_duplicate_delivery(tmp_path, monkeypatch):
    manager = SimpleNamespace(store=Store(tmp_path), processes={}, active=None)
    starts, stops = [], []
    def start(m):
        starts.append(1)
        m.processes['run'] = object()
        with m.store.connect() as db:
            db.execute("INSERT OR REPLACE INTO runs VALUES ('run','{}','preparing',1,NULL)")
        return {'run_id': 'run'}
    monkeypatch.setattr(local_session, 'start', start)
    monkeypatch.setattr(tasks, 'action', lambda s, r, a, p: stops.append(a))
    def command(identity, state):
        atomic(tmp_path/'daemon-control.json', dict(request_id=identity, state=state))
        reconcile(manager)
    command('one', 'running')
    assert manager.store.get('session_control')['status'] == 'running'
    with manager.store.connect() as db:
        db.execute("UPDATE runs SET state='running' WHERE id='run'")
    command('one', 'running')
    assert manager.store.get('session_control')['status'] == 'succeeded'
    assert len(starts) == 1
    command('two', 'stopped')
    assert stops == ['stop']
    assert manager.store.get('session_control')['state'] == 'stopping'
    manager.processes.clear()
    reconcile(manager)
    assert manager.store.get('session_control')['state'] == 'stopped'
    command('three', 'running')
    assert len(starts) == 2


def test_failed_start_does_not_report_running(tmp_path, monkeypatch):
    manager = SimpleNamespace(store=Store(tmp_path), processes={}, active=None)
    def fail(m): raise PermissionError('private detail')
    monkeypatch.setattr(local_session, 'start', fail)
    atomic(tmp_path/'daemon-control.json', dict(request_id='one', state='running'))
    reconcile(manager)
    result = manager.store.get('session_control')
    assert result['status'] == 'failed' and result['error'] == 'PermissionError'
    assert 'private detail' not in str(result)


def test_worker_failure_before_readiness_is_not_acknowledged(tmp_path, monkeypatch):
    manager = SimpleNamespace(store=Store(tmp_path), processes={}, active=None)
    def start(m):
        with m.store.connect() as db:
            db.execute("INSERT INTO runs VALUES ('run','{}','preparing',1,NULL)")
        return {'run_id': 'run'}
    monkeypatch.setattr(local_session, 'start', start)
    atomic(tmp_path/'daemon-control.json', dict(request_id='one', state='running'))
    reconcile(manager)
    assert manager.store.get('session_control')['status'] == 'running'
    with manager.store.connect() as db:
        db.execute("UPDATE runs SET state='failed' WHERE id='run'")
    reconcile(manager)
    assert manager.store.get('session_control')['status'] == 'failed'


def test_restart_drains_then_starts_once(tmp_path, monkeypatch):
    manager=SimpleNamespace(store=Store(tmp_path),processes={'old':object()},active=None)
    stops=[];starts=[]
    monkeypatch.setattr(tasks,'action',lambda *args:stops.append(args[2]))
    def start(m):
        starts.append(1)
        m.processes['new']=object()
        with m.store.connect() as db:db.execute("INSERT INTO runs VALUES ('new','{}','running',1,NULL)")
        return {'run_id':'new'}
    monkeypatch.setattr(local_session,'start',start)
    atomic(tmp_path/'daemon-control.json',dict(request_id='restart',state='restart'))
    reconcile(manager)
    assert stops==['stop'] and not starts
    manager.processes.clear()
    reconcile(manager)
    reconcile(manager)
    reconcile(manager)
    reconcile(manager)
    assert starts==[1]
    assert manager.store.get('session_control')['status']=='succeeded'

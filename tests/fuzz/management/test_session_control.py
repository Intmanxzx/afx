from types import SimpleNamespace
from autofusex.fuzz.daemon.management.store import Store
from autofusex.fuzz.daemon.management.artifacts import atomic
from autofusex.fuzz.daemon.management.session_control import reconcile
from autofusex.fuzz.daemon.management import local_session, tasks


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

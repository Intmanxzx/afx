from autofusex.fuzz.daemon.management import upgrade_drain
from autofusex.fuzz.daemon.management.store import Store


def test_quiet_window_restarts_when_descendant_reappears(tmp_path, monkeypatch):
    store = Store(tmp_path)
    clock = [0]
    monkeypatch.setattr(upgrade_drain.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(upgrade_drain.time, 'sleep', lambda _: clock.__setitem__(0, clock[0]+1))
    monkeypatch.setattr(upgrade_drain, 'task_processes', lambda _: [123] if clock[0] in (0, 4) else [])
    upgrade_drain.wait(store, {})
    assert clock[0] == 10


def test_activation_restores_execution_environment_on_rollback(tmp_path):
    import json
    from autofusex.fuzz.daemon.management.activation import select, finish
    store = Store(tmp_path)
    old = dict(id='local', python='/old/python', env={})
    new = dict(old, python='/new/python')
    with store.connect() as db:
        db.execute('INSERT INTO environments VALUES (?,?,?,?)', ('local','local',json.dumps(old),'ready'))
    switch = dict(previous=dict(python='/old/control'), target=dict(python='/new/control', version='0.2.3',
        environments=[dict(id='local',previous=old,target=new)]))
    select(store, switch)
    with store.connect() as db:
        assert json.loads(db.execute('SELECT manifest FROM environments').fetchone()[0]) == new
    finish(store, switch, False)
    with store.connect() as db:
        assert json.loads(db.execute('SELECT manifest FROM environments').fetchone()[0]) == old

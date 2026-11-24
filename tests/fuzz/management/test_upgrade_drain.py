from autofusex.client import upgrade_drain
from autofusex.client.store import Store


def test_quiet_window_restarts_when_descendant_reappears(tmp_path, monkeypatch):
    store = Store(tmp_path)
    clock = [0]
    monkeypatch.setattr(upgrade_drain.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(upgrade_drain.time, 'sleep', lambda _: clock.__setitem__(0, clock[0]+1))
    monkeypatch.setattr(upgrade_drain, 'task_processes', lambda _: [123] if clock[0] in (0, 4) else [])
    upgrade_drain.wait(store, {})
    assert clock[0] == 10


def test_activation_preserves_execution_environment_on_rollback(tmp_path, monkeypatch):
    import json
    from autofusex.client.activation import select, finish
    store = Store(tmp_path)
    old = dict(id='local', python='/old/python', env={})
    installed=[]
    monkeypatch.setattr('autofusex.client.activation.install_release', installed.append)
    with store.connect() as db:
        db.execute('INSERT INTO environments VALUES (?,?,?,?)', ('local','local',json.dumps(old),'ready'))
    switch = dict(previous=dict(python='/control',version='old'), target=dict(python='/control', version='new'))
    select(store, switch)
    with store.connect() as db:
        assert json.loads(db.execute('SELECT manifest FROM environments').fetchone()[0]) == old
    finish(store, switch, False)
    assert installed == [switch['target'], switch['previous']]
    with store.connect() as db:
        assert json.loads(db.execute('SELECT manifest FROM environments').fetchone()[0]) == old

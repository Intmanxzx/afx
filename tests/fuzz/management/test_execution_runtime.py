import json
import sys
from types import SimpleNamespace

from autofusex.fuzz.daemon.management import execution_runtime
from autofusex.fuzz.daemon.management.store import Store


def test_facts_uses_registered_pythonpath_not_control_path(tmp_path, monkeypatch):
    store = Store(tmp_path)
    monkeypatch.setenv('PYTHONPATH', '/control-only')
    manifest = {'python': sys.executable, 'env': {'PYTHONPATH': '/selected-cann/python'}}
    with store.connect() as db:
        db.execute("INSERT INTO environments VALUES ('local','local',?,'ready')", (json.dumps(manifest),))
    seen = []
    def run(*args, **kwargs):
        seen.append(kwargs['env'].get('PYTHONPATH'))
        return SimpleNamespace(stdout='{}')
    monkeypatch.setattr(execution_runtime.subprocess, 'run', run)
    execution_runtime.facts(store)
    manifest['env'] = {}
    with store.connect() as db:
        db.execute("UPDATE environments SET manifest=?", (json.dumps(manifest),))
    execution_runtime.facts(store)
    assert seen == ['/selected-cann/python', None]

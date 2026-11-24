import json
import sys
from types import SimpleNamespace

from autofusex.client import execution_runtime
from autofusex.client.store import Store


def test_facts_uses_registered_pythonpath_not_control_path(tmp_path, monkeypatch):
    store = Store(tmp_path)
    monkeypatch.setenv('PYTHONPATH', '/control-only')
    manifest = {'python': sys.executable, 'env': {'PYTHONPATH': '/selected-cann/python'}}
    with store.connect() as db:
        db.execute("INSERT INTO environments VALUES ('selected','selected',?,'ready')", (json.dumps(manifest),))
    seen = []
    def run(*args, **kwargs):
        seen.append(kwargs['env'].get('PYTHONPATH'))
        return SimpleNamespace(stdout='{}')
    monkeypatch.setattr(execution_runtime.subprocess, 'run', run)
    assert execution_runtime.facts(store) is None
    execution_runtime.facts(store, 'selected')
    manifest['env'] = {}
    with store.connect() as db:
        db.execute("UPDATE environments SET manifest=?", (json.dumps(manifest),))
    execution_runtime.facts(store, 'selected')
    from autofusex.client.installation import execution_env
    root = execution_env({})['PYTHONPATH']
    assert seen == [root + ':/selected-cann/python', root]

"""Start a configured session using the current Python/CANN environment."""
import json
import sys
import uuid
import os
from autofusex.contracts.configuration import canonical
from autofusex.client.configuration import load


_EXECUTION_ENV_KEYS = (
    'ASCEND_HOME_PATH', 'ASCEND_OPP_PATH', 'ASCEND_TOOLKIT_HOME',
    'LD_LIBRARY_PATH', 'PYTHONPATH', 'PATH',
)


def _local_manifest(existing=None):
    """Register the environment needed by isolated execution runners."""
    manifest = dict(existing or {})
    # A local session runs inside the control interpreter, but an existing
    # registered environment may point at an isolated execution interpreter.
    # Preserve that selection: replacing it with the control venv prevents
    # NPU discovery and worker startup after a control-only upgrade.
    manifest.setdefault('python', sys.executable)
    manifest.update(id='local', name='local')
    env = dict(manifest.get('env') or {})
    env.update({key: os.environ[key] for key in _EXECUTION_ENV_KEYS if os.environ.get(key)})
    manifest['env'] = env
    return manifest


def start(manager, dump=None, fuzz_only=False):
    store=manager.store
    spec=load(store.home/'task.json')
    if dump is not None:
        if type(dump) is not bool: raise ValueError('dump must be boolean')
        spec['execution']['dump']=dump
    client=json.loads((store.home/'client.json').read_text())
    if client['mode']=='local' and spec['results']['upload']:
        raise ValueError('local mode cannot upload results')
    # The local user explicitly selected this configuration. Register only the
    # current interpreter, not a fabricated remote environment installation.
    if spec['environment']=='local':
        with store.connect() as db:
            existing = db.execute("SELECT manifest FROM environments WHERE id='local'").fetchone()
        if existing:
            manifest = _local_manifest(json.loads(existing['manifest']))
        else:
            manifest = _local_manifest()
        with store.connect() as db:
            db.execute("INSERT OR REPLACE INTO environments VALUES (?,?,?,'ready')",('local','local',canonical(manifest)))
    processes = manager.processes
    if fuzz_only:
        with store.connect() as db:
            processes = {run: process for run, process in manager.processes.items()
                         if (row := db.execute('SELECT spec FROM runs WHERE id=?', (run,)).fetchone())
                         and json.loads(row['spec']).get('spec', {}).get('kind') == 'fuzz_task'}
    if processes:
        if dump is not None:
            with store.connect() as db:
                current = [json.loads(row['spec'])['spec']['execution'].get('dump',False) for row in db.execute("SELECT spec FROM runs WHERE state IN ('running','preparing','paused','pausing')")]
            if any(value != dump for value in current):
                raise ValueError('dump mode differs from active task; use afx restart --dump')
        return dict(state='running',run_ids=list(processes))
    from autofusex.client.tasks import start as start_task
    return start_task(store,spec,'local-start:'+str(uuid.uuid4()),manager.processes)

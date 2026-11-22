"""Start a configured session using the current Python/CANN environment."""
import json
import sys
import uuid
from .config import load, canonical, validate
from ...client.resources import resource_inventory


def start(manager):
    store=manager.store
    spec=load(store.home/'task.json')
    client=json.loads((store.home/'client.json').read_text())
    if client['mode']=='local' and spec['results']['upload']:
        raise ValueError('local mode cannot upload results')
    if manager.processes:
        return dict(state='running',run_ids=list(manager.processes))
    # The local user explicitly selected this configuration. Register only the
    # current interpreter, not a fabricated remote environment installation.
    if spec['environment']=='local':
        with store.connect() as db:
            existing = db.execute("SELECT manifest FROM environments WHERE id='local'").fetchone()
        if existing:
            manifest = json.loads(existing['manifest'])
        else:
            manifest=dict(id='local',name='local',python=sys.executable,env={})
        with store.connect() as db:
            db.execute("INSERT OR REPLACE INTO environments VALUES (?,?,?,'ready')",('local','local',canonical(manifest)))
    from .artifacts import atomic
    inventory=resource_inventory(store.home)
    ceiling=inventory['cpu_worker_ceiling'] or 1
    resources=dict(allowed_devices=spec['execution']['devices'],max_workers=ceiling,max_compile_threads=ceiling)
    # Keep an existing trust policy intact. A new local installation derives
    # execution authority from the local user's editable task specification.
    if not (store.home/'trust.json').exists():
        atomic(store.home/'trust.json',dict(resource_limits=resources))
    store.put('effective_policy',validate(dict(schema_version=1,kind='client_policy',resources=resources)))
    if spec['execution']['devices'] == ['all']:
        from .execution_runtime import facts
        execution = facts(store)
        devices = execution['inventory']['devices'] if execution else None
        if not devices:
            raise ValueError('registered execution environment cannot discover devices')
        spec['execution']['devices'] = devices
        resources['allowed_devices'] = devices
        store.put('effective_policy',validate(dict(schema_version=1,kind='client_policy',resources=resources)))
    from .tasks import start as start_task
    return start_task(store,spec,'local-start:'+str(uuid.uuid4()),manager.processes)

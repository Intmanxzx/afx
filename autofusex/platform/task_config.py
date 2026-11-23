"""User task overrides, inheriting the administrator's default when absent."""
import json
from autofusex.contracts.configuration import fingerprint
from autofusex.workloads.fuzz.configuration import validate
from .store import now_ms


def default():
    return validate(dict(schema_version=1,kind='fuzz_task',environment='local',workload=dict(mode='generate')))


def get(store, principal, *, administrator=False):
    principal.require_write()
    if administrator: principal.require_admin()
    owner='*' if administrator else principal.user_id
    with store.write() as db:
        db.execute('CREATE TABLE IF NOT EXISTS task_profiles(owner TEXT PRIMARY KEY,spec TEXT NOT NULL,updated INTEGER NOT NULL)')
        row=db.execute('SELECT spec FROM task_profiles WHERE owner=?',(owner,)).fetchone()
        source='personal'
        if row is None:
            row=db.execute("SELECT spec FROM task_profiles WHERE owner='*'").fetchone()
            source='administrator_default'
        spec=json.loads(row[0]) if row else default()
    return dict(config=spec,revision=fingerprint(spec),source=source)


def put(store, principal, body, *, administrator=False):
    principal.require_write()
    if administrator: principal.require_admin()
    owner='*' if administrator else principal.user_id
    spec=validate(body['config']) if body.get('config') is not None else None
    if spec and (spec['kind']!='fuzz_task' or spec['workload']['mode']!='generate'):
        raise ValueError('task profile must describe generation')
    with store.write() as db:
        db.execute('CREATE TABLE IF NOT EXISTS task_profiles(owner TEXT PRIMARY KEY,spec TEXT NOT NULL,updated INTEGER NOT NULL)')
        if spec is None:
            db.execute('DELETE FROM task_profiles WHERE owner=?',(owner,))
        else:
            db.execute('INSERT OR REPLACE INTO task_profiles VALUES (?,?,?)',(owner,json.dumps(spec),now_ms()))
    return get(store,principal,administrator=administrator)

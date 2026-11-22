"""Supervisor-owned activation journal and bounded readiness verification."""
import time
from .artifacts import atomic


def select(store, switch):
    environments(store, switch, 'target')
    atomic(store.home / 'active-client.json', switch['target'])
    switch.update(state='activating', started=time.time())
    store.put('client_switch', switch)


def finish(store, switch, success):
    switch['state'] = 'succeeded' if success else 'rolled_back'
    if not success:
        environments(store, switch, 'previous')
        atomic(store.home / 'active-client.json', switch['previous'])
    store.put('client_switch', switch)
    if switch.get('operation_id'):
        store.state(switch['operation_id'], 'succeeded' if success else 'failed',
                    dict(state=switch['state'], version=switch['target']['version']))


def ready(store, switch, status):
    if status.get('agent_python') != switch['target']['python']:
        return False
    if status.get('agent_version') != switch['target']['version']:
        return False
    connection = store.get('connection', {})
    import json
    client_path = store.home / 'client.json'
    client = json.loads(client_path.read_text()) if client_path.exists() else {}
    if client.get('mode') == 'managed' and not (
            connection.get('state') == 'ok' and connection.get('time', 0) >= switch['started']):
        return False
    if switch.get('resume'):
        if switch.get('drained'):
            for identity in switch.get('old_runs', []):
                proof = store.home/'runs'/identity/'worker-version.json'
                if not proof.exists() or json.loads(proof.read_text()).get('version') != switch['target']['version']:
                    return False
        return any(run['state'] in {'running', 'completed'} and (switch.get('drained') or run['id'] not in switch.get('old_runs', []))
                   for run in status.get('runs', []))
    return True


def environments(store, switch, side):
    import json
    with store.connect() as db:
        for item in switch['target'].get('environments', []):
            db.execute('UPDATE environments SET manifest=? WHERE id=?', (json.dumps(item[side]), item['id']))
    store.put('execution_facts', {})

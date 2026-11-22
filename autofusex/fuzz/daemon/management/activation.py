"""Supervisor-owned activation journal and bounded readiness verification."""
import time
from .artifacts import atomic


def select(store, switch):
    atomic(store.home / 'active-client.json', switch['target'])
    switch.update(state='activating', started=time.time())
    store.put('client_switch', switch)


def finish(store, switch, success):
    switch['state'] = 'succeeded' if success else 'rolled_back'
    if not success:
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
        return any(run['state'] == 'running' and run['id'] not in switch.get('old_runs', [])
                   for run in status.get('runs', []))
    return True

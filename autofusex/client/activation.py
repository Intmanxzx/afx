"""Supervisor-owned activation journal and bounded readiness verification."""
import time
from autofusex.client.artifacts import atomic


def select(store, switch):
    store.event(switch.get('operation_id'), 'activation.select target=%s python=%s' % (switch['target'].get('version'), switch['target'].get('python')))
    install_release(switch['target'])
    switch.update(state='activating', started=time.time())
    store.put('client_switch', switch)


def finish(store, switch, success):
    switch['state'] = 'succeeded' if success else 'rolled_back'
    if not success:
        install_release(switch['previous'])
    store.put('client_switch', switch)
    store.event(switch.get('operation_id'), 'activation.finish state=%s version=%s' % (switch['state'], switch['target'].get('version')))
    if switch.get('operation_id'):
        store.state(switch['operation_id'], 'succeeded' if success else 'failed',
                    dict(state=switch['state'], version=switch['target']['version']))


def ready(store, switch, status):
    return readiness(store, switch, status)[0]


def readiness(store, switch, status):
    expected_python = switch['target']['python']
    expected_version = switch['target']['version']
    if status.get('agent_python') != expected_python:
        return False, 'agent_python_mismatch expected=%s actual=%s' % (expected_python, status.get('agent_python'))
    if status.get('agent_version') != expected_version:
        return False, 'agent_version_mismatch expected=%s actual=%s' % (expected_version, status.get('agent_version'))
    connection = store.get('connection', {})
    import json
    client_path = store.home / 'client.json'
    client = json.loads(client_path.read_text()) if client_path.exists() else {}
    if client.get('mode') == 'managed' and not (
            connection.get('state') == 'ok' and connection.get('time', 0) >= switch['started']):
        return False, 'platform_connection_not_ready'
    if switch.get('resume'):
        if switch.get('drained'):
            for identity in switch.get('old_runs', []):
                proof = store.home/'runs'/identity/'worker-version.json'
                if not proof.exists() or json.loads(proof.read_text()).get('version') != switch['target']['version']:
                    return False, 'worker_version_missing run=%s' % identity
        if not any(run['state'] in {'running', 'completed'} and (switch.get('drained') or run['id'] not in switch.get('old_runs', []))
                   for run in status.get('runs', [])):
            return False, 'resumed_run_not_ready'
    return True, 'ready'


def install_release(release):
    import subprocess
    subprocess.run([release['python'], '-m', 'pip', 'install', '--no-index',
        '--find-links', release['wheels'], '--force-reinstall',
        'autofusex==' + release['version']], check=True, timeout=1800)
    actual = subprocess.check_output([release['python'], '-I', '-c',
        'import autofusex.client.daemon, importlib.metadata; print(importlib.metadata.version("autofusex"))'],
        text=True, timeout=60).strip()
    if actual != release['version']: raise ValueError('installed version mismatch')

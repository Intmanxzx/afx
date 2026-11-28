"""Independent, durable platform controls for fuzz and retests."""
import json
from autofusex.client.artifacts import atomic


def retest_enabled(store):
    saved = store.get('retest_control', {})
    if saved:
        return saved['state'] == 'running'
    path = store.home / 'daemon-control.json'
    command = json.loads(path.read_text()) if path.exists() else {}
    if command.get('state') == 'stopped' and command.get('workload') != 'fuzz':
        return False
    config = store.home / 'client.json'
    return config.exists() and json.loads(config.read_text()).get('platform', {}).get('retest', False)


def receive(store, controls):
    for name, command in controls.items():
        if name not in {'fuzz', 'retest'} or command.get('state') not in {'running', 'stopped'}:
            continue
        if store.get(name + '_control', {}).get('request_id') == command.get('request_id'):
            continue
        store.put(name + '_control', command)
        if name == 'retest' and command['state'] == 'stopped':
            from autofusex.client.paths import retest_directory
            from autofusex.workloads.fuzz.scheduling.ledger import TaskLedger
            for path in retest_directory(store.home).glob('task-*.json'):
                saved = json.loads(path.read_text())
                if not saved.get('delivered'):
                    ledger = TaskLedger(saved['queue'])
                    for identity in saved['task_ids']:
                        ledger.stop(identity)
        if name == 'fuzz':
            atomic(store.home / 'daemon-control.json', dict(command, workload='fuzz'))
            store.put('shutdown_when_idle', False)


def states(store):
    fuzz = store.get('session_control', {})
    with store.connect() as db:
        active_fuzz = any(json.loads(row['spec']).get('spec', {}).get('kind') == 'fuzz_task'
                          for row in db.execute("SELECT spec FROM runs WHERE state IN ('preparing','running','paused','pausing','draining','cancelling','needs_attention')"))
    state = 'running' if active_fuzz else 'stopped'
    if fuzz.get('state') == 'starting':
        state = 'stopped'
    if fuzz.get('state') in {'stopping', 'error'}:
        state = fuzz['state']
    retest = store.get('retest_control', {})
    return dict(fuzz=dict(state=state, request_id=fuzz.get('request_id')),
                retest=dict(state='stopping' if store.get('retest_stopping', False) else 'running' if retest_enabled(store) else 'stopped',
                            request_id=retest.get('request_id')))

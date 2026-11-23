"""Apply platform session commands while keeping the management connection alive."""
import json
import time


def reconcile(manager):
    store = manager.store
    with store.connect() as db:
        unresolved = [row['id'] for row in db.execute("SELECT id FROM runs WHERE state IN ('needs_attention','draining','cancelling')")]
    store.put('session_state', 'running' if manager.processes or unresolved else 'stopped')
    if store.get('client_switch', {}).get('state') in {'draining', 'staging', 'requested', 'activating', 'stage_failed'}:
        return
    path = store.home / 'daemon-control.json'
    if not path.exists():
        return
    try:
        command = json.loads(path.read_text())
    except (OSError, ValueError):
        return
    if not isinstance(command, dict):
        return
    identity = command.get('request_id')
    if not identity or command.get('state') not in {'running', 'stopped', 'restart'}:
        return
    result = store.get('session_control', {})
    if result.get('request_id') == identity and result.get('status') in {'succeeded', 'failed'}:
        return
    restarting = command['state'] == 'restart'
    drained = result.get('request_id') == identity and result.get('restart_drained', False)
    try:
        if command['state'] == 'stopped' or (restarting and not drained):
            from autofusex.client.tasks import action
            started = result.get('started', time.time()) if result.get('request_id') == identity else time.time()
            for run in set(manager.processes) | set(unresolved):
                # Recovered workers retain the task lock and read their private
                # control file. Never signal a PID recovered from the database.
                if run not in manager.processes:
                    from autofusex.client.artifacts import atomic
                    atomic(store.home / 'runs' / run / 'control.json', {'action': 'force-stop' if time.time() - started > 300 else 'stop'})
                    continue
                action(store, run, 'force-stop' if time.time() - started > 300 else 'stop', manager.processes)
            state = 'stopping' if manager.processes or unresolved or manager.active else 'stopped'
            store.put('session_control', dict(request_id=identity, state=state,
                      status='running' if state == 'stopping' or restarting else 'succeeded',
                      restart_drained=restarting and state == 'stopped', started=started))
        elif not manager.active:
            from autofusex.client.local_session import start
            if result.get('request_id') != identity or not result.get('run_ids'):
                started = manager.start_session() if hasattr(manager, 'start_session') else start(manager)
                runs = started.get('run_ids') or [started['run_id']]
                store.put('session_control', dict(request_id=identity, run_ids=runs, state='starting', status='running', restart_drained=restarting))
            else:
                with store.connect() as db:
                    states = [db.execute('SELECT state FROM runs WHERE id=?', (run,)).fetchone() for run in result['run_ids']]
                if all(row and row['state'] in {'running', 'completed'} for row in states):
                    result.update(state='running', status='succeeded')
                elif any(not row or row['state'] in {'failed', 'stopped', 'cancelled', 'interrupted', 'needs_attention'} for row in states):
                    result.update(state='error', status='failed', error='WorkerStartFailed')
                store.put('session_control', result)
    except Exception as exc:
        store.put('session_control', dict(request_id=identity, state='error', status='failed', error=type(exc).__name__))

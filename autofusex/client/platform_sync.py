"""Platform heartbeat/configuration and opt-in per-run result upload."""
import importlib.metadata
import json
import os
import socket
import time
import sqlite3
from pathlib import Path
from autofusex.client.artifacts import atomic
from autofusex.client.paths import retest_directory
from autofusex.client.platform_transport import API, Client


def runtime_snapshots(home) -> list:
    """Read current task snapshots published by the local execution reporter."""
    try:
        boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    except OSError:
        boot = 'unknown'
    now = time.monotonic()
    runs = []
    for path in sorted((home / 'runtime').glob('*.json'), key=lambda p: p.stat().st_mtime, reverse=True)[:64]:
        try:
            saved = json.loads(path.read_text())
            age = now - saved['monotonic']
            if saved['boot'] != boot or age < 0 or age > 86400:
                continue
            runs.append(dict(saved['sample'], age=age))
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if len(runs) >= 32:
            break
    return runs


def pending_uploads(store) -> int:
    """Count local fuzz attempts after each run's persisted upload cursor."""
    total = 0
    with store.connect() as db:
        runs = list(db.execute("SELECT id, spec FROM runs"))
    for run in runs:
        try:
            spec = json.loads(run["spec"])["spec"]
        except (TypeError, ValueError, KeyError):
            continue
        if spec.get("kind") != "fuzz_task" or not spec.get("results", {}).get("upload"):
            continue
        directory = store.home / "runs" / run["id"]
        results = directory / "results.sqlite"
        progress = directory / "upload.sqlite"
        if not results.exists():
            continue
        try:
            source = sqlite3.connect(results)
            cursor = 0
            if progress.exists():
                state = sqlite3.connect(progress)
                row = state.execute("SELECT value FROM progress WHERE key='fuzz:cursor'").fetchone()
                cursor = int(row[0]) if row else 0
                state.close()
            total += source.execute("SELECT COUNT(*) FROM attempts WHERE rowid>?", (cursor,)).fetchone()[0]
            source.close()
        except (OSError, sqlite3.Error, ValueError, TypeError):
            continue
    return total


def configured_task_devices(store, inventory) -> list:
    """Return the logical NPU cards selected by currently running managed tasks."""
    devices = []
    try:
        with store.connect() as db:
            rows = db.execute("SELECT state, spec FROM runs WHERE state IN ('preparing','running','paused','pausing','draining','cancelling')").fetchall()
        for row in rows:
            payload = json.loads(row['spec'])
            spec = payload.get('spec', payload)
            execution = spec.get('execution') or {}
            if not any(backend != 'cpu' for backend in execution.get('backends') or []):
                continue
            selected = execution.get('devices') or []
            if 'all' in selected:
                selected = inventory.get('devices') or []
            devices.extend(str(device) for device in selected if str(device).startswith('npu:'))
    except (OSError, ValueError, TypeError, KeyError, sqlite3.Error):
        return []
    return list(dict.fromkeys(devices))


def synchronize(manager, client):
    store=manager.store
    api=API(client['endpoint'],client['token'],timeout=15)
    # Enable the execution reporter before managed workers are started.  The
    # reporter is intentionally opt-in so standalone/local runs never write
    # fleet telemetry, while managed tasks publish their configured devices.
    try:
        atomic(store.home / 'platform-client.json', {'endpoint': client['endpoint'], 'pid': os.getpid()})
    except OSError:
        pass
    # Use the same deterministic identity as the standalone platform client.
    # The old daemon path generated a random UUID, so `afx start` and
    # `afx sync run` on the same host appeared as two clients.
    from autofusex.execution.environment import collect
    env = collect(store.home)
    home = str(store.home.resolve())
    identity = env.get('installation_id')
    if not identity:
        raise ValueError('managed client installation configuration is missing')
    store.put('machine_id', identity[:128])
    version=importlib.metadata.version('autofusex')
    from autofusex.execution.scheduling.devices import device_inventory_info
    inventory = device_inventory_info()
    from autofusex.client.execution_runtime import facts
    cached = store.get('execution_facts', {})
    execution = cached.get('value')
    if time.time() - cached.get('at', 0) > 300:
        try:
            execution = facts(store)
            store.put('execution_facts', dict(at=time.time(), value=execution))
        except (OSError, ValueError, __import__('subprocess').SubprocessError):
            pass
    if execution:
        env.update(execution['environment'])
        inventory = execution['inventory']
    control_path = store.home / 'daemon-control.json'
    session = store.get('session_control', {})
    current_state = store.get('session_state', 'stopped')
    update_ack = None
    request_id = store.get('platform_update_request')
    if request_id:
        with store.connect() as db:
            operation = db.execute("SELECT state FROM operations WHERE request_key=?", ('platform-update:'+request_id,)).fetchone()
        if operation and operation['state'] == 'succeeded':
            update_ack = request_id
    from autofusex.client.update_delivery import progress, receive
    claims_path = retest_directory(store.home) / 'claims.json'
    runtime = runtime_snapshots(store.home)
    selected_devices = configured_task_devices(store, inventory)
    if selected_devices and not any(run.get('state') == 'running' for run in runtime):
        runtime.append(dict(run_id='managed-config', state='running',
                            devices=[dict(logical=device) for device in selected_devices]))
    heartbeat=dict(
        machine_id=identity[:128], client_home=home[:512],
        hostname=socket.gethostname(), npu_arch=env.get('device_model') or inventory['model'],
        afx_version=version, update_ack=update_ack, update_status=progress(store),
        backend_versions={'af-mlir': env.get('cann'), 'triton-npu': env.get('triton'),
                          'cpu': env.get('torch')},
        devices=inventory['devices'], counters={}, daemon_state=current_state if current_state in {'running','stopped'} else None,
        daemon_request_id=session.get('request_id') if session.get('status') == 'succeeded' else None,
        # `busy` is reserved for a client that is actively executing a
        # server-assigned retest. Ordinary fuzz work must remain Online in the
        # fleet view; otherwise the UI incorrectly labels it "复测中".
        busy=bool(claims_path.exists() and json.loads(claims_path.read_text()).get('tasks')),
        pending_uploads=pending_uploads(store),
        runtime=runtime,
    )
    from autofusex.client import workload_control
    heartbeat['workload_states'] = workload_control.states(store)
    retest_child = getattr(manager, '_retest_process', None)
    if retest_child is not None and retest_child.poll() is None:
        heartbeat['daemon_request_id'] = None
    beat=api.call('POST','/api/client/heartbeat',heartbeat, attempts=1)
    effective=api.call('GET','/api/config/me',attempts=1)
    config = effective['config']
    atomic(store.home/'task.json',config)
    store.put('config_state',dict(revision=effective['revision'],source=effective['source']))
    if beat.get('update_to'):
        receive(store, beat['update_to'], beat.get('update_request_id'))
        heartbeat['update_status'] = progress(store)
        api.call('POST','/api/client/heartbeat',heartbeat, attempts=1)
    if beat.get('daemon_state') in {'running', 'stopped', 'restart'}:
        state = beat['daemon_state']
        atomic(control_path, {'state': state, 'request_id': beat.get('daemon_request_id')})
        store.put('shutdown_when_idle', False)
        # Legacy lifecycle commands and upgrades still stop all execution.
        if state == 'stopped':
            workload_control.receive(store, dict(retest=dict(state='stopped', request_id=beat.get('daemon_request_id'))))
    else:
        workload_control.receive(store, beat.get('workload_controls', {}))
    from autofusex.client.source_jobs import synchronize as source_jobs
    source_jobs(manager, api, identity[:128])
    with store.connect() as db:
        runs=list(db.execute('SELECT id,spec FROM runs'))
    reported = {'executions': 0, 'uploads': 0, 'upload_errors': 0, 'retests': 0}
    for run in runs:
        spec=json.loads(run['spec'])['spec']
        if not spec['results']['upload']: continue
        directory=store.home/'runs'/run['id']
        if spec['kind'] != 'fuzz_task':
            result_file=directory/'result.json'
            receipt=directory/'source-upload.json'
            if result_file.exists() and not receipt.exists():
                from autofusex.workloads.imported.registry import export
                api.call('POST','/api/source-cases',export(store.home,spec['workload']['case_key']))
                uploaded=api.call('POST','/api/source-results',dict(spec=spec,result=json.loads(result_file.read_text())))
                from autofusex.client.source_jobs import upload_artifacts
                job=api.call('GET','/api/source-jobs/'+uploaded['job_id'])
                upload_artifacts(api,directory,job['job_id'],job['lease_id'],json.loads(result_file.read_text()))
                atomic(receipt,uploaded)
            continue
        if not (directory/'results.sqlite').exists():continue
        uploader=Client(api,str(directory/'results.sqlite'),directory/'upload.sqlite')
        try:
            uploader.upload_once()
            for name, value in uploader.counters.take().items():
                reported[name] += value
        finally:uploader.state.conn.close()
    if any(reported.values()):
        heartbeat['counters'] = reported
        api.call('POST', '/api/client/heartbeat', heartbeat, attempts=1)

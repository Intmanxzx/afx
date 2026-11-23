"""Platform heartbeat/configuration and opt-in per-run result upload."""
import importlib.metadata
import json
import socket
import time
from autofusex.client.artifacts import atomic
from autofusex.client.paths import retest_directory
from autofusex.client.platform_transport import API, Client


def synchronize(manager, client):
    store=manager.store
    api=API(client['endpoint'],client['token'],timeout=15)
    # Use the same deterministic identity as the standalone platform client.
    # The old daemon path generated a random UUID, so `afx start` and
    # `afx platform client run` on the same host appeared as two clients.
    from autofusex.execution.environment import collect
    env = collect(store.home)
    home = str(store.home.resolve())
    identity = env.get('installation_id')
    if not identity:
        raise ValueError('managed client installation configuration is missing')
    store.put('machine_id', identity[:128])
    active_path=store.home/'active-client.json'
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
        pending_uploads=0,
        runtime=[]
    )
    beat=api.call('POST','/api/client/heartbeat',heartbeat, attempts=1)
    effective=api.call('GET','/api/config/me',attempts=1)
    config = effective['config']
    # A device selected during `afx init --device` is a local operator
    # override.  Keep the server config for every other field, but never let
    # a later sync silently replace that explicit local selection.
    local = json.loads((store.home/'client.json').read_text())
    if local.get('devices'):
        config.setdefault('execution', {})['devices'] = local['devices']
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
    with store.connect() as db:
        runs=list(db.execute('SELECT id,spec FROM runs'))
    for run in runs:
        spec=json.loads(run['spec'])['spec']
        if not spec['results']['upload']: continue
        directory=store.home/'runs'/run['id']
        if not (directory/'results.sqlite').exists():continue
        uploader=Client(api,str(directory/'results.sqlite'),directory/'upload.sqlite')
        try:uploader.upload_once()
        finally:uploader.state.conn.close()

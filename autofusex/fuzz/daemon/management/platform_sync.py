"""Platform heartbeat/configuration and opt-in per-run result upload."""
import importlib.metadata
import json
import socket
from .artifacts import atomic
from autofusex.platform.client import API, Client


def synchronize(manager, client):
    store=manager.store
    api=API(client['endpoint'],client['token'],timeout=15)
    # Use the same deterministic identity as the standalone platform client.
    # The old daemon path generated a random UUID, so `afx start` and
    # `afx platform client run` on the same host appeared as two clients.
    from autofusex.fuzz.client.environment import collect
    env = collect(store.home)
    home = str(store.home.resolve())
    identity = (env.get('installation_id') or socket.gethostname())[:128]
    store.put('machine_id', identity[:128])
    active_path=store.home/'active-client.json'
    version=json.loads(active_path.read_text())['version'] if active_path.exists() else importlib.metadata.version('autofusex')
    from ...client.scheduling.devices import device_inventory
    beat=api.call('POST','/api/client/heartbeat',dict(
        machine_id=identity[:128], client_home=home[:512],
        hostname=socket.gethostname(), npu_arch=env.get('device_model'),
        afx_version=version,
        backend_versions={'af-mlir': env.get('cann'), 'triton-npu': env.get('triton'),
                          'cpu': env.get('torch')},
        devices=device_inventory(), counters={},
        # `busy` is reserved for a client that is actively executing a
        # server-assigned retest. Ordinary fuzz work must remain Online in the
        # fleet view; otherwise the UI incorrectly labels it "复测中".
        busy=False, pending_uploads=0, runtime=[]), attempts=1)
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
        store.submit('client.update',dict(version=beat['update_to'],package='autofusex'),
                     'platform-update:'+str(beat.get('update_request_id') or beat['update_to']))
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

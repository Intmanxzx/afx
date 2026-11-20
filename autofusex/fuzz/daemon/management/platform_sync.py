"""Platform heartbeat/configuration and opt-in per-run result upload."""
import importlib.metadata
import json
import socket
import uuid
from .artifacts import atomic
from autofusex.platform.client import API, Client


def synchronize(manager, client):
    store=manager.store
    api=API(client['endpoint'],client['token'],timeout=15)
    identity=store.get('machine_id')
    if identity is None:
        identity=str(uuid.uuid4());store.put('machine_id',identity)
    active_path=store.home/'active-client.json'
    version=json.loads(active_path.read_text())['version'] if active_path.exists() else importlib.metadata.version('autofusex')
    beat=api.call('POST','/api/client/heartbeat',dict(machine_id=identity,hostname=socket.gethostname(),afx_version=version,busy=bool(manager.processes)),attempts=1)
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
        store.submit('client.update',dict(version=beat['update_to'],package='autofusex'),'platform-update:'+beat['update_to'])
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

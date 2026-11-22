"""Idempotent platform upgrade delivery and progress reporting."""
import importlib.metadata
import json
from .client_updates import _canonical_version


def progress(store):
    with store.connect() as db:
        row=db.execute("SELECT id,state,payload FROM operations WHERE kind='client.update' ORDER BY created DESC LIMIT 1").fetchone()
    if not row:return {'state':'idle'}
    return dict(state=row['state'],version=json.loads(row['payload'])['version'])


def receive(store, version, request_id):
    version=_canonical_version(version)
    with store.connect() as db:
        busy=db.execute("SELECT id FROM operations WHERE kind='client.update' AND state IN ('queued','running','activating') LIMIT 1").fetchone()
    if busy:return 'upgrading'
    store.put('platform_update_request',request_id)
    operation=store.submit('client.update',dict(version=version,package='autofusex'),'platform-update:'+str(request_id or version))
    if _canonical_version(importlib.metadata.version('autofusex')) == version:
        store.state(operation['id'],'succeeded',dict(state='up_to_date',version=version))
        return 'up_to_date'
    return operation['state']

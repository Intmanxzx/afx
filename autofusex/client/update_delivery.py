"""Idempotent platform upgrade delivery and progress reporting."""
import importlib.metadata
import json
from autofusex.client.client_updates import _canonical_version


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
    with store.connect() as db:
        manifests = [json.loads(r[0]) for r in db.execute("SELECT manifest FROM environments WHERE state='ready'")]
    if _canonical_version(importlib.metadata.version('autofusex')) == version and all(m.get('afx_version') == version for m in manifests):
        store.state(operation['id'],'succeeded',dict(state='up_to_date',version=version))
        return 'up_to_date'
    return operation['state']

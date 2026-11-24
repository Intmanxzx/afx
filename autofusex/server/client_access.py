"""Operator-controlled admission gate; independent of client software versions."""
import json
import os
from pathlib import Path
import tempfile

FILENAME = 'client-access.json'


def validate(value):
    if not isinstance(value, dict) or set(value) != {'enabled', 'clients'}:
        raise ValueError('client access requires enabled and clients')
    if type(value['enabled']) is not bool or not isinstance(value['clients'], list):
        raise ValueError('invalid client access policy')
    if len(value['clients']) > 10000:
        raise ValueError('too many allowed clients')
    for entry in value['clients']:
        if not isinstance(entry, dict) or set(entry) != {'user_id', 'machine_id'}:
            raise ValueError('each client requires user_id and machine_id')
        if any(not isinstance(entry[k], str) or not 1 <= len(entry[k]) <= 128 for k in entry):
            raise ValueError('invalid client identity')
    return value


def read(home):
    path = Path(home) / FILENAME
    if not path.exists():
        return {'enabled': False, 'clients': []}
    return validate(json.loads(path.read_text()))


def write(home, value):
    value = validate(value)
    home = Path(home)
    home.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.client-access-', dir=home)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, home / FILENAME)
    finally:
        Path(name).unlink(missing_ok=True)


def protected(path, method):
    return path.startswith('/api/client/') or (
        method == 'POST' and (path in {'/api/source-cases', '/api/source-results'} or path.startswith('/api/source-jobs/') ))


def allowed(store, principal, path, method, body, policy=None):
    if not protected(path, method):
        return True
    policy = read(store.dir) if policy is None else policy
    if not policy['enabled']:
        return True
    machine = body.get('machine_id')
    if path == '/api/client/executions':
        machine = (body.get('client') or {}).get('machine_id')
    elif path.startswith('/api/client/retests/') and path.endswith('/complete'):
        task = path.split('/')[-2]
        row = store.reader().execute('SELECT c.machine_id FROM retest_tasks r JOIN clients c ON c.client_id=r.client_id WHERE r.task_id=? AND c.user_id=? AND r.lease_id=?', (task, principal.user_id, body.get('lease_id'))).fetchone()
        machine = row[0] if row else None
    elif path.startswith('/api/source-jobs/') and path != '/api/source-jobs/claim':
        job = path.split('/')[3]
        row = store.reader().execute('SELECT machine_id FROM source_jobs WHERE job_id=? AND owner=? AND lease_id=?', (job, principal.user_id, body.get('lease_id'))).fetchone()
        machine = row[0] if row else None
    # Source publication in the old wire format carries no machine identity.
    # During maintenance it stays closed rather than trusting a shared user token.
    return any(entry['user_id'] == principal.user_id and entry['machine_id'] == machine for entry in policy['clients'])

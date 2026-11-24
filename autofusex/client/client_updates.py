"""Drain cases and cache wheels for the single installation and rollback."""
import sys
import uuid
import time
import logging
from autofusex.client.artifacts import command

LOG = logging.getLogger('autofusex.client_update')


def _canonical_version(value):
    parts = value.split('.')
    if not parts or any(not part.isdigit() for part in parts):
        raise ValueError('invalid client version')
    return '.'.join(str(int(part)) for part in parts)


def install(store, payload, processes, operation_id=None):
    LOG.info('update.start operation=%s version=%s processes=%d', operation_id, payload.get('version'), len(processes))
    store.event(operation_id, 'update.start version=%s processes=%d' % (payload.get('version'), len(processes)))
    from autofusex.client.upgrade_drain import BUSY, begin, wait
    if store.get('client_switch', {}).get('state') in BUSY:
        raise ValueError('another activation is in progress')
    if store.get('supervisor_protocol') != 3:
        raise ValueError('control migration required before upgrading')
    _canonical_version(payload['version'])
    switch = dict(state='draining', old_runs=list(processes), resume=bool(processes),
                  operation_id=operation_id)
    store.put('client_switch', switch)
    begin(store, processes)
    wait(store, processes)
    switch['state'] = 'staging'
    switch['install_started'] = time.time()
    store.put('client_switch', switch)
    store.event(operation_id, 'update.staging version=%s' % payload.get('version'))
    try:
        return stage(store, payload, processes, operation_id, switch)
    except Exception:
        LOG.exception('update.stage_failed operation=%s version=%s', operation_id, payload.get('version'))
        store.event(operation_id, 'update.stage_failed')
        switch.update(state='stage_failed')
        store.put('client_switch', switch)
        raise


def stage(store, payload, processes, operation_id=None, switch=None, target_wheels=None):
    if store.get('supervisor_protocol') != 3:
        raise ValueError('restart with the single-install supervisor before upgrading')
    import importlib.metadata
    version = _canonical_version(payload['version'])
    previous_version = importlib.metadata.version('autofusex')
    root = store.home / 'update-artifacts' / str(uuid.uuid4())
    root.mkdir(parents=True, mode=0o700)
    for name, release in [('target', version), ('previous', previous_version)]:
        wheelhouse = root / name
        wheelhouse.mkdir()
        if name == 'target' and target_wheels is not None:
            import shutil
            for wheel in target_wheels.glob('*.whl'):
                shutil.copy2(wheel, wheelhouse/wheel.name)
            continue
        args = [sys.executable, '-m', 'pip', 'download', '--only-binary=:all:',
                '--dest', str(wheelhouse), 'autofusex==' + release]
        if payload.get('index_url'): args.extend(['--index-url', payload['index_url']])
        args.extend(payload.get('extra_args', []))
        command(args, timeout=1800)
    target = dict(version=version, python=sys.executable, wheels=str(root/'target'))
    previous = dict(version=previous_version, python=sys.executable, wheels=str(root/'previous'))
    store.put('client_switch', dict(switch or {}, state='requested', target=target, previous=previous,
                                   operation_id=operation_id, drained=True))
    return dict(version=version, state='staged', python=sys.executable)

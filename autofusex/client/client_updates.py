"""Drain Cases, then stage matching control and execution releases."""
import json
import os
import subprocess
import sys
import uuid
import time
import logging
from autofusex.client.artifacts import atomic, command

LOG = logging.getLogger('autofusex.client_update')


def _canonical_version(value):
    parts = value.split('.')
    if not parts or any(not part.isdigit() for part in parts):
        raise ValueError('invalid client version')
    return '.'.join(str(int(part)) for part in parts)


def _venv_python(path):
    return path / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')


def install(store, payload, processes, operation_id=None):
    LOG.info('update.start operation=%s version=%s processes=%d', operation_id, payload.get('version'), len(processes))
    store.event(operation_id, 'update.start version=%s processes=%d' % (payload.get('version'), len(processes)))
    from autofusex.client.upgrade_drain import BUSY, begin, wait
    if store.get('client_switch', {}).get('state') in BUSY:
        raise ValueError('another activation is in progress')
    if store.get('supervisor_protocol') != 2:
        raise ValueError('control migration required before upgrading')
    if store.get('active_tool'):
        raise ValueError('signed tool activation must be migrated before client upgrades')
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


def stage(store, payload, processes, operation_id=None, switch=None):
    if store.get('supervisor_protocol') != 2:
        raise ValueError('control migration required before upgrading')
    if store.get('active_tool'):
        raise ValueError('signed tool activation must be migrated before client upgrades')
    pending = store.get('client_switch', {})
    if pending.get('state') in {'requested', 'activating'}:
        raise ValueError('another activation is in progress')
    version = _canonical_version(payload['version'])
    # Retrying a version must not overwrite a release used by rollback.
    root = store.home / 'client-releases' / version / str(uuid.uuid4())
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    python = _venv_python(root / 'control-venv')
    if not python.exists():
        command([sys.executable, '-m', 'venv', str(root / 'control-venv')], timeout=300)
    pip = [str(python), '-m', 'pip', 'install', '--upgrade', '--no-input',
           '--disable-pip-version-check', 'autofusex==' + version]
    if payload.get('index_url'):
        pip.extend(['--index-url', payload['index_url']])
    pip.extend(payload.get('extra_args', []))
    command(pip, timeout=1800)
    probe = 'import sys,json,autofusex,importlib.metadata as m; from pathlib import Path; import autofusex.client.daemon; assert Path(autofusex.__file__).is_relative_to(Path(sys.prefix)); print(m.version("autofusex"))'
    verified = subprocess.check_output([str(python), '-I', '-c', probe], text=True, timeout=60).strip()
    if verified != version:
        raise ValueError('installed version mismatch')
    target = dict(version=version, python=str(python))
    target['environments'] = stage_execution(store, root, payload, version)
    atomic(root / 'ready.json', target)
    previous_path = store.home / 'active-client.json'
    previous = json.loads(previous_path.read_text()) if previous_path.exists() else dict(
        python=sys.executable, version=__import__('importlib.metadata', fromlist=['version']).version('autofusex'))
    # Work has drained; only the supervisor may activate either environment.
    store.put('client_switch', dict(switch or {}, state='requested', target=target, previous=previous,
                                   operation_id=operation_id, drained=True))
    return dict(version=version, state='staged', python=str(python))


def stage_execution(store, root, payload, version):
    import hashlib
    from pathlib import Path
    with store.connect() as db:
        rows = db.execute("SELECT id,manifest FROM environments WHERE state='ready'").fetchall()
    result = []
    for row in rows:
        old = json.loads(row['manifest'])
        env = os.environ.copy(); env.pop('PYTHONPATH', None); env.update(old['env'])
        paths = json.loads(subprocess.check_output([old['python'], '-c',
            'import sys,json; print(json.dumps([p for p in sys.path if p.endswith(("site-packages","dist-packages"))]))'],
            env=env, text=True, timeout=60))
        directory = root / ('execution-' + hashlib.sha256(row['id'].encode()).hexdigest()[:16])
        command([old['python'], '-m', 'venv', str(directory)], timeout=300)
        python = _venv_python(directory)
        site = subprocess.check_output([str(python), '-c',
            'import sysconfig; print(sysconfig.get_path("purelib"))'], text=True, timeout=60).strip()
        Path(site, 'site-runtime.pth').write_text('\n'.join(paths) + '\n')
        pip = [str(python), '-m', 'pip', 'install', '--ignore-installed', '--no-deps',
               '--no-input', 'autofusex==' + version]
        if payload.get('index_url'):
            pip.extend(['--index-url', payload['index_url']])
        pip.extend(payload.get('extra_args', []))
        command(pip, timeout=1800)
        probe = 'import sys,autofusex; from pathlib import Path; assert Path(autofusex.__file__).is_relative_to(Path(sys.prefix)); print(autofusex.__version__)'
        if subprocess.check_output([str(python), '-I', '-c', probe], env=env, text=True, timeout=60).strip() != version:
            raise ValueError('execution version mismatch')
        result.append(dict(id=row['id'], previous=old, target=dict(old, python=str(python), afx_version=version)))
    return result

"""Stage isolated control releases; only the supervisor may activate them."""
import json
import os
import subprocess
import sys
from .artifacts import atomic, command


def _canonical_version(value):
    parts = value.split('.')
    if not parts or any(not part.isdigit() for part in parts):
        raise ValueError('invalid client version')
    return '.'.join(str(int(part)) for part in parts)


def _venv_python(path):
    return path / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')


def install(store, payload, processes, operation_id=None):
    if store.get('supervisor_protocol') != 2:
        raise ValueError('control migration required before upgrading')
    if store.get('active_tool'):
        raise ValueError('signed tool activation must be migrated before client upgrades')
    pending = store.get('client_switch', {})
    if pending.get('state') in {'requested', 'activating'}:
        raise ValueError('another activation is in progress')
    version = _canonical_version(payload['version'])
    root = store.home / 'client-releases' / version
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
    probe = 'import sys,json,autofusex,importlib.metadata as m; from pathlib import Path; import autofusex.fuzz.daemon.runtime; assert Path(autofusex.__file__).is_relative_to(Path(sys.prefix)); print(m.version("autofusex"))'
    verified = subprocess.check_output([str(python), '-I', '-c', probe], text=True, timeout=60).strip()
    if verified != version:
        raise ValueError('installed version mismatch')
    target = dict(version=version, python=str(python))
    atomic(root / 'ready.json', target)
    previous_path = store.home / 'active-client.json'
    previous = json.loads(previous_path.read_text()) if previous_path.exists() else dict(
        python=sys.executable, version=__import__('importlib.metadata', fromlist=['version']).version('autofusex'))
    # Staging never interrupts work or changes the active pointer.
    store.put('client_switch', dict(state='requested', target=target, previous=previous,
                                   operation_id=operation_id, resume=bool(processes)))
    return dict(version=version, state='staged', python=str(python))

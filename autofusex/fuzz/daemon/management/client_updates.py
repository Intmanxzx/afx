"""Install an explicitly requested client release from PyPI."""
import os
import subprocess
import sys
import time
import json

from .artifacts import atomic, command

def _venv_python(path):
    return path / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')

def install(store, payload, processes, operation_id=None):
    # Ask active client workers to drain at their safe boundary.  This is
    # finite even for continuous campaigns; an unresponsive worker is first
    # terminated and then left for normal recovery/reconciliation.
    for identity, child in list(processes.items()):
        control = store.home / 'runs' / identity / 'control.json'
        if control.exists():
            atomic(control, dict(action='stop', reason='client_update'))
    deadline = time.monotonic() + 300
    while processes and time.monotonic() < deadline:
        for identity, child in list(processes.items()):
            if child.poll() is not None:
                processes.pop(identity, None)
        if processes:
            time.sleep(.2)
    if processes:
        for child in processes.values():
            child.terminate()
        deadline = time.monotonic() + 30
        while processes and time.monotonic() < deadline:
            for identity, child in list(processes.items()):
                if child.poll() is not None:
                    processes.pop(identity, None)
            if processes:
                time.sleep(.2)
    if processes:
        for child in processes.values():
            child.kill()
        raise RuntimeError('client did not stop for update')
    version = payload['version']
    root = store.home / 'client-releases' / version
    venv = root / 'venv'
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    python = _venv_python(venv)
    if not python.exists():
        command([sys.executable, '-m', 'venv', str(venv)], timeout=300)
    pip = [str(python), '-m', 'pip', 'install', '--upgrade', '--no-input',
           '--disable-pip-version-check', 'autofusex==' + version]
    if payload.get('index_url'):
        pip[4:4] = ['--index-url', payload['index_url']]
    pip.extend(payload.get('extra_args', []))
    command(pip, timeout=1800)
    check = [str(python), '-c',
             'import importlib.metadata; print(importlib.metadata.version("autofusex"))']
    verified = subprocess.check_output(check, text=True, stderr=subprocess.STDOUT, timeout=60).strip().splitlines()[-1:]
    if verified and verified[0] and verified[0] != version:
        raise ValueError('installed client version does not match requested version')
    atomic(root / 'ready.json', dict(version=version, python=str(python), package='autofusex'))
    atomic(store.home / 'active-client.json', dict(version=version, python=str(python)))
    if operation_id:
        store.event(operation_id, 'Client release installed from PyPI: autofusex==' + version)
    return dict(version=version, state='active', python=str(python), source='pypi')

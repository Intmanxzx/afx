"""Install an explicitly requested client release from PyPI."""
import os
import subprocess
import sys
import time
import json

from .artifacts import atomic, command

def _canonical_version(value):
    parts = value.split('.')
    if not parts or any(not part.isdigit() for part in parts):
        raise ValueError('invalid client version')
    return '.'.join(str(int(part)) for part in parts)

def _venv_python(path):
    return path / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')

def install(store, payload, processes, operation_id=None):
    # Ask active client workers to drain at their safe boundary.  This is
    # finite even for continuous campaigns; an unresponsive worker is first
    # terminated and then left for normal recovery/reconciliation.
    resume_specs=[]
    for identity in list(processes):
        path=store.home/'runs'/identity/'spec.json'
        if path.exists():
            spec=json.loads(path.read_text())['spec']
            if spec.get('termination',{}).get('mode')=='continuous':
                import copy
                spec=copy.deepcopy(spec)
                resources=spec['execution']['resources']
                if resources['policy']=='auto':
                    resources.pop('workers',None);resources.pop('compile_threads',None)
                resume_specs.append(spec)
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
        command([sys.executable, '-m', 'venv', '--system-site-packages', str(venv)], timeout=300)
        import site
        # Nested venvs do not inherit their parent's site-packages. Preserve
        # the site's Torch/NPU runtime without reinstalling it globally.
        locations=subprocess.check_output([str(python),'-c','import site; print(site.getsitepackages()[0])'],text=True,timeout=60).strip()
        from pathlib import Path
        (Path(locations)/'afx-runtime.pth').write_text('\n'.join(site.getsitepackages())+'\n')
    pip = [str(python), '-m', 'pip', 'install', '--upgrade', '--no-input',
           '--disable-pip-version-check', 'autofusex==' + version]
    if payload.get('index_url'):
        pip[4:4] = ['--index-url', payload['index_url']]
    pip.extend(payload.get('extra_args', []))
    command(pip, timeout=1800)
    check = [str(python), '-c',
             'import importlib.metadata; print(importlib.metadata.version("autofusex"))']
    verified = subprocess.check_output(check, text=True, stderr=subprocess.STDOUT, timeout=60).strip().splitlines()[-1:]
    if verified != [_canonical_version(version)]:
        raise ValueError('installed client version does not match requested version')
    atomic(root / 'ready.json', dict(version=version, python=str(python), package='autofusex'))
    if (store.home/'active-client.json').exists():
        atomic(store.home/'previous-client.json',json.loads((store.home/'active-client.json').read_text()))
    atomic(store.home / 'active-client.json', dict(version=version, python=str(python)))
    for index,spec in enumerate(resume_specs):
        store.submit('task.submit',spec,'after-update:'+str(operation_id or version)+':'+str(index))
    if operation_id:
        store.event(operation_id, 'Client release installed from PyPI: autofusex==' + version)
    return dict(version=version, state='active', python=str(python), source='pypi')

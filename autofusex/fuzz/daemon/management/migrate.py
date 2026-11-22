"""Explicit one-time migration from legacy launchers, preserving identity and data.

Run from the installed new control environment with --execution-python pointing
at the existing, independently maintained execution environment.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
from .artifacts import atomic
from .store import Store
from ..runtime import request


def migrate(home, execution_python):
    store = Store(home)
    execution_python = str(Path(execution_python).expanduser().absolute())
    # Fail before touching an existing manager if its execution stack is broken.
    subprocess.run([execution_python, '-c',
        'from autofusex.fuzz.client.generation.runtime import source_evidence; source_evidence(); import torch'],
        check=True, timeout=120)
    backup = store.home/'backups'/('control-migration-'+str(time.time_ns()))
    backup.mkdir(parents=True, mode=0o700)
    with sqlite3.connect(store.path) as src, sqlite3.connect(backup/'manager.sqlite') as dst:
        src.backup(dst)
    (backup/'manager.sqlite').chmod(0o600)
    for name in ('active-client.json','client.json','task.json','trust.json'):
        path=store.home/name
        if path.exists():
            target=backup/name;target.write_bytes(path.read_bytes());target.chmod(0o600)
    try:
        status=request(store.home, {'action':'status'})
    except OSError:
        status={'runs':[]}
    else:
        if any(r['state'] in {'preparing','running','paused','pausing','draining','cancelling','needs_attention'} for r in status['runs']):
            raise ValueError('stop and resolve existing tasks before migration; backup retained')
        request(store.home, {'action':'shutdown'})
    # Wait for ownership release, never kill a recovered PID.
    deadline=time.monotonic()+60
    with (store.home/'launcher.lock').open('a') as lock:
        while True:
            try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
            except BlockingIOError:
                if time.monotonic()>=deadline:raise TimeoutError('legacy launcher still owns installation')
                time.sleep(.2)
        with (store.home/'manager.lock').open('a') as manager_lock:
            try: fcntl.flock(manager_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError: raise RuntimeError('legacy manager still owns installation') from None
        manifest=dict(id='local',name='local',python=execution_python,env={
            key:os.environ[key] for key in ('ASCEND_HOME_PATH','ASCEND_OPP_PATH','LD_LIBRARY_PATH','PATH') if key in os.environ})
        with store.connect() as db:
            db.execute("INSERT OR REPLACE INTO environments VALUES (?,?,?,'ready')",('local','local',json.dumps(manifest)))
        import importlib.metadata
        atomic(store.home/'active-client.json',dict(python=sys.executable,version=importlib.metadata.version('autofusex')))
        store.put('active_tool',None)
        store.put('shutdown_when_idle',False)
    from .modern_cli import _legacy
    result=_legacy(['daemon','start','--data-dir',str(store.home)])
    if store.get('supervisor_protocol')!=2:
        raise RuntimeError('new supervisor did not become ready; backup retained')
    return dict(state='migrated',backup=str(backup),pid=result.get('pid'))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-dir',required=True)
    parser.add_argument('--execution-python',required=True)
    args=parser.parse_args()
    print(json.dumps(migrate(args.data_dir,args.execution_python)))


if __name__=='__main__':main()

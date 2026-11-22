"""Stable supervisor with local readiness check and same-schema automatic rollback."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from .runtime import request
from .management.store import Store


def run(home):
    store=Store(home)
    with (store.home/'launcher.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        store.put('launcher_enabled',True)
        store.put('supervisor_protocol',2)
        try:
            while True:
                active=store.get('active_tool')
                python=sys.executable
                pointer = store.home / 'active-client.json'
                if pointer.exists():
                    python = json.loads(pointer.read_text())['python']
                if active:
                    manifest=json.loads((store.home/'tools'/active/'ready.json').read_text())
                    from .management.updates import verify_manifest
                    verify_manifest(store.home,manifest['envelope']);python=manifest['python']
                store.put('normal_shutdown',False)
                try:
                    child=subprocess.Popen([python,'-m','autofusex','daemon','run','--data-dir',str(store.home)],stdin=subprocess.DEVNULL)
                except OSError:
                    activation = store.get('client_switch', {})
                    if activation.get('state') != 'activating': raise
                    from .management.activation import finish
                    finish(store, activation, False)
                    continue
                healthy=False
                for _ in range(100):
                    if child.poll() is not None:break
                    try:
                        status=request(store.home,dict(action='status'))
                        if status.get('pid') == child.pid:
                            healthy=True;break
                    except (OSError,ValueError):time.sleep(.1)
                if not healthy:
                    if child.poll()==0 and store.get('normal_shutdown'): return 0
                    if child.poll() is None:child.terminate();child.wait(timeout=10)
                    activation = store.get('client_switch', {})
                    if activation.get('state') == 'activating':
                        from .management.activation import finish
                        finish(store, activation, False)
                        continue
                    switch=store.get('tool_switch')
                    if switch and switch['state']=='activating':
                        store.put('active_tool',switch['previous']);switch['state']='rolled_back';store.put('tool_switch',switch)
                        continue
                    raise RuntimeError('management daemon failed local health check')
                activation = store.get('client_switch', {})
                if activation.get('state') == 'rolled_back' and activation.get('resume') and not activation.get('restored'):
                    request(store.home, dict(action='resume_upgrade' if activation.get('drained') else 'start_session'))
                    activation['restored'] = True
                    store.put('client_switch', activation)
                if activation.get('state') == 'activating':
                    from .management.activation import ready, finish
                    try:
                        if activation.get('resume'):
                            request(store.home, dict(action='resume_upgrade' if activation.get('drained') else 'start_session'))
                        deadline = time.monotonic() + 120
                        while time.monotonic() < deadline:
                            if child.poll() is not None:
                                raise RuntimeError('new control process exited')
                            if ready(store, activation, request(store.home, dict(action='status'))):
                                break
                            time.sleep(.2)
                        else:
                            raise TimeoutError('activation readiness timeout')
                    except Exception:
                        try:
                            if child.poll() is None:
                                request(store.home, dict(action='drain_upgrade' if activation.get('drained') else 'stop_session'))
                                deadline = time.monotonic() + 300
                                while True:
                                    try:
                                        request(store.home, dict(action='shutdown'))
                                        break
                                    except ValueError:
                                        if time.monotonic() >= deadline: raise
                                        time.sleep(.2)
                                child.wait(timeout=30)
                        except Exception:
                            # Never activate a second manager while the first may own workers.
                            raise RuntimeError('rollback requires resolved task ownership') from None
                        finish(store, activation, False)
                        continue
                    finish(store, activation, True)
                switch=store.get('tool_switch')
                if switch and switch['state']=='activating':
                    switch['state']='active';store.put('tool_switch',switch)
                switching=False
                while child.poll() is None:
                    activation = store.get('client_switch', {})
                    if activation.get('state') == 'stage_failed':
                        activation['state'] = 'rolled_back'
                        activation['drained'] = True
                        store.put('client_switch', activation)
                        if activation.get('resume'):
                            request(store.home, dict(action='resume_upgrade'))
                        activation['restored'] = True
                        store.put('client_switch', activation)
                    if activation.get('state') == 'requested':
                        from .management.activation import select, finish
                        if not activation.get('drained') and 'drain_started' not in activation:
                            activation['old_runs'] = [r['id'] for r in request(store.home, dict(action='status'))['runs']]
                            activation['drain_started'] = time.time()
                            store.put('client_switch', activation)
                            request(store.home, dict(action='stop_session'))
                        try:
                            request(store.home, dict(action='shutdown'))
                        except (ValueError, OSError):
                            if not activation.get('drained') and time.time() - activation['drain_started'] > 300:
                                finish(store, activation, False)
                            time.sleep(.2)
                            continue
                        child.wait(timeout=30)
                        select(store, activation)
                        switching=True
                        break
                    switch=store.get('tool_switch')
                    if switch and switch['state']=='requested':
                        try:request(store.home,dict(action='shutdown'))
                        except (ValueError,OSError):time.sleep(.2);continue
                        child.wait(timeout=30)
                        store.put('active_tool',switch['target']);switch['state']='activating';store.put('tool_switch',switch)
                        switching=True;break
                    time.sleep(.2)
                if not switching:
                    if child.returncode == 0 and store.get('normal_shutdown'): return 0
                    return child.returncode
        finally:
            store.put('launcher_enabled',False)
            store.put('supervisor_protocol',None)

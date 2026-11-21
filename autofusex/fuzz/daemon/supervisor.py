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
        try:
            while True:
                control = store.home / 'daemon-control.json'
                desired = 'running'
                if control.exists():
                    try: desired = json.loads(control.read_text()).get('state', 'running')
                    except (OSError, ValueError, TypeError): pass
                if desired == 'stopped':
                    # Keep the bootstrap alive while the managed daemon is stopped.
                    # The active client agent polls the service and can flip this marker
                    # back to running without requiring a local shell command.
                    try:
                        active = json.loads((store.home/'active-client.json').read_text()) if (store.home/'active-client.json').exists() else {}
                        python = active.get('python') if isinstance(active.get('python'), str) else sys.executable
                        subprocess.run([python, '-m', 'autofusex.fuzz.daemon.management.platform_agent', '--data-dir', str(store.home)],
                                       stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120)
                    except (OSError, ValueError, subprocess.SubprocessError): pass
                    time.sleep(2)
                    continue
                active=store.get('active_tool')
                python=sys.executable
                if active:
                    manifest=json.loads((store.home/'tools'/active/'ready.json').read_text())
                    from .management.updates import verify_manifest
                    verify_manifest(store.home,manifest['envelope']);python=manifest['python']
                store.put('normal_shutdown',False)
                child=subprocess.Popen([python,'-m','autofusex','daemon','run','--data-dir',str(store.home)],stdin=subprocess.DEVNULL)
                healthy=False
                for _ in range(100):
                    if child.poll() is not None:break
                    try:
                        request(store.home,dict(action='status'));healthy=True;break
                    except (OSError,ValueError):time.sleep(.1)
                if not healthy:
                    if child.poll()==0 and store.get('normal_shutdown'): continue
                    if child.poll() is None:child.terminate();child.wait(timeout=10)
                    switch=store.get('tool_switch')
                    if switch and switch['state']=='activating':
                        store.put('active_tool',switch['previous']);switch['state']='rolled_back';store.put('tool_switch',switch)
                        continue
                    raise RuntimeError('management daemon failed local health check')
                switch=store.get('tool_switch')
                if switch and switch['state']=='activating':
                    switch['state']='active';store.put('tool_switch',switch)
                switching=False
                while child.poll() is None:
                    switch=store.get('tool_switch')
                    if switch and switch['state']=='requested':
                        try:request(store.home,dict(action='shutdown'))
                        except (ValueError,OSError):time.sleep(.2);continue
                        child.wait(timeout=30)
                        store.put('active_tool',switch['target']);switch['state']='activating';store.put('tool_switch',switch)
                        switching=True;break
                    time.sleep(.2)
                if not switching:
                    if child.returncode == 0 and store.get('normal_shutdown'): continue
                    return child.returncode
        finally:
            store.put('launcher_enabled',False)

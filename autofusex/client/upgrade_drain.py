"""Whole-case upgrade admission fence and task-tree quiescence."""
import fcntl
import time
from pathlib import Path
from autofusex.client.artifacts import atomic

BUSY = {'draining', 'staging', 'requested', 'activating'}


def task_processes(home):
    marker = ('AFX_TASK_HOME=' + str(home)).encode()
    prefix = (str(home) + '/runs/').encode()
    found = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            if marker in (entry/'environ').read_bytes().split(b'\0') or prefix in (entry/'cmdline').read_bytes():
                found.append(int(entry.name))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return found


def wait(store, processes, quiet_seconds=5):
    quiet = None
    while True:
        with (store.home/'managed-task.lock').open('a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                owned = False
            except BlockingIOError:
                owned = True
        busy = bool(processes) or owned or bool(task_processes(store.home))
        now = time.monotonic()
        if busy:
            quiet = None
        elif quiet is None:
            quiet = now
        elif now - quiet >= quiet_seconds:
            store.put('upgrade_quiet', dict(completed_at=time.time(), duration=now-quiet))
            return
        time.sleep(.2)


def begin(store, processes):
    runs = list(processes)
    for run in runs:
        atomic(store.home/'runs'/run/'control.json', {'action': 'upgrade'})
    return runs

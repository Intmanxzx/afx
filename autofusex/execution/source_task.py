"""Bounded subprocess ownership for fixed-source and whole-model tasks."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from autofusex.client.artifacts import atomic
from autofusex.contracts.configuration import quantity


def run(path, prepared, ownership_fd=None):
    directory = Path(prepared['directory'])
    env = os.environ.copy()
    threads = prepared['spec']['execution']['resources'].get('compile_threads',1)
    env['TORCHINDUCTOR_COMPILE_THREADS'] = str(threads)
    child = subprocess.Popen([sys.executable,'-m','autofusex.workloads.model.task',str(path)],
                             env=env, start_new_session=True, pass_fds=(() if ownership_fd is None else (ownership_fd,)))
    memory = prepared['spec']['execution']['resources'].get('memory_budget')
    memory_limit = quantity(memory) if memory else None
    deadline = time.monotonic() + quantity(prepared['spec']['execution']['timeout'])
    try:
        while child.poll() is None:
            control = json.loads((directory/'control.json').read_text()).get('action')
            from autofusex.execution.scheduling.budgets import group_rss
            memory_exceeded = memory_limit is not None and group_rss(child.pid) > memory_limit
            if control in {'stop','force-stop'} or time.monotonic() >= deadline or memory_exceeded:
                os.killpg(child.pid, signal.SIGTERM)
                try: child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL); child.wait()
                reason = 'memory_budget_exceeded' if memory_exceeded else 'timeout' if control not in {'stop','force-stop'} else control
                atomic(directory/'state.json', dict(state='failed' if reason in {'timeout', 'memory_budget_exceeded'} else 'stopped', reason=reason))
                return 1
            time.sleep(.1)
        return child.returncode
    finally:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try: child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL); child.wait()

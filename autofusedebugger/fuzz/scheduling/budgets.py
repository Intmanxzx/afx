"""Managed CPU budget policy and Linux process-group sampling.

Sampling is an admission/termination guard, NOT cgroup/RLIMIT hard isolation.
A task may overshoot between samples. Shared libraries count in each process RSS.
Only disposable per-attempt scratch is measured/removed, never user caches.
"""
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import shutil
import sys

from ..resources import resource_inventory


@dataclass(frozen=True)
class Budget:
    compile_threads: int = 1
    cpu_workers: int = 1
    memory_budget_bytes: int = None
    cache_budget_bytes: int = None
    disk_reserve_bytes: int = 64 * 1024 * 1024

    def __post_init__(self):
        for key, value in asdict(self).items():
            if value is None and key in {'memory_budget_bytes', 'cache_budget_bytes'}:
                continue
            if type(value) is not int or value < 1:
                raise ValueError(f'{key} must be a positive integer')
        if self.memory_budget_bytes is not None and sys.platform != 'linux':
            raise ValueError('managed memory sampling requires Linux /proc')

    @classmethod
    def configured(cls, config, compile_threads=None):
        config = config or {}
        cap = config.get('compile_threads', 1)
        if compile_threads is not None:
            if 'compile_threads' in config and compile_threads > cap:
                raise ValueError('CLI compile threads exceed setup budget')
            cap = compile_threads
        return cls(compile_threads=cap, **{key: config[key] for key in
            ('cpu_workers','memory_budget_bytes','cache_budget_bytes') if key in config})

    def admit(self, path, task=None):
        inventory = resource_inventory(path)
        ceiling = inventory['cpu_worker_ceiling']
        if ceiling is None or self.compile_threads * self.cpu_workers > ceiling:
            raise ValueError('worker × compile thread budget exceeds effective affinity/cgroup CPU capacity')
        if task is not None and json.loads(task['execution'])['compile_threads'] > self.compile_threads:
            raise ValueError('queued task compile threads exceed current budget; do not alter the task')
        memory = inventory['effective_memory_bytes']
        if self.memory_budget_bytes and (memory is None or self.memory_budget_bytes > memory):
            raise ValueError('memory budget exceeds known effective memory capacity')
        free = inventory['disk_free_bytes']
        if free is None or free < self.disk_reserve_bytes:
            raise ValueError('insufficient disk reserve before worker launch')
        if self.cache_budget_bytes and self.cache_budget_bytes > free - self.disk_reserve_bytes:
            raise ValueError('scratch budget exceeds available disk after reserve')


def group_rss(group, proc=Path('/proc')):
    """Linux RSS sum for one process group; vanished processes are ignored."""
    total = 0
    for entry in proc.iterdir():
        if not entry.name.isdecimal():
            continue
        try:
            # comm may contain spaces/parentheses; fields after its last ')' begin at state.
            fields = (entry/'stat').read_text().rsplit(')', 1)[1].split()
            if int(fields[2]) == group:
                pages = int(fields[21])  # stat field 24 RSS, relative to field 3 state
                total += max(0, pages) * os.sysconf('SC_PAGE_SIZE')
        except (FileNotFoundError, ProcessLookupError):
            continue
        except (OSError, ValueError, IndexError) as exc:
            raise ValueError('process RSS sampling unavailable') from exc
    return total


def scratch_size(root):
    total = count = 0
    def failed(error):
        raise error
    for directory, dirs, files in os.walk(root, followlinks=False, onerror=failed):
        for name in dirs + files:
            count += 1
            if count > 10000:
                raise ValueError('scratch entry bound exceeded')
            path = Path(directory)/name
            try:
                info = path.lstat()
                # Count directory metadata and symlinks without following them.
                total += max(info.st_size, getattr(info, 'st_blocks', 0)*512)
            except FileNotFoundError:
                continue
    return total


def violation(budget, group, scratch, result_parent):
    if budget.memory_budget_bytes and group_rss(group) > budget.memory_budget_bytes:
        return 'memory_budget_exceeded'
    if budget.cache_budget_bytes and scratch_size(scratch) > budget.cache_budget_bytes:
        return 'scratch_budget_exceeded'
    if min(shutil.disk_usage(scratch).free, shutil.disk_usage(result_parent).free) < budget.disk_reserve_bytes:
        return 'disk_reserve_exhausted'
    return None


def policy_path(ledger):
    return ledger.path.with_name(ledger.path.name+'.resources.json')


def read_policy(ledger):
    path = policy_path(ledger)
    return Budget(**json.loads(path.read_text())) if path.exists() else None


def write_policy(ledger, budget):
    """Caller owns queue lock. Reuse atomic sidecar helper without secret config."""
    import tempfile
    fd, name = tempfile.mkstemp(prefix='.fuzz-resource-', dir=ledger.path.parent)
    try:
        with os.fdopen(fd, 'w') as out:
            json.dump(asdict(budget), out)
            out.flush(); os.fsync(out.fileno())
        os.replace(name, policy_path(ledger))
    finally:
        Path(name).unlink(missing_ok=True)

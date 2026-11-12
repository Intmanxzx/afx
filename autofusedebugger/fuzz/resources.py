"""Read-only, stdlib resource inventory. Budgets never change case generation.

Unknown limits stay unknown. These are advisory limits, not reservations or a
scheduler; managed run consumes the inventory through scheduling.budgets.
"""
import math
import os
from pathlib import Path
import shutil


def _read(path):
    try:
        return Path(path).read_text().strip()
    except (OSError, UnicodeError):
        return None


def _positive(value):
    try:
        number = int(value)
        return number if number > 0 else None
    except (TypeError, ValueError):
        return None


def _unescape(value):
    for escaped, char in ((r"\040", " "), (r"\011", "\t"), (r"\134", "\\")):
        value = value.replace(escaped, char)
    return value


def cgroup_directories(membership, mountinfo):
    """Resolve actual controller paths, including nested/namespace mounts.

    Return child-to-mount-root ancestors: parents may impose tighter limits.
    Ignore paths that cannot be safely resolved inside the observed mount.
    """
    groups = []
    for line in membership.splitlines():
        fields = line.split(":", 2)
        if len(fields) == 3:
            groups.append((set(fields[1].split(",")) if fields[1] else set(), fields[2]))
    result = {"cpu": [], "memory": []}
    for line in mountinfo.splitlines():
        parts = line.split(" - ", 1)
        if len(parts) != 2:
            continue
        left, right = parts[0].split(), parts[1].split()
        if len(left) < 5 or len(right) < 3 or right[0] not in {"cgroup", "cgroup2"}:
            continue
        version = 2 if right[0] == "cgroup2" else 1
        root, mount = Path(_unescape(left[3])), Path(_unescape(left[4]))
        controllers = set(right[2].split(","))
        for names, group in groups:
            if (version == 2) != (not names):
                continue
            try:
                relative = Path(group).relative_to(root)
            except ValueError:
                continue
            if ".." in relative.parts:
                continue
            directory = mount / relative
            for controller in result:
                if version == 2 or controller in names & controllers:
                    current = directory
                    while True:
                        entry = (version, current)
                        if entry not in result[controller]:
                            result[controller].append(entry)
                        if current == mount:
                            break
                        current = current.parent
    return result


def cgroup_limits(membership, mountinfo, read=_read):
    limits = {"cpu_quota_cores": [], "memory_limit_bytes": []}
    directories = cgroup_directories(membership, mountinfo)
    for version, directory in directories["cpu"]:
        if version == 2:
            fields = (read(directory / "cpu.max") or "").split()
            quota, period = (fields if len(fields) == 2 else (None, None))
        else:
            quota, period = read(directory / "cpu.cfs_quota_us"), read(directory / "cpu.cfs_period_us")
        quota, period = _positive(quota), _positive(period)
        if quota and period:
            limits["cpu_quota_cores"].append(quota / period)
    for version, directory in directories["memory"]:
        value = _positive(read(directory / ("memory.max" if version == 2 else "memory.limit_in_bytes")))
        # v1 unlimited sentinel is near LONG_MAX, not an allocatable limit.
        if value and (version == 2 or value < 2**60):
            limits["memory_limit_bytes"].append(value)
    return {key: min(values) if values else None for key, values in limits.items()}


def resource_inventory(home):
    host = os.cpu_count()
    try:
        affinity = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        affinity = None
    limits = cgroup_limits(_read("/proc/self/cgroup") or "", _read("/proc/self/mountinfo") or "")
    candidates = [v for v in (host, affinity, limits["cpu_quota_cores"]) if v]
    cores = min(candidates) if candidates else None
    try:
        physical = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
        physical = physical if physical > 0 else None
    except (AttributeError, OSError, ValueError):
        physical = None
    memories = [v for v in (physical, limits["memory_limit_bytes"]) if v]
    path = Path(home)
    while not path.exists() and path != path.parent:
        path = path.parent
    try:
        disk = shutil.disk_usage(path).free
    except OSError:
        disk = None
    return {"host_cpu_count": host, "affinity_cpu_count": affinity, **limits,
            "effective_cpu_cores": cores, "cpu_worker_ceiling": max(1, math.floor(cores)) if cores else None,
            "physical_memory_bytes": physical, "effective_memory_bytes": min(memories) if memories else None,
            "disk_free_bytes": disk, "disk_probe_path": str(path),
            "enforcement": "inventory_only; managed CPU run enforces admission and sampled budgets"}


def budget_warnings(config, inventory):
    warnings = []
    product = config.get("cpu_workers", 1) * config.get("compile_threads", 4)
    cores = inventory["effective_cpu_cores"]
    if cores and product > cores:
        warnings.append(f"workers × compile_threads ({product}) exceeds effective CPU cores ({cores:g})")
    for setting, available in (("memory_budget_bytes", "effective_memory_bytes"), ("cache_budget_bytes", "disk_free_bytes")):
        if config.get(setting) and inventory.get(available) and config[setting] > inventory[available]:
            warnings.append(f"{setting} exceeds detected {available}")
    return warnings


def adaptive_defaults(home):
    """Conservative admission heuristic, not a claim of benchmark-optimal tuning.

    Workers share a CPU/compile budget; keep a conservative bound on concurrent imports.
    Respect effective cgroup/affinity limits and leave host/disk headroom.
    """
    inventory=resource_inventory(home)
    cores=inventory.get('cpu_worker_ceiling') or 1
    memory=inventory.get('effective_memory_bytes')
    disk=inventory.get('disk_free_bytes')
    workers=max(1,min(4,cores//2 or 1,(memory//(2*1024**3)) if memory else 1))
    result={'cpu_workers':workers, 'compile_threads':max(1,min(4,cores//workers))}
    if memory and os.name=='posix' and Path('/proc').exists():
        result['memory_budget_bytes']=max(1,int(memory*.5))
    if disk:result['cache_budget_bytes']=max(1,min(20*1024**3,int(disk*.2)))
    return result


def npu_resource_plan(home, devices, budget):
    """Explainable startup heuristic, not an online optimizer or hard isolation.

    Preserve device runtimes and CPU pipeline slots before adding compilation
    parallelism. Six is a conservative starting ceiling: our dual-910 probes
    found no throughput benefit from nine, not a universal saturation claim.
    """
    if type(devices) is not int or devices < 0:
        raise ValueError('device count must be a nonnegative integer')
    inventory = resource_inventory(home)
    cores = inventory.get('cpu_worker_ceiling') or 1
    available_memory = inventory.get('effective_memory_bytes')
    # A setup made on a larger machine must not carry an impossible RAM/disk
    # budget into a new container. Keep half of detected RAM as headroom.
    gib = 1024**3
    configured_memory = budget.memory_budget_bytes
    memory = max(1, available_memory//2) if available_memory else None
    if memory is not None and configured_memory is not None:
        memory = min(memory, configured_memory)
    # A saved cap is not evidence of current capacity. Unknown RAM permits
    # only one worker/thread; keep it unknown in diagnostics and admission.
    planning_memory = memory if memory is not None else min(configured_memory or 2*gib, 2*gib)
    # CPU-only callers need no compiler subprocess pool. For NPUs leave at
    # least two shared CPU slots when resources permit, not two workers/card.
    target = devices + max(2, math.ceil(devices/2)) if devices else min(4, cores)
    workers = max(1, min(cores, planning_memory//(2*gib), target))
    # Conservatively charge every slot, including potential compiler children.
    # RSS estimates are a startup guard, never a guarantee for arbitrary graphs.
    memory_threads = max(1, planning_memory//(workers*2*gib)-1)
    threads = max(1, min(6 if devices else 1, cores//workers, memory_threads))
    reasons = ['preserve NPU runtimes and shared CPU pipeline slots',
               'cap compile fan-out at a conservative six; do not fill cores with idle workers',
               'bound workers × compile_threads by effective CPU and estimated RSS budget']
    if not inventory.get('cpu_worker_ceiling') or available_memory is None:
        reasons.append('unknown resources: conservative fallback; admission still validates limits')
    if workers < target:
        reasons.append('CPU/RAM constrained: some devices or pipeline stages must run serially')
    budget_limits = {}
    if available_memory and os.name == 'posix' and Path('/proc').exists():
        budget_limits['memory_budget_bytes'] = memory
    disk = inventory.get('disk_free_bytes')
    if disk:
        cache = min(20*gib, max(1, disk//5))
        budget_limits['cache_budget_bytes'] = min(getattr(budget, 'cache_budget_bytes', None) or cache, cache)
    effective_cores = inventory.get('effective_cpu_cores')
    if effective_cores is not None and effective_cores < 1:
        reasons.append('sub-core CPU quota: one single-thread process remains subject to cgroup throttling')
    if threads < (6 if devices else 1):
        reasons.append('compile fan-out reduced by CPU capacity or the conservative RAM estimate')
    corrections = {key: {'configured': getattr(budget, key, None), 'selected': value}
                   for key, value in budget_limits.items() if getattr(budget, key, None) != value}
    return {'version': 2, 'mode': 'auto', 'inventory': inventory,
            'budget_corrections': corrections,
            'budget_limits': budget_limits,
            'devices': devices, 'memory_budget_bytes': memory,
            'target_workers': target,
            'selected': {'cpu_workers': workers, 'compile_threads': threads},
            'reasons': reasons,
            'limitations': 'startup heuristic; no runtime retuning, no independent compile/device pipeline'}


def npu_first_defaults(home, devices, budget):
    return npu_resource_plan(home, devices, budget)['selected']


def use_auto_resources(config, requested, workers, threads):
    """Explicit overrides win; legacy configured budgets remain manual."""
    explicit = workers is not None or threads is not None
    if requested is True and explicit:
        raise ValueError('--auto-resources cannot be combined with explicit worker/thread overrides')
    if requested is not None:
        return requested
    return not explicit and (not config or config.get('resource_policy') == 'auto')

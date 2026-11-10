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

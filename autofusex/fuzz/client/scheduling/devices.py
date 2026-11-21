"""Discover logical NPU indices in an isolated process; honor visibility mapping."""
import json
import re
import subprocess
import sys
import time


_inventory_cache = None
_inventory_at = None
_inventory_model = None


def device_inventory_info():
    """Visible logical NPUs; None means detection failed, [] means zero."""
    global _inventory_cache, _inventory_at, _inventory_model
    now = time.monotonic()
    if _inventory_at is not None and now - _inventory_at < 60:
        return {"devices": _inventory_cache, "model": _inventory_model}
    result = None
    try:
        probe = subprocess.run([sys.executable, '-c',
            'import json,torch,torch_npu; count=torch.npu.device_count(); '
            'models=[] if count == 0 else [str(torch.npu.get_device_properties(i).name) for i in range(count)]; '
            'print(json.dumps({"count":count,"models":models}))'],
            capture_output=True, text=True, timeout=30)
        if probe.returncode == 0:
            for line in reversed(probe.stdout.splitlines()):
                try:
                    payload = json.loads(line)
                    count = payload['count']
                    if type(count) is int and 0 <= count <= 64:
                        result = ['npu:' + str(i) for i in range(count)]
                        models = payload.get('models')
                        if isinstance(models, list):
                            _inventory_model = next((str(model) for model in models if model), None)
                        break
                except (ValueError, KeyError, TypeError):
                    continue
    except (OSError, subprocess.TimeoutExpired):
        pass
    _inventory_cache, _inventory_at = result, now
    return {"devices": result, "model": _inventory_model}


def device_inventory():
    """Visible logical NPUs; None means detection failed, [] means zero."""
    return device_inventory_info()["devices"]


def select_devices(requested=None):
    if requested:
        devices=list(dict.fromkeys(requested))
        if any(not re.fullmatch(r'npu:[0-9]+',d) for d in devices):
            raise ValueError('--device requires logical npu:index')
        return devices
    probe=subprocess.run([sys.executable,'-c',
        'import json,torch,torch_npu;print(json.dumps({"count":torch.npu.device_count()}))'],
        capture_output=True,text=True,timeout=30)
    if probe.returncode:
        raise ValueError('NPU discovery failed; check torch_npu/CANN environment or select --device explicitly')
    for line in reversed(probe.stdout.splitlines()):
        try: count=json.loads(line)['count'];break
        except (ValueError,KeyError,TypeError):continue
    else: raise ValueError('NPU discovery returned no device count')
    if type(count) is not int or not 1<=count<=64:raise ValueError('no available NPU devices')
    return ['npu:'+str(i) for i in range(count)]


def select_replay_device(requested=None):
    """Pick the least occupied visible NPU for an ad-hoc replay."""
    if requested:
        return requested
    probe = subprocess.run([sys.executable, '-c',
        'import json,torch,torch_npu; '
        'print(json.dumps([{"device":"npu:"+str(i),"free":int(torch.npu.mem_get_info(i)[0])} '
        'for i in range(torch.npu.device_count())]))'],
        capture_output=True, text=True, timeout=30)
    if probe.returncode:
        raise ValueError('NPU discovery failed; check torch_npu/CANN environment or select --device explicitly')
    for line in reversed(probe.stdout.splitlines()):
        try:
            rows = json.loads(line)
            if isinstance(rows, list) and rows:
                return max(rows, key=lambda row: int(row['free']))['device']
        except (ValueError, KeyError, TypeError):
            continue
    raise ValueError('NPU discovery returned no device inventory')

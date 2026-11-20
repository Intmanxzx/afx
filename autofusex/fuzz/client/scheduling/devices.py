"""Discover logical NPU indices in an isolated process; honor visibility mapping."""
import json
import re
import subprocess
import sys
import time


_inventory_cache = None
_inventory_at = None


def device_inventory():
    """Visible logical NPUs; None means detection failed, [] means zero."""
    global _inventory_cache, _inventory_at
    now = time.monotonic()
    if _inventory_at is not None and now - _inventory_at < 60:
        return _inventory_cache
    result = None
    try:
        probe = subprocess.run([sys.executable, '-c',
            'import json,torch,torch_npu;print(json.dumps({"count":torch.npu.device_count()}))'],
            capture_output=True, text=True, timeout=30)
        if probe.returncode == 0:
            for line in reversed(probe.stdout.splitlines()):
                try:
                    count = json.loads(line)['count']
                    if type(count) is int and 0 <= count <= 64:
                        result = ['npu:' + str(i) for i in range(count)]
                        break
                except (ValueError, KeyError, TypeError):
                    continue
    except (OSError, subprocess.TimeoutExpired):
        pass
    _inventory_cache, _inventory_at = result, now
    return result


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

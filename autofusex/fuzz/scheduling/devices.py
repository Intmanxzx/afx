"""Discover logical NPU indices in an isolated process; honor visibility mapping."""
import json
import re
import subprocess
import sys


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

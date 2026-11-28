"""Compiler environment identity for reusable coverage observations."""
import hashlib
import json


FIELDS = ('torch', 'torch_npu', 'torch_npu_build', 'cann', 'autofuse_binary',
          'compiler_flags', 'machine', 'python')


def environment_key(environment):
    if not isinstance(environment, dict):
        return None
    if not all(environment.get(k) for k in ('torch', 'torch_npu', 'torch_npu_build', 'cann', 'autofuse_binary')):
        return None
    value = {k: environment.get(k) for k in FIELDS}
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

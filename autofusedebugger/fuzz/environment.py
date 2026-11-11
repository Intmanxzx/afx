"""Read-only execution provenance. Never read upload credentials."""
import hashlib
import importlib.metadata
import importlib.util
import os
from pathlib import Path
import platform
import sys


def collect(home=None):
    from .config import load_config
    config = load_config(home) or {}
    identity = config.get('identity', {})
    result = {'python': platform.python_version(), 'platform': sys.platform,
              'machine': platform.machine(), 'device_name': identity.get('display_name'),
              'user_id': identity.get('user_id'), 'installation_id': config.get('source_id')}
    for package, key in [('torch','torch'),('torch-npu','torch_npu'),('triton-ascend','triton'),('autofusedebugger','package_version')]:
        try: result[key] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: result[key] = None
    # CANN runtime accessor, only when already imported for execution.
    try:
        if 'torch_npu' in sys.modules:
            from torch_npu.npu.utils import get_cann_version
            result['cann'] = get_cann_version()
    except (RuntimeError, ImportError, AttributeError):
        result['cann'] = None
    # CPU peers record the installed CANN version without importing torch_npu.
    # Runtime accessor above remains authoritative when available.
    if not result.get('cann'):
        root = os.environ.get('ASCEND_HOME_PATH')
        if root:
            try:
                lines = (Path(root)/'compiler/version.info').read_text().splitlines()
                result['cann'] = next((line.split('=',1)[1] for line in lines if line.startswith('Version=')), None)
            except OSError:
                pass
    # Fingerprint the actually selected binary, NOT an unrelated source checkout.
    try:
        spec = importlib.util.find_spec('autofuse')
        binary = Path(spec.origin).parent/'pyautofuse.so' if spec and spec.origin else None
        if binary and binary.is_file():
            digest = hashlib.sha256()
            with binary.open('rb') as stream:
                for block in iter(lambda: stream.read(1024*1024), b''): digest.update(block)
            result['autofuse_binary'] = digest.hexdigest()
            manifest = binary.with_name('build-info.json')
            if manifest.is_file():
                import json
                metadata = json.loads(manifest.read_text())
                if metadata.get('binary_sha256') == result['autofuse_binary']:
                    result['autofuse_commit'] = metadata.get('commit')
    except (ImportError, ValueError, OSError):
        pass
    return result

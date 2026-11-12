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
    # All backends use the selected toolkit's version.info. Import state must
    # not change provenance (the runtime uses a different version spelling).
    result['cann'] = None
    root = os.environ.get('ASCEND_HOME_PATH')
    if root:
        try:
            lines = (Path(root)/'compiler/version.info').read_text().splitlines()
            for line in lines:
                key, separator, value = line.partition('=')
                if separator and key.strip() == 'Version' and value.strip():
                    result['cann'] = value.strip()
                    break
        except (OSError, UnicodeError):
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

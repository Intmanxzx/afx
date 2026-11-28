"""Read-only execution provenance. Never read upload credentials."""
import hashlib
import importlib.metadata
import importlib.util
import os
from pathlib import Path
import platform
import sys
from functools import lru_cache


@lru_cache(maxsize=512)
def _file_digest(path, size, mtime_ns, ctime_ns):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.digest()


def collect(home=None):
    from autofusex.client.paths import load_config
    # Managed queues live under home/runs/<id>, not directly under home.
    # Their descendants inherit the installation home from the task launcher.
    task_home = os.environ.get('AFX_TASK_HOME')
    config = load_config(task_home or home)
    if task_home and config is None:
        raise ValueError('managed task installation configuration is missing: ' + task_home)
    config = config or {}
    identity = config.get('identity', {})
    result = {'python': platform.python_version(), 'platform': sys.platform,
              'machine': platform.machine(), 'device_name': identity.get('display_name'),
              'user_id': identity.get('user_id'), 'installation_id': config.get('source_id')}
    for package, key in [('torch','torch'),('torch-npu','torch_npu'),('triton-ascend','triton'),('autofusex','package_version')]:
        try: result[key] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: result[key] = None
    # Prefer the graph-autofusion run package's version.info, derived from the
    # actually imported autofuse package. Fall back to the selected toolkit
    # version only when the run-package manifest is unavailable.
    result['cann'] = None
    version_files = []
    try:
        spec = importlib.util.find_spec('autofuse')
        if spec and spec.origin:
            package_root = Path(spec.origin).resolve().parent
            version_files.extend(parent/'share/info/graph_autofusion/version.info'
                                 for parent in package_root.parents)
    except (ImportError, ValueError, OSError):
        pass
    root = os.environ.get('ASCEND_HOME_PATH')
    if root:
        root = Path(root)
        version_files.extend((root/'share/info/graph_autofusion/version.info',
                              root/'compiler/version.info'))
    for version_file in version_files:
        try:
            lines = version_file.read_text().splitlines()
            for line in lines:
                key, separator, value = line.partition('=')
                if separator and key.strip() == 'Version' and value.strip():
                    result['cann'] = value.strip()
                    break
            if result['cann']:
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
    # Include local lowering patches: a wheel version alone is not a build ID.
    digest = hashlib.sha256()
    try:
        spec = importlib.util.find_spec('torch_npu')
        package = Path(spec.origin).parent if spec and spec.origin else None
        files = sorted((package / '_inductor').rglob('*.py')) if package else []
        files += sorted((package / 'lib').glob('*.so')) if package else []
        for path in files:
            digest.update(str(path.relative_to(package)).encode())
            stat = path.stat()
            digest.update(_file_digest(str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns))
        if files:
            result['torch_npu_build'] = digest.hexdigest()
    except (ImportError, ValueError, OSError):
        pass
    result['compiler_flags'] = {k: os.environ.get(k, '') for k in
        ('AUTOFUSE_FLAGS', 'AUTOFUSE_DFX_FLAGS', 'TORCHINDUCTOR_ASCENDC_DEBUG', 'TORCHINDUCTOR_NPU_BACKEND')}
    return result

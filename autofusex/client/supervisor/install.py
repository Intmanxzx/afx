"""Install an immutable supervisor source bundle, independent of client upgrades."""
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

FILES = (
    '__init__.py', 'contracts/__init__.py', 'contracts/ipc.py', 'contracts/serialization.py',
    'client/store.py', 'client/artifacts.py', 'client/activation.py', 'client/updates.py',
    'client/supervisor/__init__.py',
)


def ensure(home):
    home = Path(home)
    root = home / 'supervisor'
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (root / 'install.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        bundle = root / 'v1'
        if not (bundle / 'ready.json').exists():
            source = Path(__file__).resolve().parents[2]
            stage = Path(tempfile.mkdtemp(prefix='.install-', dir=root))
            try:
                package = stage / 'autofusex'
                for name in FILES:
                    target = package / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source / name, target)
                # The frozen package must never import the upgradeable client facade.
                (package / 'client/__init__.py').write_text('')
                (stage / 'ready.json').write_text(json.dumps({'protocol': 2, 'python': sys.executable}))
                os.rename(stage, bundle)
            finally:
                if stage.exists():
                    shutil.rmtree(stage)
    manifest = json.loads((bundle / 'ready.json').read_text())
    return [manifest['python'], '-I', '-c',
            'import sys; sys.path.insert(0, sys.argv[1]); from autofusex.client.supervisor import run; raise SystemExit(run(sys.argv[2]))',
            str(bundle), str(home)]

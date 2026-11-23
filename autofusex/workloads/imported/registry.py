"""Immutable local source packages. Import registers bytes; it never executes them."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile

from autofusex.contracts.serialization import fingerprint

MAX_PACKAGE_BYTES = 256 * 1024 * 1024


def register(home, source, *, script=None, entry=None, name=None, origin='manual', workload='imported'):
    source = Path(source).expanduser()
    if source.is_symlink():
        raise ValueError('source package must not contain symlinks')
    source = source.resolve(strict=True)
    if workload not in {'imported', 'model'} or origin not in {'manual', 'baize_torch', 'baize_tf', 'model'}:
        raise ValueError('unsupported case origin or workload')
    if entry is not None and not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', entry):
        raise ValueError('entry must be a Python function name')
    root = source if source.is_dir() else source.parent
    script = script or (source.name if source.is_file() else 'case.py')
    script_path = (root / script).resolve()
    if not script_path.is_relative_to(root) or not script_path.is_file() or script_path.suffix != '.py':
        raise ValueError('script must be a Python file inside the source package')
    candidates = sorted(source.rglob('*')) if source.is_dir() else [source]
    files = {}; total = 0
    for path in candidates:
        if any(part in {'__pycache__', '.git', '.venv'} for part in path.relative_to(root).parts):
            continue
        if path.is_symlink():
            raise ValueError('source package must not contain symlinks')
        if not path.is_file():
            continue
        total += path.stat().st_size
        if total > MAX_PACKAGE_BYTES or len(files) >= 10000:
            raise ValueError('source package exceeds import limits')
        files[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    spec = dict(schema_version=1, workload=workload, origin=origin, script=script_path.relative_to(root).as_posix(), entry=entry, files=files)
    if spec['script'] not in files:
        raise ValueError('entry script must be included in the imported source package')
    revision = fingerprint(spec)
    key = ('M:' if workload == 'model' else 'I:') + revision
    record = dict(case_key=key, revision=revision, name=name or source.stem, spec=spec)
    registry = Path(home) / 'cases'; registry.mkdir(parents=True, exist_ok=True, mode=0o700)
    target = registry / revision
    if target.exists():
        return inspect(home, key)
    stage = Path(tempfile.mkdtemp(prefix='.import-', dir=registry))
    try:
        for relative, digest in files.items():
            destination = stage / 'source' / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / relative, destination)
            if hashlib.sha256(destination.read_bytes()).hexdigest() != digest:
                raise ValueError('source changed during import')
        (stage / 'case.json').write_text(json.dumps(record, ensure_ascii=True))
        try:
            stage.rename(target)
        except FileExistsError:
            return inspect(home, key)
    finally:
        if stage.exists(): shutil.rmtree(stage)
    return record


def inspect(home, key, revision=None):
    if not isinstance(key, str) or not re.fullmatch(r'[IM]:[0-9a-f]{64}', key):
        raise ValueError('invalid case key')
    digest = key[2:]
    if revision is not None and revision != digest:
        raise ValueError('case revision mismatch')
    root = Path(home) / 'cases' / digest
    record = json.loads((root / 'case.json').read_text())
    if record['case_key'] != key or record['revision'] != digest or fingerprint(record['spec']) != digest:
        raise ValueError('case identity mismatch')
    for relative, expected in record['spec']['files'].items():
        path = root / 'source' / relative
        if not path.resolve().is_relative_to((root / 'source').resolve()) or path.is_symlink():
            raise ValueError('case source escaped registry')
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('case source checksum mismatch')
    return record


def list_cases(home):
    root = Path(home) / 'cases'
    return [json.loads(path.read_text()) for path in sorted(root.glob('*/case.json')) if not path.parent.name.startswith('.')]


def script_path(home, record):
    inspect(home, record['case_key'], record['revision'])
    return Path(home) / 'cases' / record['revision'] / 'source' / record['spec']['script']


def export(home, key):
    import base64
    record = inspect(home,key)
    root = Path(home)/'cases'/record['revision']/'source'
    files = {name:base64.b64encode((root/name).read_bytes()).decode('ascii') for name in record['spec']['files']}
    from autofusex.contracts.source_cases import validate
    validate(record,files)
    return dict(case=record,files=files)


def receive(home, bundle):
    from autofusex.contracts.source_cases import validate
    record=bundle['case'];decoded=validate(record,bundle['files'])
    registry=Path(home)/'cases'; registry.mkdir(parents=True,exist_ok=True,mode=0o700)
    target=registry/record['revision']
    if target.exists():return inspect(home,record['case_key'])
    stage=Path(tempfile.mkdtemp(prefix='.receive-',dir=registry))
    try:
        for name,raw in decoded.items():
            path=stage/'source'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
        (stage/'case.json').write_text(json.dumps(record))
        try:stage.rename(target)
        except FileExistsError:return inspect(home,record['case_key'])
    finally:
        if stage.exists():shutil.rmtree(stage)
    return record

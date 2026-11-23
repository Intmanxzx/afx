"""Portable fixed-source case bundles, validated without importing their code."""
import base64
import hashlib
import re
from pathlib import PurePosixPath
from autofusex.contracts.serialization import fingerprint, canonical_json

MAX_BYTES = 512 * 1024


def validate(record, files):
    if not isinstance(record, dict) or not isinstance(files, dict): raise ValueError('case and files required')
    spec = record.get('spec', {})
    if spec.get('workload') not in {'imported','model'}: raise ValueError('invalid workload')
    digest = fingerprint(spec)
    prefix = 'M:' if spec['workload']=='model' else 'I:'
    if record.get('revision') != digest or record.get('case_key') != prefix+digest:
        raise ValueError('case identity mismatch')
    if not isinstance(spec.get('files'),dict) or not spec['files'] or set(files)!=set(spec['files']):
        raise ValueError('case files mismatch')
    entry=spec.get('entry')
    if entry is not None and (not isinstance(entry,str) or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*',entry)):
        raise ValueError('invalid factory entry')
    if spec.get('script') not in files or not spec['script'].endswith('.py'): raise ValueError('invalid script')
    total=0; decoded={}
    for name, value in files.items():
        path=PurePosixPath(name)
        if path.is_absolute() or '..' in path.parts or path.as_posix()!=name or '\\' in name:
            raise ValueError('invalid source path')
        if not isinstance(value,str): raise ValueError('base64 source required')
        try: raw=base64.b64decode(value,validate=True)
        except (ValueError, TypeError) as exc: raise ValueError('invalid source encoding') from exc
        total+=len(raw)
        if total>MAX_BYTES: raise ValueError('portable source exceeds 512 KiB')
        if hashlib.sha256(raw).hexdigest()!=spec['files'][name]: raise ValueError('source checksum mismatch')
        decoded[name]=raw
    if len(canonical_json(record).encode())>MAX_BYTES: raise ValueError('case metadata too large')
    return decoded

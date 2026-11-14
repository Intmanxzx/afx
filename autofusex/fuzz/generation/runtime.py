"""Pinned dependency gate shared by generation and execution."""
import hashlib
from pathlib import Path
SOURCE_SHA256 = 'c47e5611db39361963be90786c7ea208647df4b6eb9546ce911122b81721cba5'
Z3_VERSION = '4.12.2'
def source_evidence():
    import nnsmith
    import z3
    root = Path(nnsmith.__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob('*.py')):
        digest.update(path.relative_to(root).as_posix().encode() + b'\0' + path.read_bytes())
    result = {'generator': 'nnsmith.graph_gen.model_gen', 'source_sha256': digest.hexdigest(),
              'z3': z3.get_version_string(), 'source_path': str(root)}
    if result['source_sha256'] != SOURCE_SHA256 or result['z3'] != Z3_VERSION:
        raise ValueError(
            f'NNSmith dependency mismatch at {root}: source_sha256={result["source_sha256"]} '
            f'(expected {SOURCE_SHA256}), z3={result["z3"]} (expected {Z3_VERSION}). '
            'Install the pinned AutofuseX NNSmith fork and remove stale PYTHONPATH entries.')
    return result

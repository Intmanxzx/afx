"""Canonical bundle identity; independent of the graph generator implementation."""
from dataclasses import asdict
import hashlib
from autofusex.workloads.fuzz.contracts import fingerprint


def content_manifest(bundle):
    spec = asdict(bundle.spec)
    spec.pop('seed')
    return {'schema': 'nnsmith-workload/3', 'spec': spec,
            'storages': [{'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
                         for raw in bundle.storages]}


def workload_id(bundle):
    return fingerprint(content_manifest(bundle))

"""Version dispatch; generators never depend on execution, storage or hardware."""
from ..contracts import Seed


def _protocol(seed):
    seed = Seed.parse(seed)
    if seed.version == 1:
        from . import v1
        return v1
    raise ValueError(f"unsupported generation version {seed.version}")


def plan(seed):
    return _protocol(seed).plan(seed)


def materialize(seed):
    return _protocol(seed).materialize(seed)


def content_manifest(bundle):
    return _protocol(bundle.spec.seed).content_manifest(bundle)


def workload_id(bundle):
    return _protocol(bundle.spec.seed).workload_id(bundle)


def oracle_id(bundle):
    return _protocol(bundle.spec.seed).ORACLE_ID


def reference(bundle, sample_index):
    return _protocol(bundle.spec.seed).reference(bundle, sample_index)

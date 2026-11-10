"""Single NNSmith generation entry for run/replay. No template fallback."""
from ..contracts import Seed
from .fingerprint import content_manifest, workload_id


def _protocol(seed):
    seed = Seed.parse(seed)
    if seed.version != 3:
        raise ValueError('retired template seed: only NNSmith C seeds can be generated; no fallback')
    from . import nnsmith_v3
    return nnsmith_v3


def plan(seed):
    return _protocol(seed).plan(seed)


def materialize(seed):
    return _protocol(seed).materialize(seed)


def oracle_id(bundle):
    return _protocol(bundle.spec.seed).ORACLE_ID


def reference(bundle, sample_index):
    return _protocol(bundle.spec.seed).reference(bundle, sample_index)

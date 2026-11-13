"""Single NNSmith generation entry for run/replay. No template fallback."""
from ..contracts import Seed
from .fingerprint import content_manifest, workload_id


def _protocol(seed):
    seed = Seed.parse(seed)
    if seed.version in {5, 6, 7, 8, 9, 10}:
        from . import nnsmith_sampled
        return nnsmith_sampled
    if seed.version == 4:
        from . import nnsmith_dynamic
        return nnsmith_dynamic
    if seed.version != 3:
        raise ValueError('retired template seed: only NNSmith C seeds can be generated; no fallback')
    from . import nnsmith_v3
    return nnsmith_v3


def plan(seed):
    return _protocol(seed).plan(Seed.parse(seed))


def materialize(seed):
    return _protocol(seed).materialize(Seed.parse(seed))


def oracle_id(bundle):
    protocol = _protocol(bundle.spec.seed)
    return protocol.oracle_id(bundle) if hasattr(protocol, 'oracle_id') else protocol.ORACLE_ID


def reference(bundle, sample_index):
    return _protocol(bundle.spec.seed).reference(bundle, sample_index)


def materialize_spec(spec):
    """Reconstruct bytes with the captured seed protocol, never C defaults."""
    protocol = _protocol(spec.seed)
    if spec.seed.version == 4:
        from . import nnsmith_v3
        protocol = nnsmith_v3
    return protocol.materialize_spec(spec)

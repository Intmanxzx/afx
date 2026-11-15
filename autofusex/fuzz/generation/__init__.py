"""One active NNSmith generator; historical seeds are never reinterpreted."""
from ..contracts import Seed
from .fingerprint import content_manifest, workload_id


def plan(seed):
    from .nnsmith_sampled import plan
    return plan(Seed.parse(seed).require_current())


def materialize(seed):
    from .nnsmith_sampled import materialize
    return materialize(Seed.parse(seed).require_current())


def materialize_spec(spec):
    spec.seed.require_current()
    from .nnsmith_sampled import materialize_spec
    return materialize_spec(spec)


def oracle_id(bundle):
    return 'nnsmith-aten-cpu/2' if bundle.spec.seed.current else 'retired-internal-test'


def reference(bundle, sample_index):
    raise ValueError('CPU eager is the sole oracle')

"""One NNSmith generator with a validated E/F seed contract."""
from ..contracts import Seed
from .fingerprint import content_manifest, workload_id


def plan(seed):
    from .nnsmith_sampled import plan
    return plan(Seed.parse(seed))


def materialize(seed):
    from .nnsmith_sampled import materialize
    return materialize(Seed.parse(seed))


def materialize_spec(spec):
    from .nnsmith_sampled import materialize_spec
    return materialize_spec(spec)


def oracle_id(bundle):
    return 'nnsmith-aten-cpu/2'


def reference(bundle, sample_index):
    raise ValueError('CPU eager is the sole oracle')

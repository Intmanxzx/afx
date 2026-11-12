"""Frozen E/F sampling configuration, packed into a self-contained 128-bit seed.

Do not change field order, profiles or distribution formulas for this protocol.
No environment/profile file participates in replay.
"""
from dataclasses import dataclass, asdict
from ..contracts import Seed

DIM_MODES = ('uniform', 'half_normal', 'log_normal')
# Keep the first three values stable: they are already encoded in v5/v6 seeds.
# `mixed` is the historical fp16/fp32 compatibility mode; `all` restores the
# pre-refactor behavior (NNSmith's complete generation dtype set intersected
# with the selected operator signatures).
DTYPE_MODES = ('float32', 'float16', 'mixed', 'all')
VALUE_MODES = ('uniform', 'log_scale', 'llm', 'recommendation')
# Historical GenerationConfig profiles, copied rather than importing torch/nnsmith.
PROFILES = {
    'small': (1, 64, 2**16, 10, 'uniform'),
    'medium': (8, 4096, 2**20, 30, 'uniform'),
    'large': (8, 8192, 2**24, 60, 'uniform'),
    '16k': (8, 16384, 65536, 90, 'log_normal'),
    '32k': (8, 32768, 131072, 120, 'log_normal'),
    '64k': (8, 65536, 262144, 180, 'log_normal'),
    '128k': (8, 131072, 524288, 300, 'log_normal'),
    'default': (1, 131072, 524288, 300, 'half_normal'),
}
FIELDS = (('random_seed', 32), ('dim_min', 24), ('dim_max', 24),
          ('max_elem_per_tensor', 25), ('dim_mode', 2), ('value_mode', 2),
          ('max_nodes', 8), ('timeout_seconds', 9), ('dtype_mode', 2))

@dataclass(frozen=True)
class SamplingConfig:
    dim_min: int = 1
    dim_max: int = 131072
    max_elem_per_tensor: int = 524288
    dim_dist_mode: str = 'half_normal'
    value_dist_mode: str = 'uniform'
    dtype_mode: str = 'all'
    max_nodes: int = 5
    timeout_seconds: int = 300

    def __post_init__(self):
        for name, bits in FIELDS:
            if name in {'random_seed', 'dim_mode', 'value_mode', 'dtype_mode'}:
                continue
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value < 2**bits:
                raise ValueError(f'{name} must be in 1..{2**bits-1}')
        if self.dim_min > self.dim_max or self.dim_min > self.max_elem_per_tensor:
            raise ValueError('invalid dimension range/element budget')
        if self.dtype_mode not in DTYPE_MODES:
            raise ValueError('unknown dtype mode')
        if self.dim_dist_mode not in DIM_MODES or self.value_dist_mode not in VALUE_MODES:
            raise ValueError('unknown sampling distribution')

    @classmethod
    def profile(cls, name='default', **overrides):
        lo, hi, budget, timeout, mode = PROFILES[name]
        values = dict(dim_min=lo, dim_max=hi, max_elem_per_tensor=budget,
                      timeout_seconds=timeout, dim_dist_mode=mode)
        values.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**values)

    def seed(self, random_seed, dynamic=False):
        if type(random_seed) is not int or not 0 <= random_seed < 2**32:
            raise ValueError('random_seed must be uint32')
        values = dict(asdict(self), random_seed=random_seed,
                      dtype_mode=DTYPE_MODES.index(self.dtype_mode),
                      dim_mode=DIM_MODES.index(self.dim_dist_mode),
                      value_mode=VALUE_MODES.index(self.value_dist_mode))
        entropy = offset = 0
        for name, bits in FIELDS:
            entropy |= values[name] << offset
            offset += bits
        return Seed((5 if dynamic else 4)*2**128 + entropy)


def decode(seed):
    seed = Seed.parse(seed)
    if seed.version not in {5, 6}:
        raise ValueError('E/F seed required')
    remaining = seed.entropy
    values = {}
    for name, bits in FIELDS:
        values[name] = remaining & (2**bits-1)
        remaining >>= bits
    if remaining or values['dim_mode'] >= len(DIM_MODES) or values['dtype_mode'] >= len(DTYPE_MODES):
        raise ValueError('reserved sampling seed bits')
    values['dtype_mode'] = DTYPE_MODES[values['dtype_mode']]
    random_seed = values.pop('random_seed')
    values['dim_dist_mode'] = DIM_MODES[values.pop('dim_mode')]
    values['value_dist_mode'] = VALUE_MODES[values.pop('value_mode')]
    return SamplingConfig(**values), random_seed

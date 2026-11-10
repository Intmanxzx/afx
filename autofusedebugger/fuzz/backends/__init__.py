"""Public backend taxonomy. Historical identifiers remain readable, not relabeled."""
BACKEND_NAMES = ('cpu', 'npu-eager', 'triton-npu', 'af-legacy', 'af-mlir')
HISTORICAL_NAMES = ('cpu-eager', 'cpu-compile', 'ascendc', 'triton')
NPU_BACKENDS = {'npu-eager', 'triton-npu', 'af-legacy', 'af-mlir', 'ascendc', 'triton'}


from ..backend_names import display_backend


def create_backend(config):
    if config.backend not in BACKEND_NAMES + HISTORICAL_NAMES:
        from .base import UnsupportedBackend
        raise UnsupportedBackend(f'unknown backend: {config.backend}')
    from .torch import TorchAdapter
    return TorchAdapter(config)

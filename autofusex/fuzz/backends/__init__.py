"""Public backend taxonomy."""
BACKEND_NAMES = ('cpu', 'cpu-compile', 'eager-npu', 'triton-npu', 'af-legacy', 'af-mlir')
NPU_BACKENDS = {'eager-npu', 'triton-npu', 'af-legacy', 'af-mlir'}


from ..backend_names import display_backend


def create_backend(config):
    if config.backend not in BACKEND_NAMES:
        from .base import UnsupportedBackend
        raise UnsupportedBackend(f'unknown backend: {config.backend}')
    from .torch import TorchAdapter
    return TorchAdapter(config)

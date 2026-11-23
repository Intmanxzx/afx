"""Public backend taxonomy."""
BACKEND_NAMES = ('cpu', 'cpu-compile', 'eager-npu', 'triton-npu', 'af-legacy', 'af-mlir')
NPU_BACKENDS = {'eager-npu', 'triton-npu', 'af-legacy', 'af-mlir'}


from autofusex.execution.backend_names import display_backend


def create_backend(config):
    if config.backend not in BACKEND_NAMES:
        from autofusex.execution.backends.base import UnsupportedBackend
        raise UnsupportedBackend(f'unknown backend: {config.backend}')
    from autofusex.execution.backends.torch import TorchAdapter
    return TorchAdapter(config)

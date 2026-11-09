"""Lazy backend factory; importing this registry never imports Torch."""
BACKEND_NAMES = ("cpu-eager", "cpu-compile", "ascendc", "triton")


def create_backend(config):
    if config.backend not in BACKEND_NAMES:
        from .base import UnsupportedBackend
        raise UnsupportedBackend(f"unknown backend: {config.backend}")
    from .torch import TorchAdapter
    return TorchAdapter(config)

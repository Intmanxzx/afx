from __future__ import annotations

from autofusedebugger.adapters.base import BackendAdapter, CapturedInvocation, FrameworkAdapter
from autofusedebugger.adapters.detection import detect_framework, find_cann_ge_root
from autofusedebugger.adapters.jax import JaxAdapter
from autofusedebugger.adapters.inductor_autofuse import InductorAutofuseBackend
from autofusedebugger.adapters.pytorch import PyTorchAdapter
from autofusedebugger.adapters.tensorflow import TensorFlowAdapter


_FRAMEWORK_ADAPTERS: dict[str, type[FrameworkAdapter]] = {
    "pytorch": PyTorchAdapter,
    "tensorflow": TensorFlowAdapter,
    "jax": JaxAdapter,
}

_BACKEND_ADAPTERS: dict[str, type[BackendAdapter]] = {
    "inductor_autofuse": InductorAutofuseBackend,
}


def framework_names() -> tuple[str, ...]:
    return tuple(_FRAMEWORK_ADAPTERS)


def get_framework_adapter(name: str) -> FrameworkAdapter:
    resolved = "pytorch" if name == "auto" else name
    adapter_type = _FRAMEWORK_ADAPTERS.get(resolved)
    if adapter_type is None:
        raise ValueError(f"unsupported framework: {name}")
    return adapter_type()


def backend_names() -> tuple[str, ...]:
    return tuple(_BACKEND_ADAPTERS)


def get_backend_adapter(name: str) -> BackendAdapter:
    adapter_type = _BACKEND_ADAPTERS.get(name)
    if adapter_type is None:
        raise ValueError(f"unsupported backend: {name}")
    return adapter_type()


__all__ = [
    "BackendAdapter",
    "CapturedInvocation",
    "FrameworkAdapter",
    "JaxAdapter",
    "InductorAutofuseBackend",
    "PyTorchAdapter",
    "TensorFlowAdapter",
    "backend_names",
    "detect_framework",
    "find_cann_ge_root",
    "framework_names",
    "get_backend_adapter",
    "get_framework_adapter",
]

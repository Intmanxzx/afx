"""Adapter boundary. Only execution-specific operations cross this interface."""
from typing import Protocol


class UnsupportedBackend(RuntimeError):
    pass


class Backend(Protocol):
    name: str

    def check(self, bundle, evaluation): ...
    def prepare(self, bundle): ...
    def inputs(self, bundle, index): ...
    def invoke(self, prepared, inputs): ...
    def synchronize(self): ...
    def compare(self, output, expected): ...

# Optional, capability-proven hooks (absence means unknown, never zero):
# compile_duration_ns() -> int | None: isolated completed compile work only.
# measure_kernel(prepared, inputs, evaluation) -> int: device profiler mean ns;
# adapter must hold an exclusive device lease and synchronize profiler collection.
# TorchAdapter deliberately implements neither until backend evidence is available.

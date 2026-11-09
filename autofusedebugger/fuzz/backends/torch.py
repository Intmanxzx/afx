"""Torch execution adapter; heavy imports occur only after capability checks."""
from .base import UnsupportedBackend


class TorchAdapter:
    """Lazy execution dependency. No NNSmith patch needed for protocol v1."""
    def __init__(self, config):
        self.config = config
        self.name = config.backend

    def check(self, bundle, evaluation):
        from ..generation import plan
        if bundle.spec != plan(bundle.spec.seed):
            raise UnsupportedBackend("adapter accepts only the frozen minimal profile, not arbitrary graphs")
        if self.name not in {"cpu-eager", "cpu-compile", "ascendc", "triton"}:
            raise UnsupportedBackend("unknown backend")
        npu = self.name in {"ascendc", "triton"}
        device_type = self.config.device.split(":", 1)[0]
        if device_type != ("npu" if npu else "cpu"):
            raise UnsupportedBackend("backend/device mismatch")
        if npu and evaluation.performance:
            raise UnsupportedBackend("strict NPU measurement requires a device lease; not implemented yet")
        import os
        os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = str(self.config.compile_threads)
        import torch
        self.torch = torch
        if npu:
            try:
                import torch_npu  # noqa: F401
                if not torch.npu.is_available():
                    raise UnsupportedBackend("NPU unavailable")
            except ImportError as exc:
                raise UnsupportedBackend("torch_npu unavailable") from exc
        self.device = torch.device(self.config.device)
        if npu:
            from .device_lease import DeviceLease
            identity = str(torch.npu.get_device_properties(self.device).uuid)
            self.lease = DeviceLease(identity).acquire()

    def close(self):
        if hasattr(self, 'lease'):
            self.lease.close()

    def _tensor(self, bundle, spec):
        from ..tensors import storage_values
        torch = self.torch
        storage = torch.tensor(storage_values(bundle, spec), dtype=getattr(torch, spec.dtype))
        # Transfer flat storage, then construct view; .to() on a view may compact it.
        storage = storage.to(self.device)
        tensor = storage.as_strided(spec.shape, spec.stride, spec.storage_offset)
        if tuple(tensor.shape) != spec.shape or tuple(tensor.stride()) != spec.stride:
            raise ValueError("input signature changed during device binding")
        return tensor

    def prepare(self, bundle):
        import json
        torch = self.torch
        graph = json.loads(bundle.spec.graph_json)
        weight = self._tensor(bundle, bundle.spec.weights[0])
        op = graph["nodes"][1]["op"]
        axis = graph["reduce_axis"]
        def forward(x):
            value = x + weight
            value = torch.relu(value) if op == "relu" else -value if op == "neg" else value * weight
            if graph.get('transpose'):
                value = value.transpose(0, 1)
            if axis is not None:
                value = value.sum(dim=axis)
            return (value,)
        if self.name == "cpu-eager":
            return forward
        options = {"npu_backend": self.name} if self.name in {"ascendc", "triton"} else {}
        return torch.compile(forward, backend="inductor", fullgraph=True,
                             dynamic=not self.config.static_only and len(bundle.spec.shape_samples) > 1, options=options)

    def inputs(self, bundle, index):
        return tuple(self._tensor(bundle, spec) for spec in bundle.spec.shape_samples[index])

    def invoke(self, prepared, inputs):
        with self.torch.no_grad():
            return prepared(*inputs)

    def synchronize(self):
        if self.device.type == "npu":
            self.torch.npu.synchronize(self.device)

    def compare(self, output, expected):
        import struct
        from ..tensors import DTYPES
        if not isinstance(output, tuple) or len(output) != len(expected):
            return False
        for actual, (spec, raw) in zip(output, expected):
            if not isinstance(actual, self.torch.Tensor) or tuple(actual.shape) != spec.shape or str(actual.dtype) != "torch." + spec.dtype:
                return False
            fmt, width = DTYPES[spec.dtype]
            values = struct.unpack("<" + fmt * (len(raw) // width), raw)
            reference_tensor = self.torch.tensor(values, dtype=getattr(self.torch, spec.dtype)).reshape(spec.shape)
            if not self.torch.equal(actual.detach().cpu(), reference_tensor):
                return False
        return True

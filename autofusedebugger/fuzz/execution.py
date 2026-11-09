"""One sample loop for static/dynamic correctness and optional measurement.

Backend adapters own only capability checks, preparation, invocation and sync.
No artifact pickling or claim that a lazy callable is already compiled.
"""
import time
from dataclasses import asdict

from .protocol_v1 import ORACLE_ID, reference, workload_id


class UnsupportedBackend(RuntimeError):
    pass


def execute(bundle, adapter, evaluation):
    result = {"seed": str(bundle.spec.seed), "workload_id": workload_id(bundle),
              "oracle_id": ORACLE_ID, "backend": adapter.name, "shape_results": [],
              "compile_mode": "combined", "preview": True, "evaluation": asdict(evaluation)}
    if hasattr(adapter, "config"):
        result["execution"] = asdict(adapter.config)
    try:
        adapter.check(bundle, evaluation)
        started = time.perf_counter_ns()
        prepared = adapter.prepare(bundle)
        result["prepare_ns"] = time.perf_counter_ns() - started
        result["prepare_is_compile_complete"] = False
    except Exception as exc:
        result.update(status="unsupported" if isinstance(exc, UnsupportedBackend) else "error",
                      error_stage="prepare", error_summary=f"{type(exc).__name__}: {exc}"[:500])
        return result
    static_only = getattr(getattr(adapter, "config", None), "static_only", False)
    indices = range(1 if static_only else len(bundle.spec.shape_samples))
    result["sample_selection"] = "base_only" if static_only else "all"
    for index in indices:
        row = {"shape_index": index, "correctness": "not_checked", "performance": "not_requested"}
        result["shape_results"].append(row)
        stage = "input_binding"
        try:
            inputs = adapter.inputs(bundle, index)
            stage = "first_call_combined"
            started = time.perf_counter_ns()
            output = adapter.invoke(prepared, inputs)
            adapter.synchronize()
            row["first_call_combined_ns"] = time.perf_counter_ns() - started
            if evaluation.correctness:
                stage = "correctness"
                row["correctness"] = "pass" if adapter.compare(output, reference(bundle, index)) else "fail"
            if evaluation.performance:
                stage = "measurement"
                # Reset inputs outside timing even though the frozen minimal profile is functional.
                for _ in range(evaluation.warmup_steps):
                    adapter.invoke(prepared, adapter.inputs(bundle, index))
                    adapter.synchronize()
                durations = []
                for _ in range(evaluation.measured_steps):
                    inputs = adapter.inputs(bundle, index)
                    adapter.synchronize()
                    started = time.perf_counter_ns()
                    adapter.invoke(prepared, inputs)
                    adapter.synchronize()
                    durations.append(time.perf_counter_ns() - started)
                row.update(performance="measured", mean_wall_ns=sum(durations) // len(durations),
                           measured_steps=len(durations), metric="synchronized_wall_time_not_kernel_profiler")
        except Exception as exc:
            row.update(error_stage=stage, error_summary=f"{type(exc).__name__}: {exc}"[:500])
            if stage == "measurement":
                row["performance"] = "error"  # Do not erase correctness.
            else:
                row["correctness"] = "error"
                if evaluation.performance:
                    row["performance"] = "skipped"
    result["status"] = "error" if any("error_stage" in r for r in result["shape_results"]) else (
        "fail" if any(r["correctness"] == "fail" for r in result["shape_results"]) else "ok")
    return result


class TorchAdapter:
    """Lazy execution dependency. No NNSmith patch needed for protocol v1."""
    def __init__(self, config):
        self.config = config
        self.name = config.backend

    def check(self, bundle, evaluation):
        from .protocol_v1 import plan
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

    def _tensor(self, bundle, spec):
        from .protocol_v1 import storage_values
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
        from .protocol_v1 import DTYPES
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

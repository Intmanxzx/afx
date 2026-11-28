"""Torch execution adapter; heavy imports occur only after capability checks."""
from autofusex.execution.backends.base import UnsupportedBackend


def _ensure_backend_dfx_flag(flags: str, mode: str) -> str:
    """Add the backend route without replacing an explicit caller choice."""
    parts = [part.strip() for part in (flags or '').split(';') if part.strip()]
    if not any(part.lstrip('-').split('=', 1)[0] == 'mlir_codegen_config' for part in parts):
        parts.append('--mlir_codegen_config=' + mode)
    return ';'.join(parts)


class TorchAdapter:
    """Lazy execution dependency for NNSmith-generated immutable graphs."""
    from autofusex.execution.backends.base import Capabilities
    capabilities = Capabilities()

    def __init__(self, config):
        self.config = config
        self.name = config.backend
        # Workers are keyed by backend and device, so one backend's compiler state never reaches another;
        # the stage worker resets dynamo/device caches after every session before offering reuse.
        self.capabilities = self.Capabilities(persistent_runtime=True)

    def check(self, bundle, evaluation):
        from autofusex.workloads.fuzz.generation import plan
        if not getattr(self, 'validated_shared_case', False) and bundle.spec != plan(bundle.spec.seed):
            raise UnsupportedBackend("bundle does not match NNSmith seed regeneration")
        from autofusex.execution.backends import BACKEND_NAMES
        from autofusex.execution.backends import NPU_BACKENDS
        if self.name not in BACKEND_NAMES:
            raise UnsupportedBackend("unknown backend")
        npu = self.name in NPU_BACKENDS
        device_type = self.config.device.split(":", 1)[0]
        if device_type != ("npu" if npu else "cpu"):
            raise UnsupportedBackend("backend/device mismatch")
        if npu and evaluation.performance:
            raise UnsupportedBackend("strict NPU measurement requires a device lease; not implemented yet")
        import os
        if self.name in {'af-legacy', 'af-mlir'}:
            flags = os.environ.get('AUTOFUSE_DFX_FLAGS', '')
            self.old_dfx = os.environ.get('AUTOFUSE_DFX_FLAGS')
            mode = 'mlir' if self.name == 'af-mlir' else 'none'
            os.environ['AUTOFUSE_DFX_FLAGS'] = _ensure_backend_dfx_flag(flags, mode)
        if self.name == 'eager-npu':
            self.old_autofuse_flags = os.environ.get('AUTOFUSE_FLAGS')
            os.environ['AUTOFUSE_FLAGS'] = '--enable_autofuse=false'
        os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = str(self.config.compile_threads)
        import torch
        self.torch = torch
        if self.name in {'af-legacy', 'af-mlir'}:
            from autofusex.execution.backends.mlir_capability import verify_compile_adapter
            verify_compile_adapter()
        if self.name == 'af-mlir':
            from autofusex.execution.backends.mlir_capability import verify_runtime
            verify_runtime()
        if npu:
            try:
                import torch_npu  # noqa: F401
                if not torch.npu.is_available():
                    raise UnsupportedBackend("NPU unavailable")
            except ImportError as exc:
                raise UnsupportedBackend("torch_npu unavailable") from exc
        self.device = torch.device(self.config.device)
        if npu:
            from autofusex.execution.backends.device_lease import DeviceLease
            identity = str(torch.npu.get_device_properties(self.device).uuid)
            self.lease = DeviceLease(identity).acquire(timeout=getattr(self,'device_lease_wait',0))
            # Inductor/runtime helpers also consult the current device; explicit
            # tensor .to(npu:1) alone does not bind generated launches to card 1.
            self.old_device = torch.npu.current_device()
            torch.npu.set_device(self.device)
            # Per-session peak: a reused runtime must not inherit the prior case.
            torch.npu.reset_peak_memory_stats(self.device)

    def close(self):
        if hasattr(self, 'old_device'):
            self.torch.npu.set_device(self.old_device)
        if hasattr(self, 'old_dfx'):
            import os
            if self.old_dfx is None:
                os.environ.pop('AUTOFUSE_DFX_FLAGS', None)
            else:
                os.environ['AUTOFUSE_DFX_FLAGS'] = self.old_dfx
        if hasattr(self, 'old_autofuse_flags'):
            import os
            if self.old_autofuse_flags is None:
                os.environ.pop('AUTOFUSE_FLAGS', None)
            else:
                os.environ['AUTOFUSE_FLAGS'] = self.old_autofuse_flags
        if hasattr(self, 'lease'):
            self.lease.close()

    def _tensor(self, bundle, spec):
        from autofusex.workloads.fuzz.tensors import storage_values
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
        if graph.get('dialect') != 'nnsmith-aten/2':
            raise UnsupportedBackend('retired graph dialect; replay unsupported')
        from autofusex.workloads.fuzz.nnsmith.aten_dialect import validate_runtime
        validate_runtime()
        from autofusex.workloads.fuzz.generation.torch_dialect import execute_graph
        return self._compile(execute_graph(torch, graph,
            {t.name: self._tensor(bundle, t) for t in bundle.spec.weights},
            [t.name for t in bundle.spec.inputs], self.device), bundle)

    def _compile(self, forward, bundle):
        torch = self.torch
        if self.name == "cpu":
            return forward
        if self.name == "eager-npu":
            return torch.compile(forward, backend="eager")
        target = {'af-legacy': 'ascendc', 'af-mlir': 'ascendc', 'triton-npu': 'triton'}.get(self.name, self.name)
        options = {"npu_backend": target} if target in {"ascendc", "triton"} else {}
        return torch.compile(forward, backend="inductor", fullgraph=True,
                             dynamic=not self.config.static_only and len(bundle.spec.shape_samples) > 1, options=options)

    def inputs(self, bundle, index):
        # New full-range graphs must preserve NNSmith's declared outputs.
        # If a dynamic variant cannot be faithfully materialized, the shared
        # CPU reference rejects it rather than comparing a different program.
        self.expected_outputs = bundle.spec.output_samples[index]
        tensors=tuple(self._tensor(bundle, spec) for spec in bundle.spec.shape_samples[index])
        if not self.config.static_only and len(bundle.spec.shape_samples)>1 and self.name not in {'cpu','eager-npu'}:
            for position,tensor in enumerate(tensors):
                for axis in range(tensor.ndim):
                    if len({sample[position].shape[axis] for sample in bundle.spec.shape_samples})>1:
                        self.torch._dynamo.mark_dynamic(tensor,axis)
        return tensors

    def invoke(self, prepared, inputs):
        from autofusex.execution.backends.compile_overlap import overlap_compile
        with overlap_compile(self), self.torch.no_grad():
            output = prepared(*inputs)
        if getattr(self, 'expected_outputs', None) is not None:
            if (not isinstance(output, tuple) or len(output) != len(self.expected_outputs)
                    or any(tuple(actual.shape) != expected.shape or str(actual.dtype) != 'torch.' + expected.dtype
                           for actual, expected in zip(output, self.expected_outputs))):
                raise ValueError('materialized Torch graph differs from declared NNSmith output signature')
        return output

    def synchronize(self):
        if self.device.type == "npu":
            self.torch.npu.synchronize(self.device)

    def compare(self, output, expected):
        from autofusex.workloads.fuzz.tensors import decode_storage
        if not isinstance(output, tuple) or len(output) != len(expected):
            return False
        if expected and isinstance(expected[0], self.torch.Tensor):
            return self.outputs_equal(tuple(value.detach().cpu() for value in output), expected)
        for actual, (spec, raw) in zip(output, expected):
            if not isinstance(actual, self.torch.Tensor) or tuple(actual.shape) != spec.shape or str(actual.dtype) != "torch." + spec.dtype:
                return False
            values = decode_storage(raw, spec.dtype)
            reference_tensor = self.torch.tensor(values, dtype=getattr(self.torch, spec.dtype)).reshape(spec.shape)
            if not bool(self._close(actual, reference_tensor).all()):
                return False
        return True

    def cpu_reference(self, bundle, index):
        """Run the actual eager CPU implementation, independent of NPU output."""
        original_name, original_device = self.name, self.device
        try:
            self.name, self.device = 'cpu', self.torch.device('cpu')
            return self.invoke(self.prepare(bundle), self.inputs(bundle, index))
        finally:
            self.name, self.device = original_name, original_device

    def _close(self, actual, reference):
        from autofusex.execution.compare import resolve_tolerances
        actual, reference = actual.detach().cpu(), reference.detach().cpu()
        if not actual.is_floating_point() and not actual.is_complex():
            return actual == reference
        rtol, atol, _ = resolve_tolerances(reference.dtype, actual.dtype)
        # Match execution.compare: aligned NaNs and same-sign infinities pass.
        # Complex outputs must align each component, including the finite one.
        # Evaluate tolerance arithmetic without rounding it to the output dtype.
        if actual.is_complex():
            # Compare real and imaginary components with the same dtype tolerance.
            # torch.isclose's complex magnitude semantics otherwise diverge from
            # the component-wise diagnostics produced by accuracy.summarize.
            real = self.torch.isclose(actual.real.to(self.torch.float64), reference.real.to(self.torch.float64),
                                      rtol=rtol, atol=atol, equal_nan=True)
            imag = self.torch.isclose(actual.imag.to(self.torch.float64), reference.imag.to(self.torch.float64),
                                      rtol=rtol, atol=atol, equal_nan=True)
            return real & imag
        return self.torch.isclose(actual.to(self.torch.float64), reference.to(self.torch.float64),
                                  rtol=rtol, atol=atol, equal_nan=True)

    def outputs_equal(self, left, right):
        return isinstance(left,tuple) and isinstance(right,tuple) and len(left)==len(right) and all(
            a.shape==b.shape and a.dtype==b.dtype and bool(self._close(a,b).all()) for a,b in zip(left,right))

    def diagnose_accuracy(self, output, expected):
        from autofusex.workloads.fuzz.tensors import decode_storage
        from autofusex.execution.accuracy import summarize
        references = []
        for reference_value in expected:
            if isinstance(reference_value, self.torch.Tensor):
                references.append(reference_value.detach().cpu())
            else:
                spec, raw = reference_value
                references.append(self.torch.tensor(decode_storage(raw, spec.dtype),
                    dtype=getattr(self.torch, spec.dtype)).reshape(spec.shape))
        return summarize(output, tuple(references), self._close)

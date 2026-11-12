"""Torch execution adapter; heavy imports occur only after capability checks."""
from .base import UnsupportedBackend


class TorchAdapter:
    """Lazy execution dependency for NNSmith-generated immutable graphs."""
    from .base import Capabilities
    capabilities = Capabilities()

    def __init__(self, config):
        self.config = config
        self.name = config.backend
        self.capabilities = self.Capabilities(persistent_runtime=self.name in {'cpu','cpu-eager','eager-npu','npu-eager'})

    def check(self, bundle, evaluation):
        from ..generation import plan
        if not getattr(self, 'validated_shared_case', False) and bundle.spec != plan(bundle.spec.seed):
            raise UnsupportedBackend("bundle does not match NNSmith seed regeneration")
        from . import BACKEND_NAMES, HISTORICAL_NAMES, NPU_BACKENDS
        if self.name not in BACKEND_NAMES + HISTORICAL_NAMES:
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
            parts = [p for p in flags.split(';') if p and not p.strip().startswith('--mlir_codegen_config=')]
            self.old_dfx = os.environ.get('AUTOFUSE_DFX_FLAGS')
            mode = 'mlir' if self.name == 'af-mlir' else 'none'
            os.environ['AUTOFUSE_DFX_FLAGS'] = ';'.join(parts + ['--mlir_codegen_config=' + mode])
        os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = str(self.config.compile_threads)
        import torch
        self.torch = torch
        if self.name == 'af-mlir':
            from .mlir_capability import verify_runtime
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
            from .device_lease import DeviceLease
            identity = str(torch.npu.get_device_properties(self.device).uuid)
            self.lease = DeviceLease(identity).acquire()
            # Inductor/runtime helpers also consult the current device; explicit
            # tensor .to(npu:1) alone does not bind generated launches to card 1.
            self.old_device = torch.npu.current_device()
            torch.npu.set_device(self.device)

    def close(self):
        if hasattr(self, 'old_device'):
            self.torch.npu.set_device(self.old_device)
        if hasattr(self, 'old_dfx'):
            import os
            if self.old_dfx is None:
                os.environ.pop('AUTOFUSE_DFX_FLAGS', None)
            else:
                os.environ['AUTOFUSE_DFX_FLAGS'] = self.old_dfx
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
        if graph.get('dialect') not in {'nnsmith-ert/3', 'nnsmith-sampled/5'}:
            raise UnsupportedBackend('only NNSmith graphs are supported')
        from ..generation.nnsmith_v3 import execute_graph
        forward = execute_graph(torch, graph,
            {t.name: self._tensor(bundle, t) for t in bundle.spec.weights},
            [t.name for t in bundle.spec.inputs])
        return self._compile(forward, bundle)

    def _compile(self, forward, bundle):
        torch = self.torch
        if self.name in {"cpu", "cpu-eager", "npu-eager", "eager-npu"}:
            return forward
        target = {'af-legacy': 'ascendc', 'af-mlir': 'ascendc', 'triton-npu': 'triton'}.get(self.name, self.name)
        options = {"npu_backend": target} if target in {"ascendc", "triton"} else {}
        return torch.compile(forward, backend="inductor", fullgraph=True,
                             dynamic=not self.config.static_only and len(bundle.spec.shape_samples) > 1, options=options)

    def inputs(self, bundle, index):
        tensors=tuple(self._tensor(bundle, spec) for spec in bundle.spec.shape_samples[index])
        if not self.config.static_only and len(bundle.spec.shape_samples)>1 and self.name not in {'cpu','cpu-eager','npu-eager','eager-npu'}:
            for position,tensor in enumerate(tensors):
                for axis in range(tensor.ndim):
                    if len({sample[position].shape[axis] for sample in bundle.spec.shape_samples})>1:
                        self.torch._dynamo.mark_dynamic(tensor,axis)
        return tensors

    def invoke(self, prepared, inputs):
        with self.torch.no_grad():
            return prepared(*inputs)

    def synchronize(self):
        if self.device.type == "npu":
            self.torch.npu.synchronize(self.device)

    def compare(self, output, expected):
        from ..tensors import decode_storage
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
        from autofusedebugger.compare import resolve_tolerances
        actual, reference = actual.detach().cpu(), reference.detach().cpu()
        if not actual.is_floating_point():
            return actual == reference
        rtol, atol, _ = resolve_tolerances(reference.dtype, actual.dtype)
        # Evaluate tolerance arithmetic without rounding it to the output dtype.
        return self.torch.isclose(actual.to(self.torch.float64), reference.to(self.torch.float64),
                                  rtol=rtol, atol=atol, equal_nan=False)

    def outputs_equal(self, left, right):
        return isinstance(left,tuple) and isinstance(right,tuple) and len(left)==len(right) and all(
            a.shape==b.shape and a.dtype==b.dtype and bool(self._close(a,b).all()) for a,b in zip(left,right))

    def diagnose_accuracy(self, output, expected):
        from ..tensors import decode_storage
        total=mismatches=nonfinite=0
        max_abs=max_rel=absolute_sum=0.0
        for actual, reference_value in zip(output,expected):
            if isinstance(reference_value, self.torch.Tensor):
                reference = reference_value.detach().cpu()
            else:
                spec, raw = reference_value
                values = decode_storage(raw, spec.dtype)
                reference=self.torch.tensor(values,dtype=getattr(self.torch, spec.dtype)).reshape(spec.shape)
            if actual.shape!=reference.shape:
                return {'error':'output shape mismatch','basis':'cpu'}
            mismatches += int((~self._close(actual, reference)).sum())
            actual=actual.detach().cpu().to(self.torch.float64)
            reference=reference.to(self.torch.float64)
            finite=self.torch.isfinite(actual)&self.torch.isfinite(reference)
            diff=(actual-reference).abs()
            total+=actual.numel();nonfinite+=int((~finite).sum())
            if bool(finite.any()):
                absolute=diff[finite];relative=absolute/reference[finite].abs().clamp_min(1e-30)
                max_abs=max(max_abs,float(absolute.max()));max_rel=max(max_rel,float(relative.max()));absolute_sum+=float(absolute.sum())
        return {'max_abs':max_abs,'mean_abs':absolute_sum/max(1,total-nonfinite),'max_rel':max_rel,
                'mismatches':mismatches,'elements':total,'nonfinite':nonfinite}

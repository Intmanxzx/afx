"""One backend in a fresh interpreter. Environment is selected by the parent."""
from dataclasses import asdict
import argparse
import json
import os
from pathlib import Path
import sys
import traceback

from .backends import BACKENDS
from .cases import case_fingerprint, restore_opset
from .profiling import profile_kernels
from .schema import BackendResult, ProfileConfig


def execute(request: dict) -> BackendResult:
    import torch
    import torch_npu  # noqa: F401 -- registers the NPU device/backend
    from nnsmith.difftest import assert_allclose
    from nnsmith.materialize.torch import TorchModelCPU, numpify
    from nnsmith.materialize.torch.symbolnet import FxTracing
    from ..generation import GenerationConfig, generate_case, _tolerance_for_ir

    spec = BACKENDS[request["backend"]]
    print(f"PERF_STAGE backend={spec.name} stage=generation", flush=True)
    result = BackendResult(spec.name)
    stage = "generation"
    try:
        config = GenerationConfig(**request["generation"])
        case = generate_case(opset=restore_opset(request["opset"], TorchModelCPU),
                             model_cls=TorchModelCPU, case_seed=request["seed"], config=config)
        result.fingerprint = case_fingerprint(case)
        if result.fingerprint != request["expected_fingerprint"]:
            raise ValueError("Regenerated graph/weights/inputs/oracle differ from the shared case")
        print(f"PERF_CASE backend={spec.name} fingerprint={result.fingerprint}", flush=True)
        stage = "backend_setup"
        device = torch.device(request["target"])
        if device.type != "npu":
            raise ValueError("Performance profiling requires an NPU target")
        torch.npu.set_device(device)
        # Validate the default selection when the installed version exposes it.
        import torch._inductor.config as inductor_config
        configured = getattr(inductor_config, "npu_backend", None)
        if spec.npu_backend is None and configured == "ascendc":
            raise RuntimeError("NPU default resolved to AscendC; refusing to label it Triton-Ascend")
        result.resolved_backend = str(configured or "NPU default (no override)") if spec.npu_backend is None else spec.npu_backend
        result.runtime = {
            "torch": torch.__version__, "torch_npu": torch_npu.__version__,
            "device": str(device), "device_name": torch.npu.get_device_name(device),
            "npu_backend_env": os.environ.get("TORCHINDUCTOR_NPU_BACKEND"),
            "compile_kwargs": spec.compile_kwargs(), "cc": os.environ.get("CC"),
        }
        model = case.model.native_model.eval().to(device)
        inputs = [torch.from_numpy(case.oracle.input[n]).to(device) for n in model.input_like]
        stage = "compile_or_warmup"
        print(f"PERF_STAGE backend={spec.name} stage={stage}", flush=True)
        profile = ProfileConfig(**request["profile"])
        with torch.no_grad():
            with FxTracing():
                traced = torch.fx.symbolic_trace(model)
            compiled = torch.compile(traced, **spec.compile_kwargs())
            output = compiled(*inputs)  # Complete lazy compilation before profiling.
            torch.npu.synchronize(device)
            actual = {n: numpify(t) for n, t in zip(case.oracle.output, output)}
            rtol, atol = _tolerance_for_ir(model.ir)
            try:
                assert_allclose(actual, case.oracle.output, spec.name, "CPU oracle",
                                equal_nan=True, rtol=rtol, atol=atol)
                result.correctness = "pass"
            except (AssertionError, KeyError, TypeError) as e:
                result.correctness = "fail"
                result.correctness_error = str(e)[-2000:]
            del actual, output
            for _ in range(profile.warmup_steps):
                compiled(*inputs)
            torch.npu.synchronize(device)
            stage = "profiling"
            print(f"PERF_STAGE backend={spec.name} stage={stage} correctness={result.correctness}", flush=True)
            stats = profile_kernels(compiled, inputs, device=device,
                                    output_dir=Path(request["profile_dir"]), config=profile)
        for key, value in stats.items():
            setattr(result, key, value)
        result.status = "ok" if result.total_kernel_us > 0 else "no_device_time"
    except Exception:
        result.error_stage = stage
        result.error = traceback.format_exc()[-4000:]
        print(traceback.format_exc(), file=sys.stderr, flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--result", required=True)
    args = parser.parse_args()
    request = json.loads(Path(args.request).read_text())
    result = execute(request)
    path = Path(args.result)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(asdict(result), indent=2, ensure_ascii=False))
    tmp.replace(path)
    return 0 if result.status in ("ok", "no_device_time") else 1


if __name__ == "__main__":
    raise SystemExit(main())

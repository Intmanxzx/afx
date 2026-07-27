from __future__ import annotations

import importlib.util
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from autofusedebugger.autofuse_graph import attach_autofuse_graphs, collect_autofuse_graphs, enable_autofuse_graph_dump
from autofusedebugger.compare import compare_three_manifests, save_report
from autofusedebugger.graph import build_manifest
from autofusedebugger.html import write_report_html
from autofusedebugger.io import tensor_to_numpy
from autofusedebugger.inductor_autofuse import (
    activate_autofuse_backend,
    autofuse_diagnostics,
    build_inductor_autofuse_graph,
    capture_inductor_outputs,
    install_autofuse_intermediate_hooks,
    reset_autofuse_diagnostics,
)
from autofusedebugger.npu_examples import (
    _cann_dump_sort_key,
    _candidate_cann_roots,
    _clean_dir,
    _desc,
    _first_tensor,
    _iter_fx_inputs,
    _npy_output_index,
    _npu_node_type,
    _prepend_pythonpath,
    _save_json,
    _target_type,
    op_name_from_npy,
)


@dataclass(frozen=True)
class TorchCase:
    model: Any
    inputs: tuple[Any, ...]
    name: str
    kwargs: dict[str, Any] = field(default_factory=dict)
    fullgraph: bool = True
    dynamic: bool = False


_NPU_CPU_PREFIXES = {
    "Abs": ("abs",),
    "Add": ("add",),
    "BatchMatMul": ("matmul",),
    "Cast": ("_to_copy", "to"),
    "Div": ("div",),
    "Exp": ("exp",),
    "Ge": ("ge",),
    "Greater": ("gt",),
    "GreaterEqual": ("ge",),
    "Less": ("lt",),
    "Log": ("log",),
    "MatMul": ("matmul", "linear"),
    "Mul": ("mul",),
    "Muls": ("mul",),
    "Neg": ("neg",),
    "Pow": ("pow",),
    "RealDiv": ("div",),
    "ReduceMean": ("mean",),
    "ReduceMeanD": ("mean",),
    "ReduceSum": ("sum",),
    "ReduceSumD": ("sum",),
    "Relu": ("relu",),
    "Rsqrt": ("rsqrt",),
    "Sigmoid": ("sigmoid",),
    "Softmax": ("softmax",),
    "SoftmaxV2": ("softmax",),
    "Sqrt": ("sqrt",),
    "Square": ("pow",),
    "Sub": ("sub",),
    "Tanh": ("tanh",),
}

_NPU_TOKEN_RE = re.compile(
    r"(BatchMatMul|GreaterEqual|ReduceMeanD|ReduceSumD|ReduceMean|ReduceSum|SoftmaxV2|RealDiv|"
    r"MatMul|Sigmoid|Softmax|Greater|Square|Rsqrt|Sqrt|Relu|Cast|Less|Tanh|Abs|Add|Sub|Mul|Muls|Div|"
    r"Pow|Exp|Log|Neg|Ge)(?:_?(\d+))?"
)

_UNSTABLE_MATCH_TYPES = {"Data", "Transpose", "StridedSliceD", "StridedSliceV2", "Reshape"}


def _load_script_module(script: str | Path) -> Any:
    path = Path(script).resolve()
    module_name = f"autofusedebugger_case_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _case_name(script: str | Path, raw: Any) -> str:
    if isinstance(raw, dict) and raw.get("name"):
        return str(raw["name"])
    return Path(script).stem


def load_torch_case(script: str | Path, entry: str = "create_case") -> TorchCase:
    module = _load_script_module(script)
    factory = getattr(module, entry, None)
    if factory is None or not callable(factory):
        raise AttributeError(f"{Path(script).resolve()} must define callable {entry}()")
    raw = factory()
    if isinstance(raw, dict):
        if "model" not in raw or "inputs" not in raw:
            raise ValueError(f"{entry}() dict must contain 'model' and 'inputs'")
        model = raw["model"]
        inputs = raw["inputs"]
        kwargs = raw.get("kwargs", {})
        name = _case_name(script, raw)
        fullgraph = bool(raw.get("fullgraph", True))
        dynamic = bool(raw.get("dynamic", False))
    elif isinstance(raw, tuple) and len(raw) == 2:
        model, inputs = raw
        kwargs = {}
        name = Path(script).stem
        fullgraph = True
        dynamic = False
    else:
        raise ValueError(f"{entry}() must return (model, inputs) or a dict")
    if not isinstance(inputs, (tuple, list)):
        inputs = (inputs,)
    if not isinstance(kwargs, dict):
        raise ValueError(f"{entry}() 'kwargs' must be a dict")
    return TorchCase(
        model=model,
        inputs=tuple(inputs),
        kwargs=kwargs,
        name=name,
        fullgraph=fullgraph,
        dynamic=dynamic,
    )


def _tree_map(value: Any, fn: Any) -> Any:
    if isinstance(value, tuple):
        return tuple(_tree_map(item, fn) for item in value)
    if isinstance(value, list):
        return [_tree_map(item, fn) for item in value]
    if isinstance(value, dict):
        return {key: _tree_map(item, fn) for key, item in value.items()}
    return fn(value)


def _to_cpu(value: Any) -> Any:
    def convert(item: Any) -> Any:
        try:
            import torch
        except Exception:
            return item
        if isinstance(item, torch.Tensor):
            return item.detach().cpu()
        return item

    return _tree_map(value, convert)


def _to_device(value: Any, device: str) -> Any:
    def convert(item: Any) -> Any:
        try:
            import torch
        except Exception:
            return item
        if isinstance(item, torch.Tensor):
            return item.to(device)
        return item

    return _tree_map(value, convert)


def _input_info(inputs: tuple[Any, ...]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    try:
        import torch
    except Exception:
        torch = None
    for idx, value in enumerate(inputs):
        if torch is not None and isinstance(value, torch.Tensor):
            _, dtype = tensor_to_numpy(value)
            result[f"input_{idx}"] = {
                "shape": tuple(int(dim) for dim in value.shape),
                "dtype": dtype,
            }
        else:
            result[f"input_{idx}"] = {"shape": (), "dtype": type(value).__name__}
    return result


def _flatten_user_inputs(inputs: tuple[Any, ...], kwargs: dict[str, Any]) -> list[Any]:
    from torch.utils import _pytree

    values, _ = _pytree.tree_flatten((inputs, kwargs))
    return values


def _export_args(exported: Any, inputs: tuple[Any, ...], kwargs: dict[str, Any]) -> list[Any]:
    args = []
    user_idx = 0
    example_inputs = getattr(exported, "example_inputs", (inputs, kwargs))
    if not isinstance(example_inputs, tuple) or len(example_inputs) != 2:
        example_inputs = (inputs, kwargs)
    user_inputs = _flatten_user_inputs(example_inputs[0], example_inputs[1])
    for spec in exported.graph_signature.input_specs:
        kind = spec.kind.name
        if kind in {"PARAMETER", "BUFFER"}:
            args.append(exported.state_dict[spec.target])
        elif kind == "USER_INPUT":
            args.append(user_inputs[user_idx])
            user_idx += 1
    return args


def export_graph_and_dump(
    case: TorchCase,
    dump_dir: Path,
    *,
    device: str = "cpu",
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    import torch

    model = case.model.to(device).eval()
    inputs = _to_device(case.inputs, device)
    kwargs = _to_device(case.kwargs, device)
    exported = torch.export.export(model, inputs, kwargs)
    if hasattr(exported, "run_decompositions"):
        exported = exported.run_decompositions()
    args = _export_args(exported, inputs, kwargs)
    saved: dict[str, dict[str, Any]] = {}

    class CaptureInterpreter(torch.fx.Interpreter):
        def run_node(self, node):  # type: ignore[no-untyped-def]
            result = super().run_node(node)
            if node.op != "call_function":
                return result
            tensor = _first_tensor(result, torch=torch)
            if tensor is None:
                return result
            array, dtype = tensor_to_numpy(tensor)
            path = dump_dir / f"{node.name}.output.0.npy"
            np.save(path, array)
            saved[node.name] = {
                "shape": tuple(int(dim) for dim in array.shape),
                "dtype": dtype,
                "size": int(array.size),
                "path": path,
            }
            return result

    with torch.no_grad():
        CaptureInterpreter(exported.graph_module).run(*args)

    user_names = [
        spec.arg.name
        for spec in exported.graph_signature.input_specs
        if spec.kind.name == "USER_INPUT" and hasattr(spec.arg, "name")
    ]
    input_meta = _input_info(tuple(_flatten_user_inputs(inputs, kwargs)))
    ops: list[dict[str, Any]] = []
    included: set[str] = set()
    for idx, name in enumerate(user_names):
        meta = input_meta.get(f"input_{idx}", {"dtype": "unknown", "shape": ()})
        ops.append({"name": name, "type": "Data", "output_desc": [_desc(meta["dtype"], meta["shape"])]})
        included.add(name)
    for node in exported.graph_module.graph.nodes:
        if node.op != "call_function" or node.name not in saved:
            continue
        info = saved[node.name]
        info["type"] = _target_type(node.target)
        inputs_for_node = []
        seen_inputs = set()
        for input_name in _iter_fx_inputs((node.args, node.kwargs)):
            if input_name not in included or input_name in seen_inputs:
                continue
            inputs_for_node.append(f"{input_name}:0")
            seen_inputs.add(input_name)
        ops.append(
            {
                "name": node.name,
                "type": _target_type(node.target),
                "input": inputs_for_node,
                "output_desc": [_desc(info["dtype"], info["shape"])],
            }
        )
        included.add(node.name)
    return {"graph": [{"op": ops}]}, saved


def export_cpu_graph_and_dump(case: TorchCase, cpu_dump: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    return export_graph_and_dump(case, cpu_dump, device="cpu")


def _cpu_name(prefix: str, suffix: str) -> str:
    return f"{prefix}_{suffix}" if suffix else prefix


def cpu_candidates_for_npu(npu_name: str) -> list[str]:
    npu_type = _npu_node_type(npu_name)
    local = npu_name.split(".", 1)[1] if "." in npu_name else npu_name
    candidates: list[str] = []
    seen: set[str] = set()

    def add(name: str) -> None:
        if name and name not in seen:
            candidates.append(name)
            seen.add(name)

    tokens = _NPU_TOKEN_RE.findall(local)
    if npu_type != "AutomaticBufferFusionOp":
        for kind, suffix in tokens:
            if kind != npu_type:
                continue
            for prefix in _NPU_CPU_PREFIXES.get(kind, (kind[:1].lower() + kind[1:],)):
                add(_cpu_name(prefix, suffix))
    for kind, suffix in reversed(tokens):
        for prefix in _NPU_CPU_PREFIXES.get(kind, (kind[:1].lower() + kind[1:],)):
            add(_cpu_name(prefix, suffix))
    add(local[:1].lower() + local[1:])
    return candidates


def match_npu_to_cpu(converted: list[Path], cpu_outputs: dict[str, dict[str, Any]]) -> dict[str, str]:
    matches: dict[str, str] = {}
    used_cpu: set[str] = set()

    def expand_candidates(candidates: list[str]) -> list[str]:
        result = []
        seen = set()
        for candidate in candidates:
            names = [candidate]
            if candidate not in cpu_outputs:
                names.extend(sorted(name for name in cpu_outputs if name.startswith(candidate + "_")))
            for name in names:
                if name in seen:
                    continue
                result.append(name)
                seen.add(name)
        return result

    for path in sorted(converted, key=_cann_dump_sort_key):
        output_index = _npy_output_index(path)
        if output_index != 0:
            continue
        npu_name = op_name_from_npy(path, output_index)
        if _npu_node_type(npu_name) in _UNSTABLE_MATCH_TYPES:
            continue
        try:
            actual = np.load(path, mmap_mode="r")
        except Exception:
            continue
        best: tuple[float, str] | None = None
        for candidate in expand_candidates(cpu_candidates_for_npu(npu_name)):
            info = cpu_outputs.get(candidate)
            if info is None or candidate in used_cpu or int(info["size"]) != int(actual.size):
                continue
            cpu_path = info.get("path")
            if cpu_path is None:
                score = 0.0
            else:
                try:
                    golden = np.load(cpu_path, mmap_mode="r")
                    if int(golden.size) != int(actual.size):
                        continue
                    score = float(np.max(np.abs(golden.astype(np.float64).reshape(-1) - actual.astype(np.float64).reshape(-1))))
                except Exception:
                    score = 0.0
            if best is None or score < best[0]:
                best = (score, candidate)
        if best is not None:
            matches[npu_name] = best[1]
            used_cpu.add(best[1])
    return matches


def run_npu_model(
    case: TorchCase,
    npu_dump_dir: Path,
    compile_dir: Path,
    autofuse_dump_dir: Path,
    *,
    cann_root: str | Path | None,
    device: str,
) -> tuple[Any, dict[str, dict[str, Any]], dict[str, Any]]:
    enable_autofuse_graph_dump(autofuse_dump_dir)
    import torch

    for candidate in _candidate_cann_roots(cann_root):
        site_packages = candidate / "python/site-packages"
        if site_packages.is_dir():
            _prepend_pythonpath(site_packages)
            break

    os.environ.pop("TORCHINDUCTOR_NPU_EXT_DEBUG", None)
    os.environ["TORCH_COMPILE_DEBUG"] = "1"
    os.environ["TORCHINDUCTOR_FORCE_DISABLE_CACHES"] = "1"
    torch.compiler.reset()
    torch._inductor.config.trace.enabled = True
    torch._inductor.config.trace.debug_dir = str(compile_dir)
    torch._inductor.config.generate_intermediate_hooks = True
    if hasattr(torch.compiler, "config"):
        torch.compiler.config.force_disable_caches = True

    import torch_npu  # noqa: F401
    from torch._inductor.hooks import intermediate_hook

    package_root = activate_autofuse_backend()
    install_autofuse_intermediate_hooks(package_root)
    reset_autofuse_diagnostics()
    torch.npu.set_device(device)
    model = case.model.eval().to(device)
    compile_options = {"npu_backend": "ascendc"} if package_root.startswith("torch_npu.") else None
    compiled = torch.compile(model, dynamic=case.dynamic, fullgraph=case.fullgraph, options=compile_options)
    device_inputs = _to_device(case.inputs, device)
    device_kwargs = _to_device(case.kwargs, device)
    records, save_intermediate = capture_inductor_outputs(npu_dump_dir)
    with intermediate_hook(save_intermediate):
        with torch.no_grad():
            out = compiled(*device_inputs, **device_kwargs)
            torch.npu.synchronize()
    diagnostics = autofuse_diagnostics()
    return _to_cpu(out), records, {
        "mode": "torch_inductor_npu_ext_autofuse",
        "backend_package": package_root,
        "compile_dir": str(compile_dir),
        "autofuse_dump_dir": str(autofuse_dump_dir),
        "captured_nodes": list(records),
        "diagnostics": diagnostics,
    }


def run_torch_case(
    *,
    case: TorchCase,
    script: str | Path,
    out_dir: str | Path,
    html_path: str | Path | None = None,
    cann_root: str | Path | None = None,
    device: str = "npu:0",
    entry: str | None = None,
    invocation_metadata: dict[str, Any] | None = None,
    rtol: float | None = None,
    atol: float | None = None,
) -> dict[str, Any]:
    root = Path(out_dir).resolve()
    _clean_dir(root)
    cpu_dump = root / "cpu_dump"
    npu_eager_dump = root / "npu_eager_dump"
    npu_dump = root / "npu_dump"
    compile_dir = root / "inductor_compile"
    autofuse_dump = root / "autofuse_dump"
    cpu_dump.mkdir(parents=True, exist_ok=True)
    npu_eager_dump.mkdir(parents=True, exist_ok=True)
    npu_dump.mkdir(parents=True, exist_ok=True)
    compile_dir.mkdir(parents=True, exist_ok=True)
    autofuse_dump.mkdir(parents=True, exist_ok=True)

    cpu_graph_data, cpu_outputs = export_cpu_graph_and_dump(case, cpu_dump)
    npu_eager_graph_data, npu_eager_outputs = export_graph_and_dump(case, npu_eager_dump, device=device)
    _, npu_records, native_capture = run_npu_model(
        case,
        npu_dump,
        compile_dir,
        autofuse_dump,
        cann_root=cann_root,
        device=device,
    )
    diagnostics = native_capture.pop("diagnostics", {})
    diagnostics_path = root / "autofuse_diagnostics.json"
    _save_json(diagnostics_path, diagnostics)
    npu_graph_data = build_inductor_autofuse_graph(compile_dir, npu_records, cpu_outputs, diagnostics)

    cpu_graph = root / "cpu_graph.json"
    npu_eager_graph = root / "npu_eager_graph.json"
    npu_graph = root / "npu_graph.json"
    _save_json(cpu_graph, cpu_graph_data)
    _save_json(npu_eager_graph, npu_eager_graph_data)
    _save_json(npu_graph, npu_graph_data)
    cpu_manifest = root / "cpu_manifest.json"
    npu_eager_manifest = root / "npu_eager_manifest.json"
    npu_manifest = root / "npu_manifest.json"
    build_manifest(side="cpu", graph_path=cpu_graph, dump_dir=cpu_dump, out_path=cpu_manifest, model_name=case.name)
    build_manifest(
        side="npu_eager",
        graph_path=npu_eager_graph,
        dump_dir=npu_eager_dump,
        out_path=npu_eager_manifest,
        model_name=case.name,
    )
    build_manifest(side="npu", graph_path=npu_graph, dump_dir=npu_dump, out_path=npu_manifest, model_name=case.name)

    report = compare_three_manifests(
        cpu_manifest,
        npu_eager_manifest,
        npu_manifest,
        rtol=rtol,
        atol=atol,
    )
    attach_autofuse_graphs(report, autofuse_dump)
    native_capture["graph_count"] = len(report["graphs"]["autofuse"]["graphs"])
    native_capture["npu_eager_nodes"] = len(npu_eager_graph_data["graph"][0]["op"])
    native_capture["npu_eager_dump_nodes"] = len(npu_eager_outputs)
    native_capture["inductor_traces"] = npu_graph_data["metadata"]["inductor_traces"]
    native_capture["diagnostics"] = {
        "path": str(diagnostics_path),
        "lowering_fallbacks": npu_graph_data["metadata"]["lowering_fallbacks"],
        "fusion_decisions": npu_graph_data["metadata"]["fusion_decisions"],
    }
    if invocation_metadata:
        native_capture["invocation"] = invocation_metadata
    report["graphs"]["autofuse"]["capture"] = native_capture
    save_report(report, root / "report.json", root / "report.csv")
    html = Path(html_path).resolve() if html_path else root / "report.html"
    write_report_html(report, html, title=f"Autofuse {case.name} Accuracy Report")
    summary: dict[str, Any] = {
        "example": case.name,
        "script": str(Path(script).resolve()),
        "framework": "pytorch",
        "backend": "inductor_autofuse",
        "out_dir": str(root),
        "html": str(html),
        "cpu_nodes": len(cpu_graph_data["graph"][0]["op"]),
        "npu_eager_nodes": len(npu_eager_graph_data["graph"][0]["op"]),
        "npu_nodes": len(npu_graph_data["graph"][0]["op"]),
        "matched_nodes": report["summary"].get("compared", report["summary"]["total"]),
        "unmatched_nodes": report["summary"].get("unmatched", 0),
        "coverage": report["summary"].get("coverage", 1.0),
        "topology": npu_graph_data["metadata"]["topology"],
        "autofuse_graphs": native_capture["graph_count"],
        "comparisons": list(report.get("comparisons", {})),
        "inductor_traces": native_capture["inductor_traces"],
        "passed": report["summary"].get("status") == "PASS",
        "summary": report["summary"],
    }
    if entry is not None:
        summary["entry"] = entry
    if invocation_metadata:
        summary["invocation"] = invocation_metadata
    _save_json(root / "run_summary.json", summary)
    return summary


def run_torch_model(
    *,
    script: str | Path,
    out_dir: str | Path,
    html_path: str | Path | None = None,
    cann_root: str | Path | None = None,
    device: str = "npu:0",
    entry: str = "create_case",
    rtol: float | None = None,
    atol: float | None = None,
) -> dict[str, Any]:
    case = load_torch_case(script, entry)
    return run_torch_case(
        case=case,
        script=script,
        out_dir=out_dir,
        html_path=html_path,
        cann_root=cann_root,
        device=device,
        entry=entry,
        rtol=rtol,
        atol=atol,
    )


def format_run_summary(summary: dict[str, Any]) -> str:
    data = summary["summary"]
    return (
        f"example={summary['example']} "
        f"{data.get('status', 'PASS' if summary['passed'] else 'FAIL')} "
        f"passed={data['passed']}/{data.get('compared', data['total'])} "
        f"coverage={data.get('compared', data['total'])}/{data['total']} "
        f"max_abs={data['max_abs_error']:.3e} "
        f"max_rel={data['max_rel_error']:.3e}\n"
        f"html: {summary['html']}\n"
        f"out_dir: {summary['out_dir']}\n"
        f"matched_nodes: {summary['matched_nodes']}\n"
        f"autofuse_graphs: {summary['autofuse_graphs']}"
    )

from __future__ import annotations

import copy
import importlib.util
import os
import re
import sys
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from autofusedebugger.autofuse_graph import attach_autofuse_graphs, collect_autofuse_graphs, enable_autofuse_graph_dump
from autofusedebugger.compare import compare_manifests, compare_three_manifests, save_report
from autofusedebugger.graph import build_manifest
from autofusedebugger.html import write_report_html
from autofusedebugger.artifact_io import tensor_to_numpy
from autofusedebugger.inductor_autofuse import (
    activate_autofuse_backend,
    autofuse_diagnostics,
    build_inductor_autofuse_graph,
    capture_inductor_outputs,
    install_autofuse_intermediate_hooks,
    prefusion_graphs,
    reset_autofuse_diagnostics,
)
from autofusedebugger.progress import progress_message, progress_stage
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
    fullgraph: bool = False
    dynamic: bool = False
    training: bool | None = None
    grad_enabled: bool = True
    autocast: dict[str, dict[str, Any]] = field(default_factory=dict)
    rewrite_device_hints: bool = False


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
        fullgraph = bool(raw.get("fullgraph", False))
        dynamic = bool(raw.get("dynamic", False))
        rewrite_device_hints = bool(raw.get("rewrite_device_hints", False))
    elif isinstance(raw, tuple) and len(raw) == 2:
        model, inputs = raw
        kwargs = {}
        name = Path(script).stem
        fullgraph = False
        dynamic = False
        rewrite_device_hints = False
    else:
        raise ValueError(f"{entry}() must return (model, inputs) or a dict")
    if not isinstance(inputs, (tuple, list)):
        inputs = (inputs,)
    if not isinstance(kwargs, dict):
        raise ValueError(f"{entry}() 'kwargs' must be a dict")
    try:
        import torch

        grad_enabled = bool(torch.is_grad_enabled())
    except Exception:
        grad_enabled = True
    return TorchCase(
        model=model,
        inputs=tuple(inputs),
        kwargs=kwargs,
        name=name,
        fullgraph=fullgraph,
        dynamic=dynamic,
        training=bool(getattr(model, "training", False)),
        grad_enabled=grad_enabled,
        rewrite_device_hints=rewrite_device_hints,
    )


def _tree_map(value: Any, fn: Any) -> Any:
    try:
        from torch.utils import _pytree
    except Exception:
        if isinstance(value, tuple):
            return tuple(_tree_map(item, fn) for item in value)
        if isinstance(value, list):
            return [_tree_map(item, fn) for item in value]
        if isinstance(value, dict):
            return {key: _tree_map(item, fn) for key, item in value.items()}
        return fn(value)
    return _pytree.tree_map(fn, value)


def _to_cpu(value: Any) -> Any:
    try:
        import torch
    except Exception:
        return value

    def convert(item: Any) -> Any:
        if isinstance(item, torch.Tensor):
            return item.detach().cpu()
        return item

    return _tree_map(value, convert)


def _to_device(value: Any, device: str) -> Any:
    try:
        import torch
    except Exception:
        return value

    def convert(item: Any) -> Any:
        if isinstance(item, torch.Tensor):
            return item.to(device)
        if isinstance(item, np.ndarray) and item.dtype.kind in "biufc":
            if any(stride < 0 for stride in item.strides):
                item = np.ascontiguousarray(item)
            return torch.as_tensor(item, device=device)
        return item

    return _tree_map(value, convert)


def _clone_execution_value(value: Any) -> Any:
    try:
        import torch
    except Exception:
        return copy.deepcopy(value)

    def clone(item: Any) -> Any:
        if isinstance(item, torch.Tensor):
            copied = item.detach().clone()
            if item.requires_grad and (copied.is_floating_point() or copied.is_complex()):
                copied.requires_grad_(True)
            return copied
        if isinstance(item, np.ndarray):
            return item.copy()
        try:
            return copy.deepcopy(item)
        except Exception:
            return item

    return _tree_map(value, clone)


def _stage_case(case: TorchCase, stage: str) -> TorchCase:
    try:
        model = copy.deepcopy(case.model)
    except Exception as exc:
        raise RuntimeError(
            f"{stage} requires an independent model copy to avoid changing the captured model state; "
            "make the model deepcopyable or use a model-specific adapter"
        ) from exc
    return replace(
        case,
        model=model,
        inputs=tuple(_clone_execution_value(case.inputs)),
        kwargs=dict(_clone_execution_value(case.kwargs)),
    )


def _autocast_context(case: TorchCase, device: str):
    import torch

    device_type = str(device).split(":", 1)[0]
    config = case.autocast.get(device_type)
    if not config or not config.get("enabled", False):
        return nullcontext()
    dtype_name = str(config.get("dtype", "")).removeprefix("torch.")
    dtype = getattr(torch, dtype_name, None)
    kwargs: dict[str, Any] = {"device_type": device_type, "enabled": True}
    if dtype is not None:
        kwargs["dtype"] = dtype
    if "cache_enabled" in config:
        kwargs["cache_enabled"] = bool(config["cache_enabled"])
    return torch.autocast(**kwargs)


@contextmanager
def _execution_context(case: TorchCase, device: str):
    import torch

    with torch.set_grad_enabled(case.grad_enabled):
        with _autocast_context(case, device):
            yield


_DEVICE_HINT_NAMES = ("dev", "device")
_DEVICE_HINT_RE = re.compile(r"^(?:cpu|cuda|npu|xpu|mps|meta|privateuseone)(?::\d+)?$")


@contextmanager
def _model_device_hints(model: Any, device: str, *, enabled: bool = False):
    """Optionally align explicit module device hints with the execution stage."""
    if not enabled:
        yield
        return
    import torch

    restored: list[tuple[Any, str, Any]] = []
    for module in model.modules():
        attributes = vars(module)
        for name in _DEVICE_HINT_NAMES:
            if name not in attributes:
                continue
            value = attributes[name]
            if isinstance(value, str):
                if not _DEVICE_HINT_RE.fullmatch(value):
                    continue
                replacement: Any = str(device)
            elif isinstance(value, torch.device):
                replacement = torch.device(device)
            else:
                continue
            restored.append((module, name, value))
            setattr(module, name, replacement)
    try:
        yield
    finally:
        for module, name, value in reversed(restored):
            setattr(module, name, value)


def _input_info(inputs: tuple[Any, ...]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    try:
        import torch
    except Exception:
        torch = None
    for idx, value in enumerate(inputs):
        if torch is not None and isinstance(value, torch.Tensor):
            result[f"input_{idx}"] = {
                "shape": tuple(int(dim) for dim in value.shape),
                "dtype": str(value.dtype).removeprefix("torch."),
            }
        elif isinstance(value, np.ndarray):
            result[f"input_{idx}"] = {
                "shape": tuple(int(dim) for dim in value.shape),
                "dtype": str(value.dtype),
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
    for spec in _export_input_specs(exported):
        kind = spec.kind.name
        if kind in {"PARAMETER", "BUFFER"}:
            args.append(exported.state_dict[spec.target])
        elif kind == "CONSTANT_TENSOR":
            args.append(exported.constants[spec.target])
        elif kind == "USER_INPUT":
            args.append(user_inputs[user_idx])
            user_idx += 1
    return args


def _export_input_specs(exported: Any) -> list[Any]:
    supported = {"PARAMETER", "BUFFER", "CONSTANT_TENSOR", "USER_INPUT"}
    return [spec for spec in exported.graph_signature.input_specs if spec.kind.name in supported]


def export_graph_and_dump(
    case: TorchCase,
    dump_dir: Path,
    *,
    device: str = "cpu",
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    import torch

    model = case.model.to(device)
    if case.training is not None:
        model.train(case.training)
    inputs = _to_device(case.inputs, device)
    kwargs = _to_device(case.kwargs, device)
    with _model_device_hints(model, device, enabled=case.rewrite_device_hints):
        with _execution_context(case, device):
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

            CaptureInterpreter(exported.graph_module).run(*args)

    input_specs = _export_input_specs(exported)
    input_meta = _input_info(tuple(args))
    ops: list[dict[str, Any]] = []
    included: set[str] = set()
    for idx, spec in enumerate(input_specs):
        if not hasattr(spec.arg, "name"):
            continue
        name = spec.arg.name
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
    try:
        return export_graph_and_dump(case, cpu_dump, device="cpu")
    except RuntimeError as exc:
        if case.rewrite_device_hints or "Unhandled FakeTensor Device Propagation" not in str(exc):
            raise

        progress_message(
            "WARNING",
            "CPU export detected mixed FakeTensor devices; retrying the independent CPU model copy "
            "with dev/device hints mapped to cpu",
        )
        try:
            retry_case = _stage_case(case, "CPU graph capture device retry")
            return export_graph_and_dump(replace(retry_case, rewrite_device_hints=True), cpu_dump, device="cpu")
        except Exception as retry_exc:
            raise RuntimeError(
                "CPU export still failed after mapping dev/device hints to cpu. "
                f"Initial error: {exc}; retry error: {retry_exc}"
            ) from retry_exc


def _prefusion_device_value(value: Any, device: str) -> Any:
    import torch

    if isinstance(value, torch.device):
        return torch.device(device)
    if isinstance(value, tuple):
        return tuple(_prefusion_device_value(item, device) for item in value)
    if isinstance(value, list):
        return [_prefusion_device_value(item, device) for item in value]
    if isinstance(value, dict):
        return {key: _prefusion_device_value(item, device) for key, item in value.items()}
    return value


def _prefusion_graph_copy(gm: Any, device: str) -> Any:
    try:
        copied = copy.deepcopy(gm)
    except Exception as exc:
        raise RuntimeError("failed to copy the Inductor pre-fusion GraphModule") from exc
    copied.to(device)
    for node in copied.graph.nodes:
        node.args = _prefusion_device_value(node.args, device)
        node.kwargs = _prefusion_device_value(node.kwargs, device)
    copied.recompile()
    return copied


def _attention_argument(args: tuple[Any, ...], kwargs: dict[str, Any], index: int, name: str, default: Any = None) -> Any:
    if name in kwargs:
        return kwargs[name]
    return args[index] if index < len(args) else default


def _cpu_npu_fusion_attention(*args: Any, **kwargs: Any) -> Any:
    """Replay the common inference form of torch_npu's fused attention on CPU."""
    import torch.nn.functional as functional

    if len(args) < 3:
        raise NotImplementedError("npu_fusion_attention requires query, key and value")

    query, key, value = args[:3]
    head_num = _attention_argument(args, kwargs, 3, "head_num")
    input_layout = _attention_argument(args, kwargs, 4, "input_layout", "BNSD")
    pse = _attention_argument(args, kwargs, 5, "pse")
    padding_mask = _attention_argument(args, kwargs, 6, "padding_mask")
    atten_mask = _attention_argument(args, kwargs, 7, "atten_mask")
    scale = _attention_argument(args, kwargs, 8, "scale")
    keep_prob = _attention_argument(args, kwargs, 9, "keep_prob", 1.0)

    if pse is not None:
        raise NotImplementedError("CPU replay does not support npu_fusion_attention pse")
    if keep_prob != 1.0:
        raise NotImplementedError("CPU replay only supports inference npu_fusion_attention with keep_prob=1.0")

    layout = str(input_layout).upper()
    original_shape = tuple(query.shape)
    if layout == "BNSD":
        to_sdpa = lambda tensor: tensor
        from_sdpa = lambda tensor: tensor
    elif layout == "BSND":
        to_sdpa = lambda tensor: tensor.transpose(1, 2)
        from_sdpa = lambda tensor: tensor.transpose(1, 2)
    elif layout == "BSH":
        if head_num is None or query.dim() != 3 or query.shape[-1] % int(head_num) != 0:
            raise NotImplementedError("CPU replay cannot infer head dimensions for BSH attention")
        head_dim = query.shape[-1] // int(head_num)

        def to_sdpa(tensor):
            return tensor.reshape(tensor.shape[0], tensor.shape[1], int(head_num), head_dim).transpose(1, 2)

        def from_sdpa(tensor):
            return tensor.transpose(1, 2).reshape(original_shape)

    else:
        raise NotImplementedError(f"CPU replay does not support npu_fusion_attention layout={input_layout!r}")

    attention_mask = atten_mask if atten_mask is not None else padding_mask
    sdpa_kwargs: dict[str, Any] = {
        "attn_mask": attention_mask,
        "dropout_p": 0.0,
        "is_causal": False,
    }
    if scale is not None:
        sdpa_kwargs["scale"] = scale
    result = functional.scaled_dot_product_attention(to_sdpa(query), to_sdpa(key), to_sdpa(value), **sdpa_kwargs)
    # torch_npu exposes attention output as the first item of a seven-item result tuple.
    return from_sdpa(result), None, None, None, 0, 0, 0


def _cpu_mm_dtype_fallback(*args: Any, **kwargs: Any) -> Any:
    import torch

    if len(args) < 2 or kwargs:
        raise NotImplementedError("CPU replay only supports positional aten.mm arguments")
    lhs, rhs = args[:2]
    if not isinstance(lhs, torch.Tensor) or not isinstance(rhs, torch.Tensor):
        raise NotImplementedError("CPU replay aten.mm dtype fallback requires Tensor operands")
    return torch.mm(lhs, rhs.to(dtype=lhs.dtype))


def _cpu_prefusion_fallback(target: Any, args: tuple[Any, ...] | None = None) -> Any:
    if "npu_fusion_attention" in str(target):
        return _cpu_npu_fusion_attention
    if "aten.mm" in str(target) and args is not None and len(args) >= 2:
        try:
            import torch

            lhs, rhs = args[:2]
            if isinstance(lhs, torch.Tensor) and isinstance(rhs, torch.Tensor) and lhs.dtype != rhs.dtype:
                return _cpu_mm_dtype_fallback
        except Exception:
            pass
    return None


def _prefusion_display_names(graph_specs: list[dict[str, Any]]) -> dict[tuple[int, str], str]:
    counts: dict[str, int] = {}
    for spec in graph_specs:
        for node in spec["gm"].graph.nodes:
            if node.op in {"placeholder", "call_function"}:
                counts[node.name] = counts.get(node.name, 0) + 1
    result = {}
    for graph_index, spec in enumerate(graph_specs):
        for node in spec["gm"].graph.nodes:
            if node.op not in {"placeholder", "call_function"}:
                continue
            name = node.name
            result[(graph_index, name)] = f"graph{graph_index}.{name}" if counts[name] > 1 else name
    return result


def _install_prefusion_runtime_capture(
    runtime_context: dict[str, Any] | None = None,
) -> tuple[dict[int, tuple[Any, ...]], Any]:
    """Capture the real flattened arguments passed to each compiled FX graph."""
    import importlib

    compile_fx_module = importlib.import_module("torch._inductor.compile_fx")
    original_compile_fx = compile_fx_module.compile_fx
    runtime_inputs: dict[int, tuple[Any, ...]] = {}
    runtime_call_counts: dict[int, int] = {}
    if runtime_context is None:
        runtime_context = {}
    runtime_context["runtime_call_counts"] = runtime_call_counts

    def compile_fx_wrapper(model, example_inputs, *args, **kwargs):  # type: ignore[no-untyped-def]
        if args:
            inner_compile = args[0]
            wrapped_args = list(args)
        else:
            inner_compile = kwargs.get("inner_compile")
            wrapped_args = []
        if inner_compile is None:
            inner_compile = original_compile_fx.__defaults__[0]

        def capture_inner(gm, inner_inputs, *inner_args, **inner_kwargs):  # type: ignore[no-untyped-def]
            compiled_code = inner_compile(gm, inner_inputs, *inner_args, **inner_kwargs)
            if not callable(compiled_code):
                return compiled_code

            def run_compiled(*runtime_args, **runtime_kwargs):  # type: ignore[no-untyped-def]
                values = runtime_args
                if len(runtime_args) == 1 and isinstance(runtime_args[0], (list, tuple)):
                    values = tuple(runtime_args[0])
                gm_id = id(gm)
                call_index = runtime_call_counts.get(gm_id, 0)
                runtime_call_counts[gm_id] = call_index + 1
                runtime_inputs[id(gm)] = tuple(values)
                previous = runtime_context.get("current")
                runtime_context["current"] = {"gm_id": gm_id, "call_index": call_index}
                try:
                    if len(runtime_args) == 1 and isinstance(runtime_args[0], (list, tuple)):
                        return compiled_code(*runtime_args, **runtime_kwargs)
                    return compiled_code(list(runtime_args), **runtime_kwargs)
                finally:
                    if previous is None:
                        runtime_context.pop("current", None)
                    else:
                        runtime_context["current"] = previous

            return run_compiled

        if wrapped_args:
            wrapped_args[0] = capture_inner
            return original_compile_fx(model, example_inputs, *wrapped_args, **kwargs)
        kwargs = dict(kwargs)
        kwargs["inner_compile"] = capture_inner
        return original_compile_fx(model, example_inputs, **kwargs)

    compile_fx_module.compile_fx = compile_fx_wrapper

    def restore() -> None:
        compile_fx_module.compile_fx = original_compile_fx

    return runtime_inputs, restore


def _capture_prefusion_graph(
    gm: Any,
    example_inputs: tuple[Any, ...],
    graph_index: int,
    device: str,
    dump_dir: Path,
    display_names: dict[tuple[int, str], str],
    case: TorchCase,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    import torch

    module = _prefusion_graph_copy(gm, device)
    dump_dir.mkdir(parents=True, exist_ok=True)
    inputs = _to_device(_clone_execution_value(example_inputs), device)
    saved: dict[str, dict[str, Any]] = {}
    cpu_replay_fallbacks: list[dict[str, str]] = []

    class CaptureInterpreter(torch.fx.Interpreter):
        def run_node(self, node):  # type: ignore[no-untyped-def]
            node_args = node_kwargs = None
            if device == "cpu" and node.op == "call_function":
                node_args, node_kwargs = self.fetch_args_kwargs_from_env(node)
            fallback = (
                _cpu_prefusion_fallback(node.target, node_args)
                if device == "cpu" and node.op == "call_function"
                else None
            )
            if fallback is None:
                result = super().run_node(node)
            else:
                try:
                    result = fallback(*node_args, **node_kwargs)
                except Exception as exc:
                    raise RuntimeError(
                        f"CPU pre-fusion replay fallback failed for {node.name} target={node.target}: {exc}"
                    ) from exc
                cpu_replay_fallbacks.append(
                    {"node": node.name, "target": str(node.target), "fallback": fallback.__name__}
                )
            if node.op != "call_function":
                return result
            tensor = _first_tensor(result, torch=torch)
            if tensor is None:
                return result
            display_name = display_names[(graph_index, node.name)]
            array, dtype = tensor_to_numpy(tensor)
            path = dump_dir / f"{display_name}.output.0.npy"
            np.save(path, array)
            saved[display_name] = {
                "shape": tuple(int(dim) for dim in array.shape),
                "dtype": dtype,
                "size": int(array.size),
                "path": path,
            }
            return result

    with _model_device_hints(module, device, enabled=case.rewrite_device_hints):
        with _execution_context(case, device):
            CaptureInterpreter(module).run(*inputs)

    ops: list[dict[str, Any]] = []
    included: set[str] = set()
    placeholders = [node for node in module.graph.nodes if node.op == "placeholder"]
    for index, node in enumerate(placeholders):
        value = inputs[index] if index < len(inputs) else None
        if hasattr(value, "shape"):
            shape = tuple(int(dim) for dim in value.shape)
            dtype = str(value.dtype).removeprefix("torch.")
        else:
            shape = ()
            dtype = type(value).__name__
        display_name = display_names[(graph_index, node.name)]
        ops.append(
            {
                "name": display_name,
                "display_name": node.name,
                "canonical_id": f"graph{graph_index}::{node.name}",
                "type": "Data",
                "output_desc": [_desc(dtype, shape)],
            }
        )
        included.add(node.name)

    for node in module.graph.nodes:
        if node.op != "call_function":
            continue
        display_name = display_names[(graph_index, node.name)]
        info = saved.get(display_name)
        if info is None:
            continue
        inputs_for_node = []
        seen_inputs = set()
        for input_name in _iter_fx_inputs((node.args, node.kwargs)):
            if input_name not in included or input_name in seen_inputs:
                continue
            inputs_for_node.append(f"{display_names[(graph_index, input_name)]}:0")
            seen_inputs.add(input_name)
        ops.append(
            {
                "name": display_name,
                "display_name": node.name,
                "canonical_id": f"graph{graph_index}::{node.name}",
                "type": _target_type(node.target),
                "input": inputs_for_node,
                "output_desc": [_desc(info["dtype"], info["shape"])],
            }
        )
        included.add(node.name)

    return {
        "ops": ops,
        "metadata": {
            "graph_index": graph_index,
            "canonical_graph": "Inductor pre-fusion FX",
            "cpu_replay_fallbacks": cpu_replay_fallbacks,
        },
    }, saved


def capture_prefusion_graphs(
    graph_specs: list[dict[str, Any]],
    case: TorchCase,
    cpu_dump: Path,
    npu_eager_dump: Path,
    *,
    device: str,
    with_cpu: bool = False,
) -> tuple[dict[str, Any] | None, dict[str, dict[str, Any]], dict[str, Any], dict[str, dict[str, Any]]]:
    if not graph_specs:
        raise RuntimeError("Inductor did not expose a transformed pre-fusion FX graph")
    display_names = _prefusion_display_names(graph_specs)
    cpu_graphs: list[dict[str, Any]] = []
    npu_graphs = []
    cpu_outputs: dict[str, dict[str, Any]] = {}
    npu_outputs: dict[str, dict[str, Any]] = {}
    for graph_index, spec in enumerate(graph_specs):
        if with_cpu:
            cpu_graph, saved_cpu = _capture_prefusion_graph(
                spec["gm"], tuple(spec["example_inputs"]), graph_index, "cpu", cpu_dump, display_names, case
            )
            cpu_graphs.append(cpu_graph)
            cpu_outputs.update(saved_cpu)
        npu_graph, saved_npu = _capture_prefusion_graph(
            spec["gm"], tuple(spec["example_inputs"]), graph_index, device, npu_eager_dump, display_names, case
        )
        npu_graphs.append(npu_graph)
        npu_outputs.update(saved_npu)

    def merge(graphs: list[dict[str, Any]], side: str) -> dict[str, Any]:
        ops = [op for graph in graphs for op in graph["ops"]]
        cpu_replay_fallbacks = [
            fallback
            for graph in graphs
            for fallback in graph["metadata"].get("cpu_replay_fallbacks", [])
        ]
        return {
            "graph": [{"op": ops}],
            "metadata": {
                "topology": "Inductor pre-fusion FX",
                "canonical_graph": True,
                "side": side,
                "graph_count": len(graphs),
                "cpu_replay_fallbacks": cpu_replay_fallbacks,
            },
        }

    return (
        merge(cpu_graphs, "inductor_prefusion_cpu") if with_cpu else None,
        cpu_outputs,
        merge(npu_graphs, "inductor_prefusion_npu_eager"),
        npu_outputs,
    )


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
    os.environ["TORCHINDUCTOR_NPU_BACKEND"] = "ascendc"
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
    model = case.model.to(device)
    if case.training is not None:
        model.train(case.training)
    compile_options = {"npu_backend": "ascendc"} if package_root.startswith("torch_npu.") else None
    device_inputs = _to_device(case.inputs, device)
    device_kwargs = _to_device(case.kwargs, device)
    runtime_context: dict[str, Any] = {}
    records, save_intermediate = capture_inductor_outputs(npu_dump_dir, runtime_context)
    runtime_prefusion_inputs, restore_prefusion_runtime_capture = _install_prefusion_runtime_capture(
        runtime_context
    )
    with _model_device_hints(model, device, enabled=case.rewrite_device_hints):
        with _execution_context(case, device):
            try:
                compiled = torch.compile(model, dynamic=case.dynamic, fullgraph=case.fullgraph, options=compile_options)
                with intermediate_hook(save_intermediate):
                    out = compiled(*device_inputs, **device_kwargs)
                    torch.npu.synchronize()
            finally:
                restore_prefusion_runtime_capture()
    diagnostics = autofuse_diagnostics()
    captured_prefusion_graphs = prefusion_graphs()
    runtime_call_counts = runtime_context.get("runtime_call_counts", {})
    prefusion_runtime_selection: dict[int, dict[str, Any]] = {}
    for graph_index, spec in enumerate(captured_prefusion_graphs):
        placeholders = [node.name for node in spec["gm"].graph.nodes if node.op == "placeholder"]
        gm_id = id(spec["gm"])
        call_count = int(runtime_call_counts.get(gm_id, 0))
        if call_count:
            prefusion_runtime_selection[graph_index] = {
                "gm_id": gm_id,
                "call_index": call_count - 1,
            }
        runtime_inputs = runtime_prefusion_inputs.get(gm_id)
        if runtime_inputs is None:
            candidates = [
                values for values in runtime_prefusion_inputs.values() if len(values) == len(placeholders)
            ]
            if len(candidates) == 1:
                runtime_inputs = candidates[0]
                binding_method = "compiled_runtime_shape"
            elif graph_index < len(candidates):
                runtime_inputs = candidates[graph_index]
                binding_method = "compiled_runtime_order"
            else:
                binding_method = "compiled_runtime"
        else:
            binding_method = "compiled_runtime_id"
        if runtime_inputs is not None and len(runtime_inputs) == len(placeholders):
            spec["example_inputs"] = tuple(runtime_inputs)
            progress_message(
                "INFO",
                f"pre-fusion graph={graph_index} runtime binding={binding_method} inputs={len(placeholders)}",
            )
            continue
        else:
            raise RuntimeError(
                f"Inductor did not expose real runtime inputs for pre-fusion graph {graph_index} "
                f"(placeholders={len(placeholders)}, captured={len(runtime_inputs or ())})"
            )
    return _to_cpu(out), records, {
        "mode": "torch_inductor_npu_ext_autofuse",
        "backend_package": package_root,
        "compile_dir": str(compile_dir),
        "autofuse_dump_dir": str(autofuse_dump_dir),
        "captured_nodes": list(records),
        "diagnostics": diagnostics,
        "prefusion_graphs": captured_prefusion_graphs,
        "prefusion_runtime_selection": prefusion_runtime_selection,
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
    save_elementwise_data: bool = False,
    with_cpu: bool = False,
) -> dict[str, Any]:
    root = Path(out_dir).resolve()
    progress_message(
        "INFO",
        f"run started model={case.name} device={device} out_dir={root} with_cpu={with_cpu}",
    )
    _clean_dir(root)
    cpu_dump = root / "cpu_dump"
    npu_eager_dump = root / "npu_eager_dump"
    npu_dump = root / "npu_dump"
    compile_dir = root / "inductor_compile"
    autofuse_dump = root / "autofuse_dump"
    if with_cpu:
        cpu_dump.mkdir(parents=True, exist_ok=True)
    npu_eager_dump.mkdir(parents=True, exist_ok=True)
    npu_dump.mkdir(parents=True, exist_ok=True)
    compile_dir.mkdir(parents=True, exist_ok=True)
    autofuse_dump.mkdir(parents=True, exist_ok=True)

    try:
        with progress_stage("NPU Autofuse execution", detail=f"dump_dir={npu_dump}"):
            _, npu_records, native_capture = run_npu_model(
                _stage_case(case, "NPU Autofuse execution"),
                npu_dump,
                compile_dir,
                autofuse_dump,
                cann_root=cann_root,
                device=device,
            )
    except Exception as exc:
        detail = str(exc)
        if "Check dtype failed" in detail:
            detail += (
                "\nAutofuse rejected an ASCIR dtype conversion. The debugger preserved "
                f"fullgraph={case.fullgraph} and rewrite_device_hints={case.rewrite_device_hints}. "
                "Compare this with the standalone torch.compile options and inspect the generated "
                "inductor_compile/autofuse_dump files for the reported cast source."
            )
        raise RuntimeError(f"NPU Autofuse execution failed: {type(exc).__name__}: {detail}") from exc
    progress_message("INFO", f"NPU Autofuse captured kernels={len(npu_records)}")
    prefusion_specs = native_capture.pop("prefusion_graphs", [])
    try:
        with progress_stage(
            "Inductor pre-fusion NPU eager capture",
            detail=f"shared transformed FX graph with_cpu={with_cpu}",
        ):
            (
                cpu_graph_data,
                cpu_outputs,
                npu_eager_graph_data,
                npu_eager_outputs,
            ) = capture_prefusion_graphs(
                prefusion_specs,
                _stage_case(case, "Inductor pre-fusion capture"),
                cpu_dump,
                npu_eager_dump,
                device=device,
                with_cpu=with_cpu,
            )
    except Exception as exc:
        raise RuntimeError(f"Inductor pre-fusion capture failed: {type(exc).__name__}: {exc}") from exc
    if with_cpu:
        progress_message("INFO", f"Inductor pre-fusion CPU captured tensors={len(cpu_outputs)}")
    progress_message("INFO", f"Inductor pre-fusion NPU eager captured tensors={len(npu_eager_outputs)}")
    with progress_stage("NPU graph reconstruction", detail=f"compile_dir={compile_dir}"):
        diagnostics = native_capture.pop("diagnostics", {})
        diagnostics_path = root / "autofuse_diagnostics.json"
        _save_json(diagnostics_path, diagnostics)
        # The canonical pre-fusion mapping uses the shared FX graph structure; the
        # eager replay graph is the same GraphModule, so it can serve as the
        # canonical reference when CPU replay was skipped.
        canonical_graph_data = cpu_graph_data if with_cpu else npu_eager_graph_data
        npu_graph_data = build_inductor_autofuse_graph(
            compile_dir,
            npu_records,
            cpu_outputs,
            diagnostics,
            cpu_graph_data=cpu_graph_data,
            prefusion_graph_data=canonical_graph_data,
            prefusion_runtime_selection=native_capture.get("prefusion_runtime_selection"),
        )

    cpu_graph = root / "cpu_graph.json"
    npu_eager_graph = root / "npu_eager_graph.json"
    npu_graph = root / "npu_graph.json"
    with progress_stage("Graph manifest generation", detail=f"out_dir={root}"):
        npu_eager_manifest = root / "npu_eager_manifest.json"
        npu_manifest = root / "npu_manifest.json"
        if with_cpu:
            _save_json(cpu_graph, cpu_graph_data)
            cpu_manifest = root / "cpu_manifest.json"
            build_manifest(
                side="inductor_prefusion_cpu",
                graph_path=cpu_graph,
                dump_dir=cpu_dump,
                out_path=cpu_manifest,
                model_name=case.name,
            )
        _save_json(npu_eager_graph, npu_eager_graph_data)
        _save_json(npu_graph, npu_graph_data)
        build_manifest(
            side="inductor_prefusion_npu_eager",
            graph_path=npu_eager_graph,
            dump_dir=npu_eager_dump,
            out_path=npu_eager_manifest,
            model_name=case.name,
        )
        build_manifest(side="npu_autofuse", graph_path=npu_graph, dump_dir=npu_dump, out_path=npu_manifest, model_name=case.name)

    if with_cpu:
        with progress_stage("Three-way tensor comparison", detail="CPU vs NPU eager vs NPU Autofuse"):
            report = compare_three_manifests(
                cpu_manifest,
                npu_eager_manifest,
                npu_manifest,
                rtol=rtol,
                atol=atol,
                save_elementwise_data=save_elementwise_data,
            )
    else:
        with progress_stage("Two-way tensor comparison", detail="NPU eager vs NPU Autofuse"):
            report = compare_manifests(
                npu_eager_manifest,
                npu_manifest,
                rtol=rtol,
                atol=atol,
                save_elementwise_data=save_elementwise_data,
                progress_label="NPU eager vs NPU Autofuse",
            )
            report["comparison_mode"] = "two_way"
            report["comparisons"] = {
                "npu_eager_npu_autofuse": {
                    "left": "Inductor pre-fusion NPU eager",
                    "right": "NPU Autofuse",
                    "left_graph_key": "npu_eager",
                    "right_graph_key": "npu",
                    "summary": report["summary"],
                    "results": report["results"],
                    "graphs": report["graphs"],
                }
            }
            report["comparison_order"] = ["npu_eager_npu_autofuse"]
            report["graphs"]["npu_eager"] = report["graphs"].pop("cpu")
    with progress_stage("AutoFuse graph collection", detail=f"dump_dir={autofuse_dump}"):
        attach_autofuse_graphs(report, autofuse_dump)
    native_capture["graph_count"] = len(report["graphs"]["autofuse"]["graphs"])
    if with_cpu:
        native_capture["prefusion_cpu_nodes"] = len(cpu_graph_data["graph"][0]["op"])
    native_capture["prefusion_npu_eager_nodes"] = len(npu_eager_graph_data["graph"][0]["op"])
    native_capture["prefusion_source"] = "same Inductor transformed FX GraphModule as Autofuse compile"
    native_capture["prefusion_runtime_input_source"] = "compiled FX runtime boxed inputs"
    native_capture["npu_eager_dump_nodes"] = len(npu_eager_outputs)
    native_capture["inductor_traces"] = npu_graph_data["metadata"]["inductor_traces"]
    native_capture["execution_policy"] = {
        "training": case.training,
        "grad_enabled": case.grad_enabled,
        "autocast_devices": sorted(case.autocast),
        "fullgraph": case.fullgraph,
        "dynamic": case.dynamic,
        "rewrite_device_hints": case.rewrite_device_hints,
        "save_elementwise_data": save_elementwise_data,
        "stage_model_copies": True,
    }
    native_capture["diagnostics"] = {
        "path": str(diagnostics_path),
        "lowering_fallbacks": npu_graph_data["metadata"]["lowering_fallbacks"],
        "fusion_decisions": npu_graph_data["metadata"]["fusion_decisions"],
    }
    if invocation_metadata:
        native_capture["invocation"] = invocation_metadata
    report["graphs"]["autofuse"]["capture"] = native_capture
    html = Path(html_path).resolve() if html_path else root / "report.html"
    with progress_stage("Report JSON and CSV write", detail=f"out_dir={root}"):
        save_report(report, root / "report.json", root / "report.csv")
    write_report_html(report, html, title=f"Autofuse {case.name} Accuracy Report")
    summary: dict[str, Any] = {
        "example": case.name,
        "script": str(Path(script).resolve()),
        "framework": "pytorch",
        "backend": "inductor_autofuse",
        "out_dir": str(root),
        "html": str(html),
        "cpu_nodes": len(cpu_graph_data["graph"][0]["op"]) if with_cpu else 0,
        "npu_eager_nodes": len(npu_eager_graph_data["graph"][0]["op"]),
        "npu_nodes": len(npu_graph_data["graph"][0]["op"]),
        "prefusion_cpu_nodes": len(cpu_graph_data["graph"][0]["op"]) if with_cpu else 0,
        "prefusion_npu_eager_nodes": len(npu_eager_graph_data["graph"][0]["op"]),
        "autofuse_nodes": len(npu_graph_data["graph"][0]["op"]),
        "matched_nodes": report["summary"].get("compared", report["summary"]["total"]),
        "unmatched_nodes": report["summary"].get("unmatched", 0),
        "coverage": report["summary"].get("coverage", 1.0),
        "topology": npu_graph_data["metadata"]["topology"],
        "autofuse_graphs": native_capture["graph_count"],
        "comparisons": list(report.get("comparisons", {})),
        "execution_policy": native_capture["execution_policy"],
        "inductor_traces": native_capture["inductor_traces"],
        "passed": report["summary"].get("status") == "PASS",
        "summary": report["summary"],
    }
    if entry is not None:
        summary["entry"] = entry
    if invocation_metadata:
        summary["invocation"] = invocation_metadata
    _save_json(root / "run_summary.json", summary)
    progress_message("INFO", f"run completed html={html}")
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
    save_elementwise_data: bool = False,
    fullgraph: bool | None = None,
    rewrite_device_hints: bool = False,
    with_cpu: bool = False,
) -> dict[str, Any]:
    case = load_torch_case(script, entry)
    if fullgraph is not None:
        case = replace(case, fullgraph=fullgraph)
    if rewrite_device_hints:
        case = replace(case, rewrite_device_hints=True)
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
        save_elementwise_data=save_elementwise_data,
        with_cpu=with_cpu,
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
        f"prefusion_nodes: cpu={summary.get('prefusion_cpu_nodes', summary.get('cpu_nodes', 0))} "
        f"npu_eager={summary.get('prefusion_npu_eager_nodes', summary.get('npu_eager_nodes', 0))} "
        f"autofuse={summary.get('autofuse_nodes', summary.get('npu_nodes', 0))}\n"
        f"autofuse_graphs: {summary['autofuse_graphs']}"
    )

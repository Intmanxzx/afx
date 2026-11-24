from __future__ import annotations

import ast
import importlib
import os
import re
import sys
import threading
from pathlib import Path
from typing import Any

import numpy as np

from autofusex.execution.artifact_io import tensor_to_numpy
from autofusex.workloads.model.npu_examples import _desc


_FUSION_OUTPUT_MARKER = "::autofusex_output::"
_CAPTURE_LOCK = threading.Lock()
_CAPTURE_STATE = threading.local()
_FUSION_DECISION_GROUPS: list[list[dict[str, Any]]] = []
_LOWERING_FALLBACKS: list[dict[str, Any]] = []
_IMPLICIT_FALLBACK_TARGETS: set[Any] = set()
_PREFUSION_GRAPHS: list[dict[str, Any]] = []
_PREFUSION_GRAPH_IDS: set[int] = set()
_GENERATED_KERNELS: list[dict[str, Any]] = []
_ASCENDC_PACKAGE_ROOTS = ("torch_npu._inductor.ascendc", "inductor_npu_ext")
_DTYPE_NAMES = {
    "b1": "bool",
    "bf16": "bfloat16",
    "f16": "float16",
    "f32": "float32",
    "f64": "float64",
    "i8": "int8",
    "i16": "int16",
    "i32": "int32",
    "i64": "int64",
    "u8": "uint8",
}


def reset_autofuse_diagnostics() -> None:
    with _CAPTURE_LOCK:
        _FUSION_DECISION_GROUPS.clear()
        _LOWERING_FALLBACKS.clear()
        _IMPLICIT_FALLBACK_TARGETS.clear()
        _PREFUSION_GRAPHS.clear()
        _PREFUSION_GRAPH_IDS.clear()
        _GENERATED_KERNELS.clear()
    _CAPTURE_STATE.fusion_group = None


def autofuse_diagnostics() -> dict[str, Any]:
    with _CAPTURE_LOCK:
        return {
            "fusion_decision_groups": [[dict(item) for item in group] for group in _FUSION_DECISION_GROUPS],
            "lowering_fallbacks": [dict(item) for item in _LOWERING_FALLBACKS],
            "generated_kernels": [dict(item) for item in _GENERATED_KERNELS],
        }


def prefusion_graphs() -> list[dict[str, Any]]:
    """Return transformed FX graphs captured from the active Inductor compile."""
    with _CAPTURE_LOCK:
        return [dict(item) for item in _PREFUSION_GRAPHS]


def activate_autofuse_backend() -> str:
    os.environ["TORCHINDUCTOR_NPU_BACKEND"] = "ascendc"
    errors = []
    for package_root in _ASCENDC_PACKAGE_ROOTS:
        try:
            importlib.import_module(package_root)
            return package_root
        except Exception as exc:
            errors.append(f"{package_root}: {type(exc).__name__}: {exc}")
    raise RuntimeError(
        "failed to load the torch_npu bundled AscendC backend or the legacy standalone inductor_npu_ext. "
        + " | ".join(errors)
    )


def install_autofuse_intermediate_hooks(package_root: str | None = None) -> None:
    """Capture native Autofuse outputs and node-level non-fusion diagnostics."""
    package_root = package_root or activate_autofuse_backend()
    npu_module = importlib.import_module(f"{package_root}.npu")
    NPUKernel = npu_module.NPUKernel
    NPUScheduling = npu_module.NPUScheduling

    if getattr(NPUScheduling, "_autofusex_hooks_installed", False):
        return

    from torch._inductor.virtualized import V

    lowering_summary = importlib.import_module(f"{package_root}.lowering.common")._summary
    from torch._inductor.graph import GraphLowering
    from torch._inductor.lowering import lowerings

    state = threading.local()
    original_trace = NPUKernel.tracing_asc
    original_codegen = NPUScheduling.codegen_nodes
    original_scheduling_init = NPUScheduling.__init__
    original_log_can_fuse = NPUScheduling.__dict__["_log_can_fuse_result"].__func__
    original_lowering_fallback = lowering_summary.fallback
    original_graph_call_function = GraphLowering.call_function
    original_graph_lowering_init = GraphLowering.__init__

    def graph_lowering_init(graph, gm, *args, **kwargs):  # type: ignore[no-untyped-def]
        original_graph_lowering_init(graph, gm, *args, **kwargs)
        example_inputs = kwargs.get("example_inputs")
        if example_inputs is None and args:
            example_inputs = args[0]
        if example_inputs is None:
            return
        graph_key = id(gm)
        with _CAPTURE_LOCK:
            if graph_key in _PREFUSION_GRAPH_IDS:
                return
            _PREFUSION_GRAPH_IDS.add(graph_key)
            _PREFUSION_GRAPHS.append(
                {
                    "graph_id": getattr(graph, "graph_id", None),
                    "gm": gm,
                    "example_inputs": tuple(example_inputs),
                }
            )

    def scheduling_init(scheduling, scheduler):  # type: ignore[no-untyped-def]
        original_scheduling_init(scheduling, scheduler)
        with _CAPTURE_LOCK:
            group = len(_FUSION_DECISION_GROUPS)
            _FUSION_DECISION_GROUPS.append([])
        _CAPTURE_STATE.fusion_group = group

    def log_can_fuse_result(  # type: ignore[no-untyped-def]
        cls, kind, node1, node2, result, reason, triton_can_fuse
    ):
        group = getattr(_CAPTURE_STATE, "fusion_group", None)
        if group is not None:
            record = {
                "kind": str(kind),
                "node1": [sub.get_name() for sub in node1.get_nodes()],
                "node2": [sub.get_name() for sub in node2.get_nodes()],
                "result": bool(result),
                "reason": str(reason),
                "triton_can_fuse": triton_can_fuse,
            }
            with _CAPTURE_LOCK:
                if group < len(_FUSION_DECISION_GROUPS):
                    _FUSION_DECISION_GROUPS[group].append(record)
        return original_log_can_fuse(cls, kind, node1, node2, result, reason, triton_can_fuse)

    def lowering_fallback(node, reason):  # type: ignore[no-untyped-def]
        record = {
            "category": "lowering_fallback",
            "node": str(node.name),
            "target": str(node.target),
            "reason": str(reason),
        }
        with _CAPTURE_LOCK:
            if record not in _LOWERING_FALLBACKS:
                _LOWERING_FALLBACKS.append(record)
        return original_lowering_fallback(node, reason)

    def graph_call_function(graph, target, args, kwargs):  # type: ignore[no-untyped-def]
        missing_lowering = target not in lowerings
        result = original_graph_call_function(graph, target, args, kwargs)
        if missing_lowering:
            _IMPLICIT_FALLBACK_TARGETS.add(target)
        if target in _IMPLICIT_FALLBACK_TARGETS:
            node = getattr(graph, "current_node", None)
            record = {
                "category": "implicit_fallback",
                "node": str(getattr(node, "name", target)),
                "target": str(target),
                "reason": "TorchInductor created an implicit fallback because no lowering was registered",
            }
            with _CAPTURE_LOCK:
                if record not in _LOWERING_FALLBACKS:
                    _LOWERING_FALLBACKS.append(record)
        return result

    def tracing_asc(kernel):  # type: ignore[no-untyped-def]
        result = original_trace(kernel)
        pending = getattr(state, "pending", None)
        if pending is None:
            pending = []
            state.pending = pending
        pending.append(result)
        return result

    def codegen_nodes(scheduling, nodes):  # type: ignore[no-untyped-def]
        pending = getattr(state, "pending", None)
        if pending is None:
            pending = []
            state.pending = pending
        start = len(pending)
        original_codegen(scheduling, nodes)
        if len(pending) == start:
            return
        kernel = pending[start]
        live_outputs = set(kernel.args.live_output_buffers())
        output_index = 0
        for node in nodes:
            if node.node is None:
                continue
            buffer_name = node.node.get_name()
            if buffer_name not in live_outputs:
                continue
            origin = node.node.get_origin_node()
            if origin is None:
                continue
            capture_name = f"{kernel.kernel_name}__{origin.name}"
            hook_name = f"{capture_name}{_FUSION_OUTPUT_MARKER}{output_index}"
            V.graph.wrapper_code.writeline(f"run_intermediate_hooks({hook_name!r}, {buffer_name})")
            output_index += 1

        # Record every atomic op this generated kernel fused, independent of the
        # live-output filtering above. `nodes` is the full set of scheduler nodes
        # that went into this one kernel; the live-output filter above only decides
        # which node's buffer gets an intermediate capture hook and would otherwise
        # make us undercount fused ops by dropping all non-output (intermediate)
        # nodes in the fusion group.
        fused_ops: list[dict[str, Any]] = []
        for node in nodes:
            if node.node is None:
                continue
            origin = node.node.get_origin_node()
            fused_ops.append(
                {
                    "buffer_name": node.node.get_name(),
                    "target": str(origin.target) if origin is not None else None,
                }
            )
        with _CAPTURE_LOCK:
            _GENERATED_KERNELS.append(
                {
                    "kernel_name": str(kernel.kernel_name),
                    "fused_ops": fused_ops,
                    "op_count": len(fused_ops),
                }
            )

    NPUKernel.tracing_asc = tracing_asc
    NPUScheduling.codegen_nodes = codegen_nodes
    NPUScheduling.__init__ = scheduling_init
    NPUScheduling._log_can_fuse_result = classmethod(log_can_fuse_result)
    lowering_summary.fallback = lowering_fallback
    GraphLowering.call_function = graph_call_function
    GraphLowering.__init__ = graph_lowering_init
    NPUScheduling._autofusex_hooks_installed = True


def _tensor_leaves(value: Any, torch: Any) -> list[Any]:
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (tuple, list)):
        result = []
        for item in value:
            result.extend(_tensor_leaves(item, torch))
        return result
    if isinstance(value, dict):
        result = []
        for item in value.values():
            result.extend(_tensor_leaves(item, torch))
        return result
    return []


def _capture_name(raw_name: str) -> tuple[str, int]:
    if _FUSION_OUTPUT_MARKER not in raw_name:
        return raw_name, 0
    name, raw_index = raw_name.rsplit(_FUSION_OUTPUT_MARKER, 1)
    return name, int(raw_index)


def capture_inductor_outputs(dump_dir: Path, runtime_context: dict[str, Any] | None = None):
    import torch

    records: dict[str, dict[str, Any]] = {}

    def save(raw_name: str, value: Any) -> None:
        name, base_index = _capture_name(raw_name)
        tensors = _tensor_leaves(value, torch)
        if not tensors:
            return
        if any(int(item["index"]) == base_index for item in records.get(name, {}).get("outputs", [])):
            call_index = 1
            while f"{name}__call{call_index}" in records:
                call_index += 1
            name = f"{name}__call{call_index}"
        record = records.setdefault(name, {"name": name, "outputs": []})
        if runtime_context is not None and "current" in runtime_context:
            current = runtime_context["current"]
            record.setdefault("runtime_gm_id", int(current["gm_id"]))
            record.setdefault("runtime_call_index", int(current["call_index"]))
        for offset, tensor in enumerate(tensors):
            index = base_index + offset
            array, dtype = tensor_to_numpy(tensor)
            array = array.copy()
            path = dump_dir / f"{name}.output.{index}.npy"
            np.save(path, array)
            record["outputs"].append(
                {
                    "index": index,
                    "path": path,
                    "dtype": dtype,
                    "shape": tuple(int(dim) for dim in array.shape),
                    "size": int(array.size),
                }
            )

    return records, save


def _values(text: str) -> list[Any]:
    result = []
    for value in (item.strip() for item in text.split(",")):
        if not value:
            continue
        try:
            result.append(int(value))
        except ValueError:
            result.append(value)
    return result


def _tensor_meta(text: str) -> tuple[str, list[Any]]:
    match = re.match(r"([A-Za-z0-9_]+)\[([^\]]*)\]", text)
    if not match:
        return "unknown", []
    return _DTYPE_NAMES.get(match.group(1), match.group(1)), _values(match.group(2))


def _parse_fx_fragment(text: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    nodes = []
    edges = []
    known: set[str] = set()
    exit_name = ""
    call_re = re.compile(
        r'^\s*#\s+%([A-Za-z0-9_]+)\s+:\s+Tensor\s+"([^"]+)".*?'
        r"call_function\[target=torch\.ops\.aten\.([^\]]+)\]\(args = \((.*?)\), kwargs"
    )
    for line in text.splitlines():
        match = call_re.match(line)
        if match:
            name, tensor_text, target, args = match.groups()
            dtype, shape = _tensor_meta(tensor_text)
            node = {
                "name": name,
                "type": target.split(".", 1)[0],
                "inputs": [],
                "outputs": [{"index": 0, "dtype": dtype, "shape": shape, "format": "ND"}],
                "origin": True,
            }
            for index, source in enumerate(re.findall(r"%([A-Za-z0-9_]+)", args)):
                node["inputs"].append({"src": source, "index": index})
                if source in known:
                    edges.append({"src": source, "dst": name, "dst_input": index})
            nodes.append(node)
            known.add(name)
            continue
        return_match = re.match(r"^\s*#\s+return\s+(.+)", line)
        if return_match:
            returned = re.findall(r"%([A-Za-z0-9_]+)", return_match.group(1))
            if returned:
                exit_name = returned[-1]
    return nodes, edges, exit_name


def _canonical_type(value: str) -> str:
    text = value.lower()
    if text.startswith("aten."):
        text = text[len("aten."):]
    text = text.split(".", 1)[0]
    aliases = {
        "_softmax": "softmax",
        "_unsafe_view": "reshape",
        "cat": "stack",
        "clone": "contiguous",
        "permute": "transpose",
        "reducesum": "sum",
        "reduce_sum": "sum",
        "reducemean": "mean",
        "reduce_mean": "mean",
        "realdiv": "div",
        "reshape": "reshape",
        "t": "transpose",
        "view": "reshape",
    }
    return aliases.get(text, text)


def _parse_ascir_hint(text: str, fx_nodes: list[dict[str, Any]]) -> dict[str, Any] | None:
    start = text.find("graph_hint = ascir.HintGraph")
    end = text.find("graph_hint.infer_dtypes()", start)
    if start < 0 or end < 0:
        return None
    section = text[start:end]
    node_re = re.compile(
        r"^\s*([A-Za-z0-9_]+)\s*=\s*ascir\.ops\.([A-Za-z0-9_]+)\('([^']+)',\s*graph_hint\)",
        re.MULTILINE,
    )
    variables: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for variable, node_type, name in node_re.findall(section):
        variables[variable] = {
            "name": name,
            "type": node_type,
            "inputs": [],
            "outputs": [{"index": 0, "dtype": "unknown", "shape": [], "format": "ND"}],
            "origin": False,
        }
        order.append(variable)
    if not variables:
        return None

    placeholders = []
    for tensor_text in re.findall(
        r'^\s*#\s+%[A-Za-z0-9_]+\s+:\s+Tensor\s+"([^"]+)"\s*=\s*PlaceHolder', text, re.MULTILINE
    ):
        dtype, shape = _tensor_meta(tensor_text)
        placeholders.append({"dtype": dtype, "shape": shape})
    for variable, raw_index in re.findall(
        r"^\s*([A-Za-z0-9_]+)\.attr\.ir_attr\.index\s*=\s*(\d+)", section, re.MULTILINE
    ):
        if variable not in variables or variables[variable]["type"] != "Data":
            continue
        index = int(raw_index)
        if index < len(placeholders):
            variables[variable]["outputs"][0].update(placeholders[index])

    for variable, raw_shape in re.findall(r"^\s*([A-Za-z0-9_]+)\.y\.size\s*=\s*\[([^\]]*)\]", section, re.MULTILINE):
        if variable in variables:
            variables[variable]["outputs"][0]["shape"] = _values(raw_shape)
    for variable, dtype in re.findall(
        r"^\s*([A-Za-z0-9_]+)\.y\.dtype\s*=\s*ascir\.dtypes\.([A-Za-z0-9_]+)", section, re.MULTILINE
    ):
        if variable in variables:
            variables[variable]["outputs"][0]["dtype"] = dtype

    edges = []
    for line in section.splitlines():
        match = re.match(r"^\s*([A-Za-z0-9_]+)\.(x\d*|x)\s*=\s*(?:\[)?([A-Za-z0-9_]+)\.y", line)
        if not match:
            continue
        destination, input_name, source = match.groups()
        if source not in variables or destination not in variables:
            continue
        index_match = re.search(r"(\d+)$", input_name)
        index = max(0, int(index_match.group(1)) - 1) if index_match else len(variables[destination]["inputs"])
        variables[destination]["inputs"].append({"src": variables[source]["name"], "index": index})
        edges.append({"src": variables[source]["name"], "dst": variables[destination]["name"], "dst_input": index})

    origin_counts: dict[str, int] = {}
    for node in fx_nodes:
        kind = _canonical_type(node["type"])
        origin_counts[kind] = origin_counts.get(kind, 0) + 1
    for variable in order:
        node = variables[variable]
        kind = _canonical_type(node["type"])
        if origin_counts.get(kind, 0) > 0:
            node["origin"] = True
            origin_counts[kind] -= 1

    changed = True
    while changed:
        changed = False
        by_name = {node["name"]: node for node in variables.values()}
        for variable in order:
            node = variables[variable]
            output = node["outputs"][0]
            for item in node["inputs"]:
                source = by_name.get(item["src"])
                if source is None:
                    continue
                source_output = source["outputs"][0]
                if output["dtype"] == "unknown" and source_output["dtype"] != "unknown":
                    output["dtype"] = source_output["dtype"]
                    changed = True
                if not output["shape"] and source_output["shape"]:
                    output["shape"] = source_output["shape"]
                    changed = True
    return {
        "source": "inductor_npu_ext ASCIR graph_hint",
        "nodes": [variables[variable] for variable in order],
        "edges": edges,
        "missing": [],
    }


def _benchmark_metadata(path: Path) -> dict[str, Any] | None:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    fx_nodes, fx_edges, exit_name = _parse_fx_fragment(text)
    scheduler_ids = re.findall(
        r"^\s*(op\d+(?:_op\d+)*):\s+(?:SchedulerNode|FusedSchedulerNode)", text, re.MULTILINE
    )
    if not scheduler_ids or not exit_name:
        return None
    kernel_name = path.parent.name
    subgraph = _parse_ascir_hint(text, fx_nodes) or {
        "source": "TorchInductor fused FX fragment",
        "nodes": fx_nodes,
        "edges": fx_edges,
        "missing": [],
    }
    exit_type = next((node["type"] for node in fx_nodes if node["name"] == exit_name), exit_name)
    original_aten_types = []
    for raw_types in re.findall(r"Original ATen:\s*\[([^\]]*)\]", text):
        for target in re.findall(r"\baten\.([A-Za-z0-9_]+)(?:\.[A-Za-z0-9_]+)?", raw_types):
            if target not in original_aten_types:
                original_aten_types.append(target)
    return {
        "kernel_name": kernel_name,
        "capture_name": f"{kernel_name}__{exit_name}",
        "scheduler_ids": scheduler_ids,
        "origin_ops": [node["name"] for node in fx_nodes],
        "exit_name": exit_name,
        "exit_type": exit_type,
        "original_aten_types": original_aten_types,
        "origin_subgraph": subgraph,
    }


def _dependency_names(text: str) -> list[str]:
    return re.findall(r"(?:MemoryDep|StarDep)\((?:name=)?'([^']+)'", text)


def _statement(block: str, field: str) -> str:
    match = re.search(rf"^[A-Za-z0-9_]+\.{field}\s*=\s*\[(.*?)\]\s*$", block, re.MULTILINE | re.DOTALL)
    return match.group(1) if match else ""


def _layout_map(text: str) -> dict[str, dict[str, Any]]:
    layouts: dict[str, dict[str, Any]] = {}
    pattern = re.compile(
        r"^([A-Za-z0-9_]+)_layout\s*=\s*FixedLayout\([^\n]*?torch\.([A-Za-z0-9_]+),\s*"
        r"size=\[([^\]]*)\]",
        re.MULTILINE,
    )
    for name, dtype, raw_shape in pattern.findall(text):
        layouts[name] = {"dtype": dtype, "shape": _values(raw_shape)}
    pattern = re.compile(
        r"^\s*([A-Za-z0-9_]+)\.layout\s*=\s*FixedLayout\([^\n]*?torch\.([A-Za-z0-9_]+),\s*"
        r"size=\[([^\]]*)\]",
        re.MULTILINE,
    )
    for name, dtype, raw_shape in pattern.findall(text):
        layouts[name] = {"dtype": dtype, "shape": _values(raw_shape)}
    return layouts


def _fx_layout_map(path: Path) -> dict[str, dict[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return {}
    layouts = {}
    for name, tensor_text in re.findall(r'\b([A-Za-z0-9_]+):\s+"([A-Za-z0-9_]+\[[^"]*)"', text):
        dtype, shape = _tensor_meta(tensor_text)
        layouts[name] = {"dtype": dtype, "shape": shape}
    return layouts


def _fx_input_layout_map(path: Path, layouts: dict[str, dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    try:
        text = path.read_text(encoding="utf-8")
        text = re.sub(r"^class <lambda>", "class LambdaModule", text, count=1, flags=re.MULTILINE)
        tree = ast.parse(text)
    except (OSError, UnicodeDecodeError, SyntaxError):
        return {}

    def referenced_names(value: ast.AST) -> list[str]:
        if isinstance(value, ast.Name):
            return [value.id] if value.id in layouts else []
        if isinstance(value, (ast.List, ast.Tuple)):
            return [name for item in value.elts for name in referenced_names(item)]
        return []

    result: dict[str, list[dict[str, Any]]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.AnnAssign) or not isinstance(node.target, ast.Name):
            continue
        if not isinstance(node.value, ast.Call):
            continue
        names = [name for arg in node.value.args for name in referenced_names(arg)]
        names.extend(name for keyword in node.value.keywords for name in referenced_names(keyword.value))
        result[node.target.id] = [dict(layouts[name]) for name in names]
    return result


def _ir_blocks(text: str) -> list[dict[str, Any]]:
    starts = list(re.finditer(r"^(op\d+(?:_op\d+)*):\s+([^\n]+)$", text, re.MULTILINE))
    blocks = []
    for index, match in enumerate(starts):
        end = starts[index + 1].start() if index + 1 < len(starts) else len(text)
        block = text[match.start():end]
        writes = _dependency_names(_statement(block, "writes"))
        dependencies = _dependency_names(_statement(block, "unmet_dependencies"))
        dependencies.extend(_dependency_names(_statement(block, "met_dependencies")))
        kernel_match = re.search(r"\.node\.kernel\s*=\s*torch\.ops\.aten\.([^\s]+)", block)
        blocks.append(
            {
                "id": match.group(1),
                "kind": match.group(2),
                "writes": writes,
                "dependencies": list(dict.fromkeys(dependencies)),
                "kernel": kernel_match.group(1) if kernel_match else "",
            }
        )
    return blocks


def _expand_scheduler_ids(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        for scheduler_id in re.findall(r"op\d+", str(value)):
            if scheduler_id not in result:
                result.append(scheduler_id)
    return result


def _load_pre_fusion_ir(trace_dir: Path) -> dict[str, Any]:
    ir_path = trace_dir / "ir_pre_fusion.txt"
    if not ir_path.is_file():
        return {"status": "UNAVAILABLE", "source": str(ir_path), "by_id": {}}
    try:
        text = ir_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return {"status": "READ_ERROR", "source": str(ir_path), "by_id": {}, "error": str(exc)}
    blocks = _ir_blocks(text)
    return {"status": "AVAILABLE", "source": str(ir_path), "by_id": {block["id"]: block for block in blocks}}


def _pre_fusion_ir_info(
    ir_index: dict[str, Any],
    post_scheduler_id: str,
    required_scheduler_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Describe the pre-fusion SchedulerNodes behind one post-fusion node."""
    if ir_index["status"] != "AVAILABLE":
        return {
            "status": ir_index["status"],
            "source": ir_index["source"],
            "post_scheduler_id": post_scheduler_id,
            "scheduler_ids": [],
            "writes": [],
            "dependencies": [],
            **({"error": ir_index["error"]} if "error" in ir_index else {}),
        }
    by_id = ir_index["by_id"]
    required = _expand_scheduler_ids(required_scheduler_ids or [post_scheduler_id])
    matched = [scheduler_id for scheduler_id in required if scheduler_id in by_id]
    missing = [scheduler_id for scheduler_id in required if scheduler_id not in by_id]
    writes = []
    dependencies = []
    for scheduler_id in matched:
        block = by_id[scheduler_id]
        for name in block["writes"]:
            if name not in writes:
                writes.append(name)
        for name in block["dependencies"]:
            if name not in dependencies:
                dependencies.append(name)
    return {
        "status": "MATCHED" if required and not missing else "PARTIAL" if matched else "NOT_FOUND",
        "source": ir_index["source"],
        "post_scheduler_id": post_scheduler_id,
        "scheduler_ids": matched,
        "missing_scheduler_ids": missing,
        "writes": writes,
        "dependencies": dependencies,
    }


def _hook_capture_map(path: Path) -> dict[str, str]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return {}
    captures = {}
    pattern = re.compile(
        r"run_intermediate_hooks\(((?:'(?:\\.|[^'])*')|(?:\"(?:\\.|[^\"])*\")),\s*([A-Za-z0-9_]+)\)"
    )
    for raw_name, buffer_name in pattern.findall(text):
        try:
            name = ast.literal_eval(raw_name)
        except (SyntaxError, ValueError):
            continue
        captures[buffer_name] = _capture_name(name)[0]
    return captures


def _fused_component_parents(blocks: list[dict[str, Any]]) -> dict[str, str]:
    parents = {}
    for block in blocks:
        if not block["kind"].startswith("FusedSchedulerNode"):
            continue
        ids = re.findall(r"op\d+", block["id"])
        if len(ids) > 1:
            parents.update({component: block["id"] for component in ids})
    return parents


def _record_output(records: dict[str, dict[str, Any]], name: str, index: int = 0) -> dict[str, Any] | None:
    for output in records.get(name, {}).get("outputs", []):
        if int(output["index"]) == index:
            return output
    return None


def _capture_base_name(name: str) -> str:
    return re.sub(r"__call\d+$", "", name)


def _record_matches_runtime(
    record: dict[str, Any], runtime_selection: dict[str, Any] | None
) -> bool:
    if runtime_selection is None:
        return True
    return (
        int(record.get("runtime_gm_id", -1)) == int(runtime_selection.get("gm_id", -2))
        and int(record.get("runtime_call_index", -1)) == int(runtime_selection.get("call_index", -2))
    )


def _fusion_capture_name(
    base: str,
    records: dict[str, dict[str, Any]],
    used: set[str],
    runtime_selection: dict[str, Any] | None = None,
) -> str:
    for name in records:
        if (
            name not in used
            and _record_matches_runtime(records[name], runtime_selection)
            and (name == base or re.fullmatch(re.escape(base) + r"__call\d+", name))
        ):
            used.add(name)
            return name
    return base


def _fallback_capture_name(
    kernel: str,
    records: dict[str, dict[str, Any]],
    used: set[str],
    runtime_selection: dict[str, Any] | None = None,
) -> str:
    base = kernel.split(".", 1)[0]
    aliases = {"mm": {"mm", "matmul"}, "bmm": {"bmm", "matmul"}}
    candidates = aliases.get(base, {base})
    for name in records:
        if name in used or name.startswith("autofused_") or not _record_matches_runtime(
            records[name], runtime_selection
        ):
            continue
        canonical = re.sub(r"__call\d+$", "", name)
        canonical = re.sub(r"_\d+$", "", canonical)
        if canonical in candidates:
            used.add(name)
            return name
    return base


def _capture_for_buffers(
    buffers: list[str],
    hook_captures: dict[str, str],
    records: dict[str, dict[str, Any]],
    used: set[str],
    runtime_selection: dict[str, Any] | None = None,
) -> str | None:
    for buffer_name in buffers:
        base = hook_captures.get(buffer_name)
        if base is None:
            continue
        for name in records:
            if (
                name in used
                or not _record_matches_runtime(records[name], runtime_selection)
                or (name != base and not re.fullmatch(re.escape(base) + r"__call\d+", name))
            ):
                continue
            used.add(name)
            return name
    return None


def _input_names(op: dict[str, Any]) -> set[str]:
    names = set()
    for value in op.get("input", []):
        source, _, _ = str(value).partition(":")
        if source:
            names.add(source)
    return names


def _cpu_graph_inputs(graph_data: dict[str, Any] | None) -> dict[str, set[str]]:
    if not graph_data:
        return {}
    graphs = graph_data.get("graph", [])
    ops = graphs[0].get("op", []) if graphs else []
    result = {}
    for op in ops:
        name = str(op.get("name", ""))
        if name:
            result[name] = _input_names(op)
    return result


def _resolve_input_origins(
    name: str,
    npu_inputs: dict[str, set[str]],
    npu_origins: dict[str, str],
) -> set[str]:
    result = set()
    stack = list(npu_inputs.get(name, set()))
    visited = set()
    while stack:
        current = stack.pop()
        if current in visited:
            continue
        visited.add(current)
        origin = npu_origins.get(current)
        if origin is not None:
            result.add(origin)
            continue
        stack.extend(npu_inputs.get(current, set()))
    return result


def _fusion_rejections(
    scheduler_id: str,
    decisions: list[dict[str, Any]],
    node_by_component: dict[str, str],
) -> list[dict[str, Any]]:
    selected = set(re.findall(r"op\d+", scheduler_id))
    aggregated: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for decision in decisions:
        if decision.get("result") is not False:
            continue
        left = set(decision.get("node1", []))
        right = set(decision.get("node2", []))
        if selected & left:
            peer_ids = right
            direction = "output" if decision.get("kind", "").startswith("vertical") else "sibling"
        elif selected & right:
            peer_ids = left
            direction = "input" if decision.get("kind", "").startswith("vertical") else "sibling"
        else:
            continue
        peers = sorted({node_by_component.get(item, item) for item in peer_ids})
        peers = [peer for peer in peers if peer not in {node_by_component.get(item, item) for item in selected}]
        if not peers:
            continue
        kind = str(decision.get("kind", "unknown"))
        reason = str(decision.get("reason", "unknown"))
        peer = ", ".join(peers)
        key = (kind, direction, peer, reason)
        if key not in aggregated:
            aggregated[key] = {
                "category": "fusion_rejection",
                "kind": kind,
                "direction": direction,
                "peer": peer,
                "reason": reason,
                "attempts": 0,
            }
        aggregated[key]["attempts"] += 1
    return list(aggregated.values())


def _attach_lowering_fallbacks(ops: list[dict[str, Any]], diagnostics: list[dict[str, Any]]) -> None:
    used: set[int] = set()
    op_names = {op["name"] for op in ops}
    reserved = {index for index, item in enumerate(diagnostics) if item.get("node") in op_names}

    def reason(item: dict[str, Any], display_name: str) -> dict[str, Any]:
        result = {
            "category": item.get("category", "lowering_fallback"),
            "target": item.get("target", ""),
            "reason": item.get("reason", "unknown"),
        }
        if item.get("node") and item.get("node") != display_name:
            result["source_node"] = item["node"]
        return result

    for op in ops:
        exact = [
            (index, item)
            for index, item in enumerate(diagnostics)
            if index not in used and item.get("node") == op["name"]
        ]
        if exact:
            op["unfused_reasons"] = [reason(item, op["name"]) for _, item in exact]
            used.update(index for index, _ in exact)

    for op in ops:
        if op.get("unfused_reasons") or not op.get("_fallback_kernel"):
            continue
        target = op.get("_kernel_target", "")
        candidate = next(
            (
                (index, item)
                for index, item in enumerate(diagnostics)
                if index not in used and index not in reserved and item.get("target") == target
            ),
            None,
        )
        if candidate is None:
            continue
        _, first = candidate
        source_node = first.get("node")
        aliases = [
            (index, item)
            for index, item in enumerate(diagnostics)
            if index not in used
            and index not in reserved
            and item.get("target") == target
            and item.get("node") == source_node
        ]
        op["unfused_reasons"] = [reason(item, op["name"]) for _, item in aliases]
        used.update(index for index, _ in aliases)


def _trace_graph(
    trace_dir: Path,
    records: dict[str, dict[str, Any]],
    used_captures: set[str],
    fusion_decisions: list[dict[str, Any]],
    lowering_fallbacks: list[dict[str, Any]],
    runtime_selection: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ir_path = trace_dir / "ir_post_fusion.txt"
    text = ir_path.read_text(encoding="utf-8")
    layouts = _layout_map(text)
    fx_layouts = _fx_layout_map(trace_dir / "fx_graph_readable.py")
    fx_input_layouts = _fx_input_layout_map(trace_dir / "fx_graph_readable.py", fx_layouts)
    for name, layout in fx_layouts.items():
        layouts.setdefault(name, layout)
    blocks = _ir_blocks(text)
    pre_fusion_ir = _load_pre_fusion_ir(trace_dir)
    fused_component_parents = _fused_component_parents(blocks)
    fused_components = set(fused_component_parents)
    hook_captures = _hook_capture_map(trace_dir / "output_code.py")
    fusion_by_scheduler: dict[str, dict[str, Any]] = {}
    fusion_by_kernel: dict[str, dict[str, Any]] = {}
    for benchmark in sorted(trace_dir.glob("autofused_*/benchmark.py")):
        metadata = _benchmark_metadata(benchmark)
        if metadata is None:
            continue
        fusion_by_kernel[metadata["kernel_name"]] = metadata
        for scheduler_id in metadata["scheduler_ids"]:
            fusion_by_scheduler[scheduler_id] = metadata
    for block in blocks:
        if not block["kind"].startswith("FusedSchedulerNode"):
            continue
        component_metadata = [
            fusion_by_scheduler.get(item) for item in re.findall(r"op\d+", block["id"])
        ]
        component_metadata = [item for item in component_metadata if item is not None]
        if component_metadata and all(
            item["kernel_name"] == component_metadata[0]["kernel_name"] for item in component_metadata
        ):
            fusion_by_scheduler[block["id"]] = component_metadata[0]

    aliases: dict[str, str] = {}
    producer: dict[str, str] = {}
    for block in blocks:
        if "MultiOutput" in block["kind"] and block["writes"] and block["dependencies"]:
            aliases[block["writes"][0]] = block["dependencies"][0]
        elif "NopKernelSchedulerNode" not in block["kind"]:
            for name in block["writes"]:
                producer[name] = fused_component_parents.get(block["id"], block["id"])

    def resolve_buffer(name: str) -> str:
        seen = set()
        while name in aliases and name not in seen:
            seen.add(name)
            name = aliases[name]
        return name

    node_by_scheduler: dict[str, str] = {}
    used_node_names: set[str] = set()
    ops: list[dict[str, Any]] = []
    block_by_id = {block["id"]: block for block in blocks}
    for block in blocks:
        if block["id"] in fused_components:
            continue
        if "MultiOutput" in block["kind"] or "NopKernelSchedulerNode" in block["kind"]:
            continue
        fusion = fusion_by_scheduler.get(block["id"])
        captured_name = _capture_for_buffers(
            block["writes"], hook_captures, records, used_captures, runtime_selection
        )
        if fusion is None and captured_name is not None:
            fusion = next(
                (
                    metadata
                    for kernel_name, metadata in fusion_by_kernel.items()
                    if captured_name.startswith(kernel_name + "__")
                ),
                None,
            )
        if fusion is not None:
            name = captured_name
            if name is None:
                name = _fusion_capture_name(
                    fusion["capture_name"], records, used_captures, runtime_selection
                )
            node_type = "AutofuseFusion"
            origin_ops = fusion["origin_ops"]
            original_aten_types = fusion["original_aten_types"]
            origin_subgraph = fusion["origin_subgraph"]
            prefix = fusion["kernel_name"] + "__"
            exit_name = (
                _capture_base_name(name[len(prefix):])
                if name.startswith(prefix)
                else fusion["exit_name"]
            )
            exit_type = exit_name
        else:
            name = captured_name
            if name is None:
                name = _fallback_capture_name(
                    block["kernel"], records, used_captures, runtime_selection
                )
            node_type = f"aten.{block['kernel'].split('.', 1)[0]}" if block["kernel"] else block["kind"]
            origin_ops = [_capture_base_name(name)]
            original_aten_types = []
            origin_subgraph = None
            exit_name = _capture_base_name(name)
            exit_type = block["kernel"].split(".", 1)[0] if block["kernel"] else name
        if not name:
            name = f"inductor_{block['id']}"
        if name in used_node_names:
            base_name = name
            name = f"{base_name}__{block['id']}"
            suffix = 1
            while name in used_node_names:
                name = f"{base_name}__{block['id']}_{suffix}"
                suffix += 1
        used_node_names.add(name)
        if not exit_name:
            exit_name = name
        if not exit_type:
            exit_type = node_type
        if not any(origin_ops):
            origin_ops = [name]
        node_by_scheduler[block["id"]] = name
        ir_pre_fusion = _pre_fusion_ir_info(
            pre_fusion_ir,
            block["id"],
            fusion.get("scheduler_ids") if fusion is not None else None,
        )
        outputs = []
        captured_outputs = records.get(name, {}).get("outputs", []) if fusion is not None else []
        if captured_outputs:
            for captured in sorted(captured_outputs, key=lambda item: int(item["index"])):
                outputs.append(_desc(captured["dtype"], tuple(captured["shape"])))
        else:
            for output_index, buffer_name in enumerate(block["writes"] or [""]):
                captured = _record_output(records, name, output_index)
                layout = layouts.get(buffer_name, {})
                dtype = captured["dtype"] if captured else layout.get("dtype", "unknown")
                shape = captured["shape"] if captured else layout.get("shape", [])
                outputs.append(_desc(dtype, tuple(shape)))
        op = {
            "name": name,
            "type": node_type,
            "input": [],
            "output_desc": outputs,
            "origin_ops": origin_ops,
            "original_aten_types": original_aten_types,
            "_exit_name": exit_name,
            "_exit_type": exit_type,
            "_dependencies": block["dependencies"],
            "_scheduler_id": block["id"],
            "_fallback_kernel": "FallbackKernel" in block["kind"],
            "_kernel_target": f"aten.{block['kernel']}" if block["kernel"] else "",
            "ir_pre_fusion": ir_pre_fusion,
        }
        if name in fx_input_layouts:
            op["input_desc"] = [
                _desc(item.get("dtype", "unknown"), tuple(item.get("shape", [])))
                for item in fx_input_layouts[name]
            ]
        if origin_subgraph is not None:
            op["origin_subgraph"] = origin_subgraph
        ops.append(op)

    _attach_lowering_fallbacks(ops, lowering_fallbacks)
    node_by_component = {
        component: op["name"]
        for op in ops
        for component in re.findall(r"op\d+", op["_scheduler_id"])
    }
    for op in ops:
        if op["type"] != "AutofuseFusion":
            rejections = _fusion_rejections(op["_scheduler_id"], fusion_decisions, node_by_component)
            if rejections:
                op.setdefault("unfused_reasons", []).extend(rejections)
        else:
            # Fused kernels participate in further-fusion attempts as a whole;
            # their rejection records explain why two fusion groups did not merge.
            rejections = _fusion_rejections(op["_scheduler_id"], fusion_decisions, node_by_component)
            if rejections:
                op["fusion_rejections"] = rejections
        op.pop("_scheduler_id", None)
        op.pop("_fallback_kernel", None)
        op.pop("_kernel_target", None)

    data_names = []
    for op in ops:
        for dependency in op.pop("_dependencies"):
            resolved = resolve_buffer(dependency)
            source_scheduler = producer.get(resolved)
            source_name = node_by_scheduler.get(source_scheduler, "") if source_scheduler else ""
            if source_name:
                op["input"].append(f"{source_name}:0")
                continue
            data_name = f"Data.{resolved}"
            op["input"].append(f"{data_name}:0")
            if resolved not in data_names:
                data_names.append(resolved)

    data_ops = []
    for name in data_names:
        layout = layouts.get(name, {})
        data_ops.append(
            {
                "name": f"Data.{name}",
                "type": "Data",
                "output_desc": [_desc(layout.get("dtype", "unknown"), tuple(layout.get("shape", [])))],
            }
        )
    return {"ops": [*data_ops, *ops], "source": str(ir_path)}


def _trace_graph_id(path: Path) -> int | None:
    match = re.search(r"model__(\d+)(?:_|$)", path.name)
    return int(match.group(1)) if match is not None else None


def _trace_sort_key(path: Path) -> tuple[int, str]:
    """Keep post-fusion traces aligned with Inductor's numeric graph order."""
    graph_id = _trace_graph_id(path)
    if graph_id is not None:
        return graph_id, path.as_posix()
    return sys.maxsize, path.as_posix()


def _canonical_match(value: str) -> str:
    text = _canonical_type(re.sub(r"_\d+$", "", value))
    aliases = {
        "mm": "matmul",
        "bmm": "matmul",
        "addmm": "linear",
        "baddbmm": "add",
        "masked_fill": "where",
    }
    return aliases.get(text, text)


def _output_shape(info: dict[str, Any]) -> tuple[int, ...] | None:
    raw_shape = info.get("shape")
    if raw_shape is None:
        return None
    return tuple(int(dim) for dim in raw_shape)


def _description_shape(desc: dict[str, Any]) -> tuple[int, ...] | None:
    raw_shape = desc.get("shape")
    if isinstance(raw_shape, dict):
        raw_shape = raw_shape.get("dim")
    if not isinstance(raw_shape, (list, tuple)):
        return None
    return tuple(int(dim) for dim in raw_shape)


def _compatible_output_shape(left: tuple[int, ...] | None, right: tuple[int, ...] | None) -> bool:
    if left is None or right is None or left == right:
        return True
    return {left, right} == {(), (1,)}


def _upstream_distance(
    candidate: str,
    input_origins: set[str],
    cpu_inputs: dict[str, set[str]],
) -> int | None:
    if not input_origins:
        return None
    frontier = list(cpu_inputs.get(candidate, set()))
    visited = set()
    distance = 1
    while frontier:
        next_frontier = []
        for current in frontier:
            if current in input_origins:
                return distance
            if current in visited:
                continue
            visited.add(current)
            next_frontier.extend(cpu_inputs.get(current, set()))
        frontier = next_frontier
        distance += 1
    return None


def _match_cpu_output(
    op: dict[str, Any],
    records: dict[str, dict[str, Any]],
    cpu_outputs: dict[str, dict[str, Any]],
    used: set[str],
    diagnostics: dict[str, Any] | None = None,
    cpu_inputs: dict[str, set[str]] | None = None,
    input_origins: set[str] | None = None,
) -> str | None:
    if diagnostics is not None:
        diagnostics.clear()
    actual_info = _record_output(records, op["name"], 0)
    if actual_info is None:
        return None
    expected_name = op.get("_exit_name", op["name"])
    expected_type = _canonical_match(op.get("_exit_type", expected_name))
    suffix_match = re.search(r"(_\d+)$", expected_name)
    instance_suffix = suffix_match.group(1) if suffix_match else ""
    original_types = [str(value) for value in op.get("original_aten_types", []) if value]
    original_canonical = {_canonical_match(value) for value in original_types}
    original_names = {f"{value}{instance_suffix}" for value in original_types}
    candidates = []
    for order, (name, info) in enumerate(cpu_outputs.items()):
        if name in used or int(info["size"]) != int(actual_info["size"]):
            continue
        exact_name = name == expected_name
        name_match = _canonical_match(name) == _canonical_match(expected_name)
        type_match = _canonical_match(str(info.get("type", ""))) == expected_type
        original_name_match = name in original_names
        original_type_match = _canonical_match(str(info.get("type", name))) in original_canonical
        if exact_name or name_match or type_match or original_name_match or original_type_match:
            candidates.append(
                (name, info, order, exact_name, name_match, type_match, original_name_match, original_type_match)
            )
    if not candidates:
        return None
    actual_shape = _output_shape(actual_info)
    has_fusion_provenance = bool(original_types)
    input_origins = input_origins or set()
    ranked_candidates = []
    for name, info, order, exact_name, name_match, type_match, original_name_match, original_type_match in candidates:
        candidate_shape = _output_shape(info)
        compatible_shape = _compatible_output_shape(candidate_shape, actual_shape)
        exact_shape = candidate_shape == actual_shape
        upstream_distance = _upstream_distance(name, input_origins, cpu_inputs or {})
        ranked_candidates.append(
            (
                name,
                order,
                exact_name,
                name_match,
                type_match,
                original_name_match,
                original_type_match,
                candidate_shape,
                compatible_shape,
                exact_shape,
                upstream_distance,
            )
        )

    has_topology_candidate = any(item[8] and item[10] is not None for item in ranked_candidates)
    if input_origins and cpu_inputs and not has_fusion_provenance and not has_topology_candidate:
        if diagnostics is not None:
            exact = next((item for item in ranked_candidates if item[2]), ranked_candidates[0])
            diagnostics.update(
                {
                    "output_index": 0,
                    "candidate": exact[0],
                    "candidate_index": 0,
                    "reference_shape": list(exact[7]) if exact[7] is not None else None,
                    "npu_shape": list(actual_shape) if actual_shape is not None else None,
                    "reason": (
                        "NPU 节点已有上游 CPU origin，但候选 CPU 节点均无法从该 origin 拓扑到达；"
                        "该节点可能由编译器插入，未建立映射"
                    ),
                }
            )
        return None
    best = None
    for (
        name,
        order,
        exact_name,
        name_match,
        type_match,
        original_name_match,
        original_type_match,
        candidate_shape,
        compatible_shape,
        exact_shape,
        upstream_distance,
    ) in ranked_candidates:
        if has_fusion_provenance:
            source_exit_match = exact_name and expected_name in op.get("origin_ops", [])
            score = (
                0 if compatible_shape else 1,
                0 if exact_shape else 1,
                0 if upstream_distance is not None else 1,
                0 if source_exit_match else 1,
                upstream_distance if upstream_distance is not None else sys.maxsize,
                0 if original_name_match else 1,
                0 if original_type_match else 1,
                0 if exact_name else 1,
                0 if name_match else 1,
                0 if type_match else 1,
                order,
            )
        elif has_topology_candidate:
            score = (
                0 if compatible_shape else 1,
                0 if upstream_distance is not None else 1,
                upstream_distance if upstream_distance is not None else sys.maxsize,
                0 if exact_shape else 1,
                0 if exact_name else 1,
                0 if name_match else 1,
                0 if type_match else 1,
                order,
            )
        else:
            # A non-fused output has no compiler-level semantic alias. Keep an
            # exact node name ahead of suffix-normalized candidates when no
            # compatible topology evidence exists.
            score = (
                0 if exact_name else 1,
                0 if exact_shape else 1,
                0 if compatible_shape else 1,
                0 if name_match else 1,
                0 if type_match else 1,
                order,
            )
        if best is None or score < best[0]:
            best = (score, name, candidate_shape, compatible_shape)
    if best is None:
        return None
    if not best[3]:
        if diagnostics is not None:
            diagnostics.update(
                {
                    "output_index": 0,
                    "candidate": best[1],
                    "candidate_index": 0,
                    "reference_shape": list(best[2]) if best[2] is not None else None,
                    "npu_shape": list(actual_shape) if actual_shape is not None else None,
                    "reason": (
                        "provenance 候选的完整 shape 与 NPU 输出不兼容；"
                        "为避免按元素个数展平后误比较，未建立映射"
                    ),
                }
            )
        return None
    used.add(best[1])
    return best[1]


def build_inductor_autofuse_graph(
    compile_root: str | Path,
    records: dict[str, dict[str, Any]],
    cpu_outputs: dict[str, dict[str, Any]],
    diagnostics: dict[str, Any] | None = None,
    cpu_graph_data: dict[str, Any] | None = None,
    prefusion_graph_data: dict[str, Any] | None = None,
    prefusion_runtime_selection: dict[int, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    root = Path(compile_root).resolve()
    traces = sorted({path.parent for path in root.rglob("ir_post_fusion.txt")}, key=_trace_sort_key)
    if not traces:
        raise RuntimeError(f"TorchInductor post-fusion IR was not generated under {root}")
    all_ops = []
    sources = []
    used_captures: set[str] = set()
    diagnostics = diagnostics or {}
    decision_groups = diagnostics.get("fusion_decision_groups", [])
    decision_offset = max(0, len(decision_groups) - len(traces))
    lowering_fallbacks = diagnostics.get("lowering_fallbacks", [])
    for trace_position, trace in enumerate(traces):
        graph_index = _trace_graph_id(trace)
        if graph_index is None:
            graph_index = trace_position
        decision_index = decision_offset + trace_position
        decisions = decision_groups[decision_index] if decision_index < len(decision_groups) else []
        parsed = _trace_graph(
            trace,
            records,
            used_captures,
            decisions,
            lowering_fallbacks,
            (prefusion_runtime_selection or {}).get(graph_index),
        )
        sources.append(parsed["source"])
        existing_names = {item["name"] for item in all_ops}
        rename = {
            op["name"]: f"graph{graph_index}.{op['name']}"
            for op in parsed["ops"]
            if op["name"] in existing_names
        }
        for op in parsed["ops"]:
            op["name"] = rename.get(op["name"], op["name"])
            op["_prefusion_graph_index"] = graph_index
            rewritten_inputs = []
            for value in op.get("input", []):
                source, separator, suffix = value.partition(":")
                rewritten_inputs.append(f"{rename.get(source, source)}{separator}{suffix}")
            op["input"] = rewritten_inputs
            all_ops.append(op)

    prefusion_lookup: dict[tuple[int, str], tuple[str, str]] = {}
    prefusion_ops: dict[str, dict[str, Any]] = {}
    if prefusion_graph_data:
        for op in prefusion_graph_data.get("graph", [{}])[0].get("op", []):
            canonical_id = op.get("canonical_id")
            if not canonical_id:
                continue
            prefusion_ops[str(canonical_id)] = op
            match = re.match(r"graph(\d+)::(.+)$", str(canonical_id))
            if match:
                prefusion_lookup[(int(match.group(1)), match.group(2))] = (
                    str(canonical_id),
                    str(op.get("name", match.group(2))),
                )

    def resolve_prefusion_name(graph_index: int, name: str) -> tuple[str, str] | None:
        direct = prefusion_lookup.get((graph_index, name))
        if direct is not None:
            return direct
        suffix = re.sub(r"^graph\d+\.", "", name)
        return prefusion_lookup.get((graph_index, suffix))

    for op in all_ops:
        graph_index = int(op.get("_prefusion_graph_index", 0))
        origin_node_ids = []
        origin_names = []
        for source in op.get("origin_ops", []):
            resolved = resolve_prefusion_name(graph_index, str(source))
            if resolved is None:
                continue
            canonical_id, display_name = resolved
            origin_node_ids.append(canonical_id)
            origin_names.append(display_name)
        if origin_node_ids:
            op["origin_node_ids"] = list(dict.fromkeys(origin_node_ids))
            exit_name = str(op.get("_exit_name", ""))
            exit_resolved = resolve_prefusion_name(graph_index, exit_name)
            if exit_resolved is not None:
                op["canonical_id"] = exit_resolved[0]
                op["_prefusion_origin_name"] = exit_resolved[1]
            elif len(origin_node_ids) == 1:
                op["canonical_id"] = origin_node_ids[0]
                op["_prefusion_origin_name"] = origin_names[0]
        op.pop("_prefusion_graph_index", None)

    used_cpu: set[str] = set()
    cpu_inputs = _cpu_graph_inputs(cpu_graph_data)
    npu_inputs = {op["name"]: _input_names(op) for op in all_ops}
    npu_origins: dict[str, str] = {}
    for op in all_ops:
        if op["type"] == "Data":
            continue
        mapping_diagnostic: dict[str, Any] = {}
        canonical_origin = op.get("_prefusion_origin_name")
        if canonical_origin is not None and op.get("canonical_id") is not None:
            canonical_id = str(op["canonical_id"])
            reference_op = prefusion_ops.get(canonical_id)
            reference_shape = None
            if reference_op is not None:
                reference_outputs = reference_op.get("output_desc", [])
                if reference_outputs:
                    reference_shape = _description_shape(reference_outputs[0])
            npu_shape = _description_shape(op.get("output_desc", [{}])[0]) if op.get("output_desc") else None
            if reference_shape is not None and npu_shape is not None and not _compatible_output_shape(
                reference_shape, npu_shape
            ):
                reason = (
                    "IR/pre-fusion canonical 出口 shape 冲突："
                    f"reference={list(reference_shape)}, NPU={list(npu_shape)}；"
                    "未建立映射，避免把不同 graph variant 或不同出口误认为同一算子"
                )
                op["provenance_conflicts"] = [
                    {
                        "output_index": 0,
                        "candidate": canonical_id,
                        "candidate_index": 0,
                        "reference_shape": list(reference_shape),
                        "npu_shape": list(npu_shape),
                        "reason": reason,
                    }
                ]
                op["mapping_method"] = "provenance_conflict"
                op.pop("canonical_id", None)
                op.pop("_prefusion_origin_name", None)
                continue
            ir_status = op.get("ir_pre_fusion", {}).get("status")
            mapping_method = (
                "ir_pre_fusion_canonical_id" if ir_status == "MATCHED" else "canonical_source_id_ir_unverified"
            )
            op["mapping_method"] = mapping_method
            if op.get("output_desc"):
                attrs = [
                    {"key": "_autofusex_prefusion_origin_id", "value": {"s": op["canonical_id"]}},
                    {"key": "_autofusex_cpu_origin_name", "value": {"s": canonical_origin}},
                    {"key": "_autofusex_cpu_origin_output_index", "value": {"i": 0}},
                    {"key": "_autofusex_mapping_method", "value": {"s": mapping_method}},
                ]
                ir_scheduler_ids = op.get("ir_pre_fusion", {}).get("scheduler_ids", [])
                if ir_scheduler_ids:
                    attrs.append(
                        {
                            "key": "_autofusex_prefusion_ir_scheduler_ids",
                            "value": {"s": ",".join(ir_scheduler_ids)},
                        }
                    )
                op["output_desc"][0].setdefault("attr", []).extend(attrs)
            npu_origins[op["name"]] = canonical_origin
            op.pop("_exit_name", None)
            op.pop("_exit_type", None)
            continue
        if op.get("mapping_method") == "provenance_conflict":
            op.pop("_exit_name", None)
            op.pop("_exit_type", None)
            continue
        if prefusion_graph_data:
            op["provenance_conflicts"] = [
                {
                    "output_index": 0,
                    "candidate": op.get("name", ""),
                    "candidate_index": 0,
                    "reason": "Inductor post-fusion 节点没有对应的 pre-fusion canonical source，视为编译器插入节点",
                }
            ]
            op.pop("_exit_name", None)
            op.pop("_exit_type", None)
            continue
        input_origins = _resolve_input_origins(op["name"], npu_inputs, npu_origins)
        origin = _match_cpu_output(
            op,
            records,
            cpu_outputs,
            used_cpu,
            mapping_diagnostic,
            cpu_inputs,
            input_origins,
        )
        if origin is not None and op.get("output_desc"):
            op["output_desc"][0].setdefault("attr", []).extend(
                [
                    {"key": "_autofusex_cpu_origin_name", "value": {"s": origin}},
                    {"key": "_autofusex_cpu_origin_output_index", "value": {"i": 0}},
                ]
            )
            npu_origins[op["name"]] = origin
        elif mapping_diagnostic:
            op["provenance_conflicts"] = [mapping_diagnostic]
        op.pop("_exit_name", None)
        op.pop("_exit_type", None)
    return {
        "graph": [{"op": all_ops}],
        "metadata": {
            "topology": "TorchInductor post-fusion / inductor_npu_ext Autofuse",
            "inductor_traces": sources,
            "pre_fusion_ir": sorted(
                {
                    str(op.get("ir_pre_fusion", {}).get("source"))
                    for op in all_ops
                    if op.get("ir_pre_fusion", {}).get("source")
                    and op.get("ir_pre_fusion", {}).get("status") != "UNAVAILABLE"
                }
            ),
            "mapping_policy": "pre-fusion IR SchedulerNode -> transformed FX canonical_id -> CPU/NPU eager",
            "runtime_capture_selection": prefusion_runtime_selection or {},
            "lowering_fallbacks": len(diagnostics.get("lowering_fallbacks", [])),
            "fusion_decisions": sum(len(group) for group in decision_groups),
        },
    }

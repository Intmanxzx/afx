from __future__ import annotations

import json
import importlib.util
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from autofusex.autofuse_graph import attach_autofuse_graphs, enable_autofuse_graph_dump
from autofusex.compare import compare_manifests, format_summary, save_report
from autofusex.graph import build_manifest
from autofusex.html import write_report_html


@dataclass(frozen=True)
class ExampleSpec:
    key: str
    script_name: str
    input_names: tuple[str, ...]
    input_shape: tuple[int, ...]
    cpu_nodes: tuple[dict[str, Any], ...]
    output_origin: str
    output_dtype: str
    output_shape: tuple[int, ...]
    npu_type: str = "AscBackend"
    input_shapes: tuple[tuple[int, ...], ...] | None = None


@dataclass(frozen=True)
class _GeGraphSnapshot:
    source: str
    model: Any


def _desc(dtype: str, shape: tuple[int, ...], attrs: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    item: dict[str, Any] = {"layout": "ND", "data_type": dtype, "shape": {"dim": list(shape)}}
    if attrs:
        item["attr"] = attrs
    return item


EXAMPLES: dict[str, ExampleSpec] = {
    "af_add_ge.py": ExampleSpec(
        key="af_add_ge",
        script_name="af_add_ge.py",
        input_names=("x", "y", "z"),
        input_shape=(128, 50),
        cpu_nodes=(
            {
                "name": "Add_1",
                "type": "Add",
                "input": ["x:0", "y:0"],
                "dtype": "float32",
                "shape": (128, 50),
            },
            {
                "name": "Ge_2",
                "type": "Ge",
                "input": ["Add_1:0", "z:0"],
                "dtype": "bool",
                "shape": (128, 50),
            },
        ),
        output_origin="Ge_2",
        output_dtype="bool",
        output_shape=(128, 50),
    ),
    "af_mul_reducesum.py": ExampleSpec(
        key="af_mul_reducesum",
        script_name="af_mul_reducesum.py",
        input_names=("x", "y"),
        input_shape=(256, 100),
        cpu_nodes=(
            {
                "name": "Mul_1",
                "type": "Mul",
                "input": ["x:0", "y:0"],
                "dtype": "float32",
                "shape": (256, 100),
            },
            {
                "name": "ReduceSum_2",
                "type": "ReduceSum",
                "input": ["Mul_1:0"],
                "dtype": "float32",
                "shape": (1,),
            },
        ),
        output_origin="ReduceSum_2",
        output_dtype="float32",
        output_shape=(1,),
    ),
    "af_broadcast_add_reducesum.py": ExampleSpec(
        key="af_broadcast_add_reducesum",
        script_name="af_broadcast_add_reducesum.py",
        input_names=("x", "bias"),
        input_shape=(32, 64),
        input_shapes=((32, 64), (64,)),
        cpu_nodes=(
            {
                "name": "Add_1",
                "type": "Add",
                "input": ["x:0", "bias:0"],
                "dtype": "float32",
                "shape": (32, 64),
            },
            {
                "name": "ReduceSum_2",
                "type": "ReduceSum",
                "input": ["Add_1:0"],
                "dtype": "float32",
                "shape": (1,),
            },
        ),
        output_origin="ReduceSum_2",
        output_dtype="float32",
        output_shape=(1,),
    ),
    "af_qwen.py": ExampleSpec(
        key="af_qwen",
        script_name="af_qwen.py",
        input_names=("input_ids",),
        input_shape=(2, 16),
        cpu_nodes=(
            {
                "name": "ModelOutput",
                "type": "MiniQwen",
                "input": ["input_ids:0"],
                "dtype": "float32",
                "shape": (2, 16, 256),
            },
        ),
        output_origin="ModelOutput",
        output_dtype="float32",
        output_shape=(2, 16, 256),
        npu_type="CompiledMiniQwen",
    ),
}


def example_from_script(script: str | Path) -> ExampleSpec:
    path = Path(script)
    try:
        return EXAMPLES[path.name]
    except KeyError as exc:
        supported = ", ".join(sorted(EXAMPLES))
        raise ValueError(f"unsupported example script {path.name!r}; supported: {supported}") from exc


def _clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _cpu_graph(spec: ExampleSpec) -> dict[str, Any]:
    ops: list[dict[str, Any]] = [
        {"name": name, "type": "Data", "output_desc": [_desc("float32", shape)]}
        for name, shape in zip(spec.input_names, _input_shapes(spec), strict=True)
    ]
    for node in spec.cpu_nodes:
        ops.append(
            {
                "name": node["name"],
                "type": node["type"],
                "input": node["input"],
                "output_desc": [_desc(node["dtype"], tuple(node["shape"]))],
            }
        )
    return {"graph": [{"op": ops}]}


def _match_broadcast_add_reducesum_dumps(converted: list[Path]) -> dict[str, str]:
    matches: dict[str, str] = {}
    for path in converted:
        name = op_name_from_npy(path, _npy_output_index(path))
        if name.startswith("Add."):
            matches[name] = "Add_1"
        elif name.startswith("ReduceSum"):
            matches[name] = "ReduceSum_2"
    return matches


def _save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_script_module(script: str | Path) -> Any:
    path = Path(script).resolve()
    module_name = f"autofusex_user_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _input_tensors(spec: ExampleSpec, *, torch: Any) -> list[Any]:
    shapes = spec.input_shapes or tuple(spec.input_shape for _ in spec.input_names)
    return [torch.randn(*shape) for shape in shapes]


def _input_shapes(spec: ExampleSpec) -> tuple[tuple[int, ...], ...]:
    return spec.input_shapes or tuple(spec.input_shape for _ in spec.input_names)


def _cpu_outputs(spec: ExampleSpec, inputs: list[Any], *, torch: Any) -> dict[str, Any]:
    if spec.key == "af_add_ge":
        x, y, z = inputs
        add = torch.add(x, y)
        return {"Add_1": add, "Ge_2": torch.ge(add, z)}
    if spec.key == "af_mul_reducesum":
        x, y = inputs
        mul = torch.mul(x, y)
        return {"Mul_1": mul, "ReduceSum_2": torch.sum(mul)}
    if spec.key == "af_broadcast_add_reducesum":
        x, bias = inputs
        add = torch.add(x, bias)
        return {"Add_1": add, "ReduceSum_2": torch.sum(add)}
    raise ValueError(f"unsupported example: {spec.key}")


def _model_factory(spec: ExampleSpec) -> Any:
    import torch

    class AddGeModel(torch.nn.Module):
        def forward(self, x, y, z):
            return torch.ge(torch.add(x, y), z)

    class MulReduceSumModel(torch.nn.Module):
        def forward(self, x, y):
            return torch.sum(torch.mul(x, y))

    class BroadcastAddReduceSumModel(torch.nn.Module):
        def forward(self, x, bias):
            return torch.sum(torch.add(x, bias))

    if spec.key == "af_add_ge":
        return AddGeModel()
    if spec.key == "af_mul_reducesum":
        return MulReduceSumModel()
    if spec.key == "af_broadcast_add_reducesum":
        return BroadcastAddReduceSumModel()
    raise ValueError(f"unsupported example: {spec.key}")


def _candidate_cann_roots(raw: str | Path | None) -> list[Path]:
    roots = []
    if raw:
        roots.append(Path(raw))
    for name in ("ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME"):
        value = os.environ.get(name)
        if value:
            roots.append(Path(value))
    roots.extend(
        [
            Path("/home/developer/Ascend/cann-9.1.0"),
            Path("/usr/local/Ascend/ascend-toolkit/latest"),
            Path("/usr/local/Ascend/ascend-toolkit"),
        ]
    )
    unique = []
    seen = set()
    for root in roots:
        resolved = root.expanduser()
        if str(resolved) not in seen:
            unique.append(resolved)
            seen.add(str(resolved))
    return unique


def find_msaccucmp(cann_root: str | Path | None = None) -> Path:
    for root in _candidate_cann_roots(cann_root):
        path = root / "tools/operator_cmp/compare/msaccucmp.py"
        if path.is_file():
            return path
    raise FileNotFoundError("msaccucmp.py not found; pass --cann-root or source CANN set_env.sh")


def _raw_npu_dump_files(raw_dump_dir: Path) -> list[Path]:
    files = []
    for path in raw_dump_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix == ".npy" or path.name.startswith("Data."):
            continue
        files.append(path)
    return sorted(files)


def convert_cann_dump_to_npy(raw_dump_dir: Path, npy_dump_dir: Path, msaccucmp: Path) -> list[Path]:
    npy_dump_dir.mkdir(parents=True, exist_ok=True)
    converted: list[Path] = []
    for raw in _raw_npu_dump_files(raw_dump_dir):
        cmd = [
            sys.executable,
            str(msaccucmp),
            "convert",
            "-d",
            str(raw),
            "-out",
            str(npy_dump_dir),
            "-t",
            "npy",
            "-v",
            "2",
        ]
        subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True)
    converted = sorted(npy_dump_dir.glob("*.output.*.npy"))
    if not converted:
        raise FileNotFoundError(f"no converted npu npy files under {npy_dump_dir}")
    return converted


def _cann_dump_sort_key(path: Path) -> tuple[int, int, int, str]:
    parts = path.name.split(".")
    try:
        output_pos = parts.index("output")
    except ValueError:
        return (10**9, 10**9, 10**18, path.name)
    task_parts = parts[max(0, output_pos - 3) : output_pos]
    if len(task_parts) == 3 and all(part.isdigit() for part in task_parts):
        return (int(task_parts[0]), int(task_parts[1]), int(task_parts[2]), path.name)
    return (10**9, 10**9, 10**18, path.name)


def convert_cann_dump_tree_to_npy(raw_dump_dir: Path, npy_dump_dir: Path, msaccucmp: Path) -> list[Path]:
    npy_dump_dir.mkdir(parents=True, exist_ok=True)
    dump_dirs = sorted({path.parent for path in _raw_npu_dump_files(raw_dump_dir)})
    if not dump_dirs:
        raise FileNotFoundError(f"no raw npu dump files under {raw_dump_dir}")
    for dump_dir in dump_dirs:
        cmd = [
            sys.executable,
            str(msaccucmp),
            "convert",
            "-d",
            str(dump_dir),
            "-out",
            str(npy_dump_dir),
            "-t",
            "npy",
            "-v",
            "2",
        ]
        subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True)
    converted = [
        path
        for path in npy_dump_dir.glob("*.output.*.npy")
        if _npy_output_index(path) == 0 and "." in op_name_from_npy(path, _npy_output_index(path))
    ]
    converted = sorted(converted, key=_cann_dump_sort_key)
    if not converted:
        raise FileNotFoundError(f"no converted npu npy files under {npy_dump_dir}")
    return converted


def op_name_from_npy(path: str | Path, output_index: int = 0) -> str:
    name = Path(path).name
    suffix = f".output.{output_index}.npy"
    if name.endswith(suffix):
        name = name[: -len(suffix)]
    parts = name.split(".")
    if len(parts) > 3 and all(part.isdigit() for part in parts[-3:]):
        return ".".join(parts[:-3])
    return name


def _npy_output_index(path: str | Path) -> int:
    parts = Path(path).name.split(".")
    for idx, part in enumerate(parts):
        if part == "output" and idx + 1 < len(parts) and parts[idx + 1].isdigit():
            return int(parts[idx + 1])
    return 0


def _npu_node_type(name: str) -> str:
    return name.split(".", 1)[0] if "." in name else name.split("_", 1)[0]


def _runtime_node_name(op: Any) -> str:
    name = str(op.name)
    op_type = str(op.type)
    return name if name.startswith(f"{op_type}.") else f"{op_type}.{name}"


def _protobuf_attr(attrs: Any, key: str) -> Any | None:
    try:
        return attrs.get(key)
    except (AttributeError, TypeError):
        return attrs[key] if key in attrs else None


def _protobuf_string(attrs: Any, key: str) -> str | None:
    value = _protobuf_attr(attrs, key)
    text = getattr(value, "s", None) if value is not None else None
    if isinstance(text, bytes):
        return text.decode("utf-8")
    return str(text) if text else None


def _protobuf_strings(attrs: Any, key: str) -> list[str]:
    value = _protobuf_attr(attrs, key)
    items = getattr(getattr(value, "list", None), "s", ()) if value is not None else ()
    return [item.decode("utf-8") if isinstance(item, bytes) else str(item) for item in items]


def _protobuf_int(attrs: Any, key: str) -> int | None:
    value = _protobuf_attr(attrs, key)
    raw = getattr(value, "i", None) if value is not None else None
    return int(raw) if raw is not None else None


_FX_CPU_NAME_ALIASES = {
    "mm": ("matmul",),
}


def _fx_tensor_origin(desc: Any, cpu_names: set[str]) -> tuple[str, int] | None:
    tensor_name = _protobuf_string(desc.attr, "_fx_tensor_name")
    if not tensor_name or ".OUTPUT." not in tensor_name:
        return None
    head, raw_index = tensor_name.rsplit(".OUTPUT.", 1)
    fx_name = head.split("-", 1)[0]
    if not raw_index.isdigit():
        return None
    cpu_name = next(
        (candidate for candidate in (fx_name, *_FX_CPU_NAME_ALIASES.get(fx_name, ())) if candidate in cpu_names),
        None,
    )
    if cpu_name is None:
        return None
    return cpu_name, int(raw_index)


def _load_runtime_ge_model(path: Path) -> Any:
    try:
        import torch_npu  # noqa: F401
        from google.protobuf import text_format
        from torchair._ge_concrete_graph import ge_ir_pb2
    except ImportError as exc:
        raise RuntimeError("CANN GE protobuf parser is unavailable; source CANN and import torch_npu first") from exc
    model = ge_ir_pb2.ModelDef()
    text_format.Parse(path.read_text(encoding="utf-8"), model)
    return model


def _select_runtime_ge_graph(ge_dump_dir: Path, converted: list[Path]) -> tuple[Path, Any]:
    candidates = sorted(ge_dump_dir.rglob("ge_proto_*_PreRunAfterBuild.txt"))
    if not candidates:
        candidates = sorted(ge_dump_dir.rglob("ge_proto_*_Build.txt"))
    if not candidates:
        raise RuntimeError(
            f"CANN did not emit a Build/PreRunAfterBuild graph under {ge_dump_dir}; "
            "refusing to construct topology from dump file order"
        )
    dumped_names = {op_name_from_npy(path, _npy_output_index(path)).replace("/", "_") for path in converted}
    best: tuple[int, Path, Any] | None = None
    for candidate in candidates:
        model = _load_runtime_ge_model(candidate)
        score = sum(
            _runtime_node_name(op).replace("/", "_") in dumped_names
            for graph in model.graph
            for op in graph.op
        )
        if best is None or score > best[0]:
            best = (score, candidate, model)
    if best is None or best[0] == 0:
        raise RuntimeError("no Build/PreRunAfterBuild graph node matches the CANN tensor dump")
    return best[1], best[2]


_PREFUSION_GE_STAGES = (
    "RunCustomPassAfterOriginGraphOptimize",
    "PreRunAfterOptimizeAfterStage1",
    "PreRunAfterOptimizeOriginalGraph",
    "RunCustomPassAfterBuiltinFusionPass",
    "PreRunBegin",
)


def _load_pre_run_ge_models(ge_dump_dir: Path) -> list[Any]:
    for stage in _PREFUSION_GE_STAGES:
        paths = sorted(ge_dump_dir.rglob(f"ge_proto_*_{stage}.txt"))
        if paths:
            return [_GeGraphSnapshot(source=f"CANN GE {stage}", model=_load_runtime_ge_model(path)) for path in paths]
    return []


def _ge_snapshot(value: Any) -> tuple[Any, str]:
    if isinstance(value, _GeGraphSnapshot):
        return value.model, value.source
    return value, "CANN GE PreRunBegin"


def _ge_fx_origins(
    ge_dump_dir: Path,
    cpu_names: set[str],
    models: list[Any] | None = None,
) -> dict[tuple[str, int], tuple[str, int]]:
    origins: dict[tuple[str, int], tuple[str, int]] = {}
    for snapshot in models if models is not None else _load_pre_run_ge_models(ge_dump_dir):
        model, _ = _ge_snapshot(snapshot)
        for graph in model.graph:
            for op in graph.op:
                for index, desc in enumerate(op.output_desc):
                    origin = _fx_tensor_origin(desc, cpu_names)
                    if origin is not None:
                        origins[(str(op.name), index)] = origin
    return origins


def _ge_dtype_name(desc: Any) -> str:
    raw = getattr(desc, "dtype", None)
    try:
        field = desc.DESCRIPTOR.fields_by_name["dtype"]
        value = field.enum_type.values_by_number.get(int(raw))
        if value is not None:
            return value.name
    except (AttributeError, KeyError, TypeError, ValueError):
        pass
    return str(raw) if raw is not None else ""


def _ge_tensor_summary(desc: Any, index: int) -> dict[str, Any]:
    shape = [int(dim) for dim in getattr(getattr(desc, "shape", None), "dim", ())]
    if not shape:
        meta = _protobuf_string(getattr(desc, "attr", {}), "_meta") or ""
        match = re.search(r"shape=torch\.Size\(\[([^\]]*)\]\)", meta)
        if match:
            values = [value.strip() for value in match.group(1).split(",") if value.strip()]
            if all(re.fullmatch(r"-?\d+", value) for value in values):
                shape = [int(value) for value in values]
    return {
        "index": index,
        "dtype": _ge_dtype_name(desc),
        "shape": shape,
        "format": str(getattr(desc, "layout", "") or "ND"),
    }


def _ge_output_summaries(op: Any) -> list[dict[str, Any]]:
    return [_ge_tensor_summary(desc, index) for index, desc in enumerate(op.output_desc)]


def _ge_input_summaries(op: Any) -> list[dict[str, Any]]:
    raw_inputs = list(getattr(op, "input", ()))
    inputs = []
    for index, desc in enumerate(getattr(op, "input_desc", ())):
        item = _ge_tensor_summary(desc, index)
        if index < len(raw_inputs):
            item["src"] = str(raw_inputs[index]).split(":", 1)[0].removeprefix("^")
        inputs.append(item)
    return inputs


def _origin_subgraph(origin_names: list[str], models: list[Any]) -> dict[str, Any] | None:
    if not origin_names or not models:
        return None
    best: tuple[int, dict[str, Any]] | None = None
    for snapshot in models:
        model, source = _ge_snapshot(snapshot)
        ops = [op for graph in model.graph for op in graph.op]
        by_name = {str(op.name): op for op in ops}
        selected = {name: by_name[name] for name in origin_names if name in by_name}
        if not selected:
            continue
        all_edges: list[dict[str, Any]] = []
        incoming: dict[str, list[str]] = {name: [] for name in by_name}
        outgoing: dict[str, list[str]] = {name: [] for name in by_name}
        for dst_name, op in by_name.items():
            for index, raw in enumerate(op.input):
                src_name = str(raw).split(":", 1)[0].removeprefix("^")
                if src_name not in by_name:
                    continue
                all_edges.append({"src": src_name, "dst": dst_name, "dst_input": index})
                incoming[dst_name].append(src_name)
                outgoing[src_name].append(dst_name)

        def reachable(starts: set[str], adjacency: dict[str, list[str]]) -> set[str]:
            result = set(starts)
            queue = list(starts)
            while queue:
                current = queue.pop()
                for name in adjacency.get(current, []):
                    if name in result:
                        continue
                    result.add(name)
                    queue.append(name)
            return result

        selected_names = set(selected)
        included = reachable(selected_names, outgoing) & reachable(selected_names, incoming)
        nodes = [
            {
                "name": str(op.name),
                "type": str(op.type),
                "inputs": _ge_input_summaries(op),
                "outputs": _ge_output_summaries(op),
                "origin": str(op.name) in selected_names,
            }
            for op in ops
            if str(op.name) in included
        ]
        edges = [edge for edge in all_edges if edge["src"] in included and edge["dst"] in included]
        data = {
            "source": source,
            "nodes": nodes,
            "edges": edges,
            "missing": [name for name in origin_names if name not in selected],
        }
        score = len(selected)
        if best is None or score > best[0]:
            best = (score, data)
    return best[1] if best is not None else None


def runtime_ge_graph_data(
    model: Any,
    converted: list[Path],
    cpu_names: set[str],
    source: str,
    ge_fx_origins: dict[tuple[str, int], tuple[str, int]] | None = None,
    origin_models: list[Any] | None = None,
) -> dict[str, Any]:
    dump_arrays: dict[tuple[str, int], np.ndarray] = {}
    for path in converted:
        index = _npy_output_index(path)
        name = op_name_from_npy(path, index)
        try:
            dump_arrays[(name.replace("/", "_"), index)] = np.load(path, mmap_mode="r")
        except Exception:
            continue

    ops = [op for graph in model.graph for op in graph.op]
    display_names = {str(op.name): _runtime_node_name(op) for op in ops}
    graph_ops: list[dict[str, Any]] = []
    for op in ops:
        display_name = display_names[str(op.name)]
        inputs = []
        for raw in op.input:
            internal = str(raw).split(":", 1)[0]
            if internal in display_names:
                suffix = str(raw).split(":", 1)[1] if ":" in str(raw) else "0"
                inputs.append(f"{display_names[internal]}:{suffix}")

        artifact_name = display_name.replace("/", "_")
        output_count = max(len(op.output_desc), max((idx + 1 for name, idx in dump_arrays if name == artifact_name), default=0))
        output_desc = []
        for index in range(output_count):
            desc = op.output_desc[index] if index < len(op.output_desc) else None
            array = dump_arrays.get((artifact_name, index))
            dtype = str(array.dtype) if array is not None else ""
            shape = tuple(int(dim) for dim in array.shape) if array is not None else ()
            origin_format = _protobuf_string(desc.attr, "_datadump_origin_format") if desc is not None else None
            if array is not None:
                layout = "ND"
            else:
                layout = origin_format or (str(getattr(desc, "layout", "") or "ND") if desc is not None else "ND")
            attrs = []
            origin = _fx_tensor_origin(desc, cpu_names) if desc is not None else None
            if origin is None and desc is not None and ge_fx_origins:
                ge_name = _protobuf_string(desc.attr, "_datadump_origin_name")
                ge_index = _protobuf_int(desc.attr, "_datadump_origin_output_index")
                if ge_name is not None:
                    origin = ge_fx_origins.get((ge_name, ge_index if ge_index is not None else index))
            if origin is not None:
                attrs.extend(
                    [
                        {"key": "_autofusex_cpu_origin_name", "value": {"s": origin[0]}},
                        {"key": "_autofusex_cpu_origin_output_index", "value": {"i": origin[1]}},
                    ]
                )
            output_desc.append(_desc(dtype, shape, attrs) | {"layout": layout})

        origin_ops = _protobuf_strings(op.attr, "_datadump_original_op_names")
        graph_ops.append(
            {
                "name": display_name,
                "type": str(op.type),
                "input": inputs,
                "output_desc": output_desc,
                "origin_ops": origin_ops,
                "origin_subgraph": _origin_subgraph(origin_ops, origin_models or []),
            }
        )
    stage = "PreRunAfterBuild" if "PreRunAfterBuild" in source else "Build"
    return {
        "graph": [{"op": graph_ops}],
        "metadata": {"topology": f"CANN GE {stage}", "source": source},
    }


def add_cpu_origin_mappings(graph_data: dict[str, Any], matches: dict[str, str]) -> None:
    for op in graph_data.get("graph", [{}])[0].get("op", []):
        origin = matches.get(op.get("name", ""))
        if origin is None:
            continue
        for index, desc in enumerate(op.get("output_desc", [])):
            attrs = desc.setdefault("attr", [])
            if any(item.get("key") == "_autofusex_cpu_origin_name" for item in attrs):
                continue
            attrs.extend(
                [
                    {"key": "_autofusex_cpu_origin_name", "value": {"s": origin}},
                    {"key": "_autofusex_cpu_origin_output_index", "value": {"i": index}},
                ]
            )


def _target_type(target: Any) -> str:
    text = str(target)
    if text.startswith("aten."):
        return text.removeprefix("aten.").split(".", 1)[0]
    return getattr(target, "__name__", text)


def _iter_fx_inputs(value: Any) -> list[str]:
    try:
        import torch
    except Exception:
        torch = None
    node_type = getattr(getattr(torch, "fx", None), "Node", None) if torch is not None else None
    if node_type is not None and isinstance(value, node_type):
        return [value.name]
    if isinstance(value, (list, tuple)):
        names: list[str] = []
        for item in value:
            names.extend(_iter_fx_inputs(item))
        return names
    if isinstance(value, dict):
        names = []
        for item in value.values():
            names.extend(_iter_fx_inputs(item))
        return names
    return []


def _first_tensor(value: Any, *, torch: Any) -> Any | None:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (list, tuple)):
        for item in value:
            result = _first_tensor(item, torch=torch)
            if result is not None:
                return result
    if isinstance(value, dict):
        for item in value.values():
            result = _first_tensor(item, torch=torch)
            if result is not None:
                return result
    return None


def _cpu_export_args(exported: Any, tokens: Any) -> list[Any]:
    args = []
    for spec in exported.graph_signature.input_specs:
        kind = spec.kind.name
        if kind in {"PARAMETER", "BUFFER"}:
            args.append(exported.state_dict[spec.target])
        elif kind == "USER_INPUT":
            args.append(tokens)
    return args


def _export_qwen_cpu_graph(script: str | Path, state: dict[str, Any], tokens: Any, cpu_dump: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    import torch

    module = _load_script_module(script)
    config = module.ModelConfig()
    model = module.MiniQwen(config).eval()
    model.load_state_dict(state)
    exported = torch.export.export(model, (tokens,))
    args = _cpu_export_args(exported, tokens)
    saved: dict[str, dict[str, Any]] = {}

    class CaptureInterpreter(torch.fx.Interpreter):
        def run_node(self, node):  # type: ignore[no-untyped-def]
            result = super().run_node(node)
            if node.op != "call_function":
                return result
            tensor = _first_tensor(result, torch=torch)
            if tensor is None:
                return result
            array = tensor.detach().cpu().numpy()
            np.save(cpu_dump / f"{node.name}.output.0.npy", array)
            saved[node.name] = {
                "shape": tuple(int(dim) for dim in array.shape),
                "dtype": str(array.dtype),
                "size": int(array.size),
            }
            return result

    with torch.no_grad():
        CaptureInterpreter(exported.graph_module).run(*args)

    user_input_info = {}
    user_input_tensors = [tokens]
    user_input_idx = 0
    for spec in exported.graph_signature.input_specs:
        if spec.kind.name != "USER_INPUT" or not hasattr(spec.arg, "name"):
            continue
        tensor = user_input_tensors[user_input_idx]
        user_input_idx += 1
        user_input_info[spec.arg.name] = {
            "shape": tuple(int(dim) for dim in tensor.shape),
            "dtype": str(tensor.detach().cpu().numpy().dtype),
        }
    user_inputs = {
        spec.arg.name
        for spec in exported.graph_signature.input_specs
        if spec.kind.name == "USER_INPUT" and hasattr(spec.arg, "name")
    }
    ops: list[dict[str, Any]] = [
        {
            "name": name,
            "type": "Data",
            "output_desc": [_desc(user_input_info[name]["dtype"], user_input_info[name]["shape"])]
            if name in user_input_info
            else [],
        }
        for name in sorted(user_inputs)
    ]
    included = set(user_inputs)
    for node in exported.graph_module.graph.nodes:
        if node.op != "call_function" or node.name not in saved:
            continue
        info = saved[node.name]
        inputs = []
        seen_inputs = set()
        for input_name in _iter_fx_inputs((node.args, node.kwargs)):
            if input_name not in included or input_name in seen_inputs:
                continue
            inputs.append(f"{input_name}:0")
            seen_inputs.add(input_name)
        ops.append(
            {
                "name": node.name,
                "type": _target_type(node.target),
                "input": inputs,
                "output_desc": [_desc(info["dtype"], info["shape"])],
            }
        )
        included.add(node.name)
    return {"graph": [{"op": ops}]}, saved


def _prepend_pythonpath(path: Path) -> None:
    current = os.environ.get("PYTHONPATH", "")
    items = [item for item in current.split(os.pathsep) if item]
    value = str(path)
    if value not in items:
        os.environ["PYTHONPATH"] = os.pathsep.join([value, *items])
    if value not in sys.path:
        sys.path.insert(0, value)


def _enable_ge_graph_dump(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    os.environ["DUMP_GE_GRAPH"] = "2"
    os.environ["DUMP_GRAPH_PATH"] = str(path)


def _run_npu(
    spec: ExampleSpec,
    inputs: list[Any],
    raw_dump_dir: Path,
    ge_dump_dir: Path,
    *,
    seed: int,
    device: str,
) -> Any:
    enable_autofuse_graph_dump(ge_dump_dir)
    import torch

    try:
        import inductor_npu_ext  # noqa: F401
    except Exception:
        pass
    import torch_npu  # noqa: F401
    import torchair
    from torchair.configs.compiler_config import CompilerConfig

    os.environ["TORCH_COMPILE_DEBUG"] = "1"
    os.environ["TORCHINDUCTOR_FORCE_DISABLE_CACHES"] = "1"
    _enable_ge_graph_dump(ge_dump_dir)
    torch.manual_seed(seed)
    torch.npu.set_device(device)
    config = CompilerConfig()
    config.dump_config.enable_dump = True
    config.dump_config.dump_path = str(raw_dump_dir)
    config.dump_config.dump_mode = "output"
    config.dump_config.dump_data = "tensor"
    model = _model_factory(spec).eval().to(device)
    compiled = torch.compile(
        model,
        backend=torchair.get_npu_backend(compiler_config=config),
        dynamic=False,
        fullgraph=True,
    )
    with torch.no_grad():
        out = compiled(*[tensor.to(device) for tensor in inputs])
        torch.npu.synchronize()
        return out.cpu()


def _run_qwen_npu(
    script: str | Path,
    state: dict[str, Any],
    tokens: Any,
    raw_dump_dir: Path,
    ge_dump_dir: Path,
    *,
    device: str,
) -> Any:
    enable_autofuse_graph_dump(ge_dump_dir)
    import torch

    try:
        import inductor_npu_ext  # noqa: F401
    except Exception:
        pass
    import torch_npu  # noqa: F401
    import torchair
    from torchair.configs.compiler_config import CompilerConfig

    module = _load_script_module(script)
    config = module.ModelConfig()
    model = module.MiniQwen(config).eval().to(device)
    model.load_state_dict(state)
    torch.npu.set_device(device)
    _enable_ge_graph_dump(ge_dump_dir)
    compiler_config = CompilerConfig()
    compiler_config.dump_config.enable_dump = True
    compiler_config.dump_config.dump_path = str(raw_dump_dir)
    compiler_config.dump_config.dump_mode = "output"
    compiler_config.dump_config.dump_data = "tensor"
    compiled = torch.compile(
        model,
        backend=torchair.get_npu_backend(compiler_config=compiler_config),
        dynamic=False,
        fullgraph=True,
    )
    with torch.no_grad():
        out = compiled(tokens.to(device))
        torch.npu.synchronize()
        return out.cpu()


def _run_qwen_example(
    *,
    script: str | Path,
    out_dir: str | Path,
    html_path: str | Path | None,
    cann_root: str | Path | None,
    device: str,
    seed: int,
    rtol: float | None,
    atol: float | None,
) -> dict[str, Any]:
    spec = example_from_script(script)
    root = Path(out_dir).resolve()
    _clean_dir(root)
    cpu_dump = root / "cpu_dump"
    raw_npu_dump = root / "npu_dump_raw"
    npu_dump = root / "npu_dump_npy"
    ge_dump = root / "ge_dump"
    cpu_dump.mkdir(parents=True, exist_ok=True)
    raw_npu_dump.mkdir(parents=True, exist_ok=True)
    npu_dump.mkdir(parents=True, exist_ok=True)
    ge_dump.mkdir(parents=True, exist_ok=True)
    _enable_ge_graph_dump(ge_dump)

    import torch

    for candidate in _candidate_cann_roots(cann_root):
        site_packages = candidate / "python/site-packages"
        if site_packages.is_dir():
            _prepend_pythonpath(site_packages)
            break

    torch.manual_seed(seed)
    module = _load_script_module(script)
    config = module.ModelConfig()
    model = module.MiniQwen(config).eval()
    state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    tokens = torch.randint(0, config.vocab_size, spec.input_shape)
    cpu_graph_data, cpu_outputs = _export_qwen_cpu_graph(script, state, tokens, cpu_dump)
    npu_out = _run_qwen_npu(script, state, tokens, raw_npu_dump, ge_dump, device=device)
    np.save(npu_dump / "model_output.npy", npu_out.numpy())
    converted = convert_cann_dump_tree_to_npy(raw_npu_dump, npu_dump, find_msaccucmp(cann_root))
    runtime_graph_path, runtime_model = _select_runtime_ge_graph(ge_dump, converted)
    origin_models = _load_pre_run_ge_models(ge_dump)
    ge_fx_origins = _ge_fx_origins(ge_dump, set(cpu_outputs), origin_models)
    npu_graph_data = runtime_ge_graph_data(
        runtime_model,
        converted,
        set(cpu_outputs),
        str(runtime_graph_path),
        ge_fx_origins,
        origin_models,
    )

    cpu_graph = root / "cpu_graph.json"
    npu_graph = root / "npu_graph.json"
    _save_json(cpu_graph, cpu_graph_data)
    _save_json(npu_graph, npu_graph_data)
    shutil.copy2(runtime_graph_path, root / "runtime_ge_graph.txt")
    cpu_manifest = root / "cpu_manifest.json"
    npu_manifest = root / "npu_manifest.json"
    build_manifest(side="cpu", graph_path=cpu_graph, dump_dir=cpu_dump, out_path=cpu_manifest, model_name=spec.key)
    build_manifest(side="npu", graph_path=npu_graph, dump_dir=npu_dump, out_path=npu_manifest, model_name=spec.key)

    report = compare_manifests(cpu_manifest, npu_manifest, rtol=rtol, atol=atol)
    attach_autofuse_graphs(report, ge_dump)
    save_report(report, root / "report.json", root / "report.csv")
    html = Path(html_path).resolve() if html_path else root / "report.html"
    write_report_html(report, html, title=f"Autofuse {spec.key} Accuracy Report")
    summary = {
        "example": spec.key,
        "script": str(Path(script).resolve()),
        "out_dir": str(root),
        "html": str(html),
        "fused_op": "qwen_whole_graph",
        "converted_dump": str(npu_dump),
        "raw_npu_dump": str(raw_npu_dump),
        "cpu_nodes": len(cpu_graph_data["graph"][0]["op"]),
        "npu_nodes": len(npu_graph_data["graph"][0]["op"]),
        "matched_nodes": report["summary"].get("compared", report["summary"]["total"]),
        "unmatched_nodes": report["summary"].get("unmatched", 0),
        "coverage": report["summary"].get("coverage", 1.0),
        "topology": npu_graph_data["metadata"]["topology"],
        "runtime_ge_graph": str(root / "runtime_ge_graph.txt"),
        "passed": report["summary"].get("status") == "PASS",
        "summary": report["summary"],
    }
    _save_json(root / "run_summary.json", summary)
    return summary


def run_npu_example(
    *,
    script: str | Path,
    out_dir: str | Path,
    html_path: str | Path | None = None,
    cann_root: str | Path | None = None,
    device: str = "npu:0",
    seed: int = 20260714,
    rtol: float | None = None,
    atol: float | None = None,
) -> dict[str, Any]:
    del seed
    from autofusex.torch_model import run_torch_model

    return run_torch_model(
        script=script,
        out_dir=out_dir,
        html_path=html_path,
        cann_root=cann_root,
        device=device,
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
        f"topology: {summary.get('topology', '-')}"
    )

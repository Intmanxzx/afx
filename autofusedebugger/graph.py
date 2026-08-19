from __future__ import annotations

import glob
import re
from pathlib import Path
from typing import Any

from autofusedebugger.artifact_io import SCHEMA, read_json, write_json

def normalize_op_name(name: str) -> str:
    return name.replace("/", "_")


def get_ops(graph_data: dict[str, Any]) -> list[dict[str, Any]]:
    graphs = graph_data.get("graph", [])
    if not graphs:
        return []
    ops = graphs[0].get("op", [])
    return ops if isinstance(ops, list) else []


def parse_origin_attrs(attrs: list[dict[str, Any]]) -> tuple[str | None, int | None]:
    origin_name = None
    origin_index = None
    for attr in attrs:
        key = attr.get("key", "")
        value = attr.get("value", {})
        if key in {"_autofusedebugger_cpu_origin_name", "_datadump_origin_name"}:
            origin_name = value.get("s")
        elif key in {"_autofusedebugger_cpu_origin_output_index", "_datadump_origin_output_index"}:
            raw = value.get("i")
            origin_index = int(raw) if raw is not None else None
    return origin_name, origin_index


def extract_fusions(ops: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fusions: list[dict[str, Any]] = []
    for op in ops:
        fused_op = op.get("name", "")
        for idx, desc in enumerate(op.get("output_desc", [])):
            origin_op, origin_index = parse_origin_attrs(desc.get("attr", []))
            if origin_op is None or origin_index is None:
                continue
            fusions.append(
                {
                    "fused_op": fused_op,
                    "fused_index": idx,
                    "origin_op": origin_op,
                    "origin_index": origin_index,
                }
            )
    return fusions


def _input_to_op_name(value: Any, known_names: set[str]) -> str | None:
    if isinstance(value, dict):
        for key in ("name", "src_name", "op_name", "node_name"):
            raw = value.get(key)
            if isinstance(raw, str) and raw:
                return raw
        for key in ("input", "src", "value"):
            resolved = _input_to_op_name(value.get(key), known_names)
            if resolved:
                return resolved
        return None
    if not isinstance(value, str) or not value:
        return None
    text = value.strip()
    if text in known_names:
        return text
    for sep in (":", ";", ","):
        head = text.split(sep, 1)[0]
        if head in known_names:
            return head
    for name in sorted(known_names, key=len, reverse=True):
        if text == name or text.startswith(name + ":") or text.startswith(name + "/"):
            return name
    return None


def extract_edges(ops: list[dict[str, Any]]) -> list[dict[str, Any]]:
    known_names = {op.get("name", "") for op in ops if op.get("name")}
    edges: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int]] = set()
    for op in ops:
        dst = op.get("name", "")
        if not dst:
            continue
        raw_inputs = op.get("input", [])
        if isinstance(raw_inputs, dict):
            raw_inputs = list(raw_inputs.values())
        if not isinstance(raw_inputs, list):
            raw_inputs = [raw_inputs]
        for index, raw in enumerate(raw_inputs):
            src = _input_to_op_name(raw, known_names)
            if not src or src == dst:
                continue
            key = (src, dst, index)
            if key in seen:
                continue
            seen.add(key)
            edge = {"src": src, "dst": dst, "dst_input": index}
            input_desc = op.get("input_desc", [])
            if index < len(input_desc):
                desc = input_desc[index]
                edge["format"] = output_format(desc)
                edge["dtype"] = desc.get("data_type") or desc.get("dtype")
                edge["shape"] = desc.get("shape", {}).get("dim") if isinstance(desc.get("shape"), dict) else None
            edges.append(edge)
    return edges


def output_format(desc: dict[str, Any]) -> str | None:
    return desc.get("layout") or desc.get("format")


def _candidate_patterns(dump_dir: Path, op_name: str, index: int) -> list[str]:
    normalized = normalize_op_name(op_name)
    return [
        str(dump_dir / f"{normalized}.output.{index}.npy"),
        str(dump_dir / f"{normalized}_{index}.npy"),
        str(dump_dir / f"{normalized}.npy"),
        str(dump_dir / f"{normalized}.*.output.{index}.npy"),
        str(dump_dir / f"*.{normalized}.*.output.{index}.npy"),
    ]


def find_output_file(dump_dir: Path, op_name: str, index: int) -> Path | None:
    seen: set[Path] = set()
    for pattern in _candidate_patterns(dump_dir, op_name, index):
        matches = [Path(p) for p in glob.glob(pattern)]
        for match in sorted(matches):
            if match in seen:
                continue
            seen.add(match)
            return match
    return None


def build_manifest(
    *,
    side: str,
    graph_path: str | Path,
    dump_dir: str | Path,
    out_path: str | Path,
    model_name: str | None = None,
) -> dict[str, Any]:
    if side not in {"cpu", "npu", "npu_eager", "npu_autofuse"}:
        raise ValueError("side must be 'cpu', 'npu_eager', or 'npu_autofuse'")
    graph_file = Path(graph_path).resolve()
    root = Path(dump_dir).resolve()
    if not graph_file.is_file():
        raise FileNotFoundError(str(graph_file))
    if not root.is_dir():
        raise NotADirectoryError(str(root))

    graph_data = read_json(graph_file)
    ops = get_ops(graph_data)
    nodes: list[dict[str, Any]] = []
    for op in ops:
        name = op.get("name", "")
        node = {
            "name": name,
            "type": op.get("type", ""),
            "outputs": [],
            "origin_ops": op.get("origin_ops", []),
            "original_aten_types": op.get("original_aten_types", []),
            "origin_subgraph": op.get("origin_subgraph"),
            "unfused_reasons": op.get("unfused_reasons", []),
            "provenance_conflicts": op.get("provenance_conflicts", []),
        }
        outputs = op.get("output_desc", [])
        for idx, desc in enumerate(outputs):
            file_path = find_output_file(root, name, idx)
            rel_path = str(file_path.relative_to(root)) if file_path is not None else None
            node["outputs"].append(
                {
                    "index": idx,
                    "path": rel_path,
                    "format": output_format(desc),
                    "dtype": desc.get("data_type") or desc.get("dtype"),
                    "shape": desc.get("shape", {}).get("dim") if isinstance(desc.get("shape"), dict) else None,
                }
            )
        nodes.append(node)

    manifest = {
        "schema": SCHEMA,
        "side": side,
        "model": {
            "name": model_name or graph_file.stem,
            "graph": str(graph_file),
            **graph_data.get("metadata", {}),
        },
        "artifacts": {"root": str(root)},
        "nodes": nodes,
        "edges": extract_edges(ops),
        "fusions": extract_fusions(ops) if side in {"npu", "npu_autofuse"} else [],
    }
    write_json(out_path, manifest)
    return manifest
